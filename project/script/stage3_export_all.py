"""
stage3_export_all.py — Submit all 16 events to GEE with corrected band name.
Run after cancelling old tasks.
"""
import ee
import os
import sys
import time
import pandas as pd
from stage3_events import EVENTS

ee.Initialize(project='deductive-tempo-485113-n8')
print("GEE OK", flush=True)

NTL_BAND = 'Gap_Filled_DNB_BRDF_Corrected_NTL'
VNP46A2 = 'NASA/VIIRS/002/VNP46A2'

def export_one(event_id, date_str, period):
    cfg = EVENTS[event_id]
    roi = ee.Geometry.Rectangle(cfg['bounds'])
    date_ee = ee.Date(date_str)
    img = ee.Image(
        ee.ImageCollection(VNP46A2)
        .filterBounds(roi)
        .filterDate(date_ee, date_ee.advance(1, 'day'))
        .first()
    )
    ntl = img.select(NTL_BAND).multiply(0.1).clip(roi)
    fname = f"{cfg['drive_root']}_{period}_{date_str}"
    folder = f"{cfg['drive_root']}-VNP46A2-{period}"
    task = ee.batch.Export.image.toDrive(
        image=ntl, description=fname, folder=folder,
        fileNamePrefix=fname, region=roi, scale=500,
        crs=cfg['metric_crs'], maxPixels=1e8,
    )
    task.start()
    return fname

data_dir = os.path.join(os.path.dirname(__file__), 'data')
total = 0

for eid in EVENTS:
    cfg = EVENTS[eid]
    csv = os.path.join(data_dir, f"{eid}_cloud_screening.csv")
    if not os.path.exists(csv):
        print(f"SKIP {eid}: no screening", flush=True)
        continue
    df = pd.read_csv(csv)
    usable = df[df['usable']]
    pre = usable[(usable['date'] >= cfg['pre_start']) & (usable['date'] <= cfg['pre_end'])]['date'].tolist()
    post = usable[(usable['date'] >= cfg['post_start']) & (usable['date'] <= cfg['post_end'])]['date'].tolist()

    print(f"\n{eid}: {len(pre)} pre + {len(post)} post", flush=True)
    for d in pre:
        export_one(eid, d, 'pre')
        total += 1
    for d in post:
        export_one(eid, d, 'post')
        total += 1
    print(f"  submitted {len(pre)+len(post)} tasks (running total: {total})", flush=True)
    time.sleep(1)

print(f"\nDONE: {total} tasks submitted to Google Drive", flush=True)
