# Image Captions · Year End Show

## 01_hero_ntl_before_after.png — *When the Lights Go Out*
Hurricane Maria (Sep 20, 2017) · San Juan, Puerto Rico.
Mean nighttime light brightness before and after landfall, captured by the NASA VIIRS Black Marble (VNP46A2) sensor at 500 m resolution. The before image shows a fully lit metropolitan area; the after image shows months of widespread darkness with isolated bright pockets — the ones that retained electricity through backup power. These persistent bright pockets are the signal our project detects.

## 02_facility_signal_miami.png — *Where Backup Power Runs*
Hurricane Irma (Sep 2017) · Miami-Dade County. The heatmap is the Production Model's per-pixel predicted probability of backup-power activity. Overlaid are 148 standalone-generator permits filed with the county: 30 commercial (yellow diamonds) and 118 residential (orange dots). 83% of permitted commercial generator locations score above the event-wide median probability — vs. only 14% of residential permits — confirming the 500 m sensor resolves institutional-scale backup behavior but not household units.

## 03_probability_heatmap.png — *Predicting Backup-Power Activity from Space*
Three events spanning hurricane, urban, coastal-rural, and international contexts: Hurricane Maria (San Juan), Hurricane Ian (Fort Myers), and the 2023 Türkiye Earthquake (Hatay). The Production Model uses only nighttime light temporal patterns — no facility-location features — and achieves Leave-One-Event-Out AUC = 0.704 across all 25 disasters in the panel.

## 04_national_coverage.png — *A National-Scale Test*
The geographic spread of the 25 disasters: 18 hurricanes, 2 earthquakes, 3 winter storms, 2 severe-storm events, plus a derecho and an ice storm, across 17 U.S. states, Puerto Rico, and Türkiye (2016 – 2023). The side panel summarizes the closed-loop validation: the model's predicted backup-power probability is positively associated with EAGLE-I-recorded outage severity (standardized β ≈ 0.08, p = 0.012) — consistent with the natural deployment direction that backup power gets installed where outages strike most.
