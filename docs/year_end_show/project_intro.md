# Project Introduction

**Detecting Backup Power from Space During Disasters**

When a hurricane knocks out the grid, hospitals, airports, and fire stations
stay lit on backup generators — yet no public registry records where these
generators are or whether they actually work. We ask whether NASA's VIIRS
Black Marble nighttime-light satellite (500 m resolution, daily revisit) can
detect the brightness signature of backup power running during major outages.

Across **25 disasters** in **17 U.S. states, Puerto Rico, and Türkiye
(2016 – 2023)**, we extract a panel of ~33,700 pixels, treat critical
infrastructure (hospitals, airports, fire stations, police, power plants) as
proxy labels, and train a tree-based model on **purely temporal nighttime-light
features** — no spatial proximity to facilities — so its predictions reflect
satellite signal alone. Across all events the model achieves **LOEO
AUC = 0.704**; against actual generator-permit records in Miami-Dade,
**83 % of commercial installations** score above the event-wide median
probability (vs. 14 % of residential), confirming the resolution boundary.
At zip-code scale, model probability is positively associated with real
outage severity (standardized β ≈ 0.08, p = 0.012).

**Authors:** Zhiyuan Zhao · Qiushi Yu  ·  University of Pennsylvania
