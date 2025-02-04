# Interpolate-values-across-a-gridded-geographic-entity
A geographic object with overlaid grid (hexbins) contains missing values, we interpolate to impute values.

## Overview
We want to understand the complete picture of network propagation within a building when aggregating data points into hexbins​.
However, when aggregating crowdsourced data, there are hexbins that contain no data​ (first image below).
We want to estimate mean RSRP values for missing hexbin (second image below).

<img width="314" alt="image" src="https://github.com/user-attachments/assets/b31ef4dd-d9d5-4544-b44d-0f2a379c4197" />

<img width="321" alt="image" src="https://github.com/user-attachments/assets/3d38266e-1e9d-49c3-ac5e-98500c213f43" />

## Data requirements
For commercial reasons, the input data cannot be shared. The process uses building shape data (open source from OpenStreetMap) and commercially available mobile sensor data.

## Notes
Two versions are included:
1 - Where a bespoke hexbin grid for the building is created 
2 - Where Uber's H3 package is used to supply a hexbin grid
