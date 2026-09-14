# GWRCK Spatial Model

Web application for spatial prediction of soil organic carbon using:

GWR → GWRC → Kriging of GWRC residuals → GWRCK

The model uses an adaptive bisquare kernel and local condition number correction through local ridge regression.

## Predictor variables

- ELEV
- Curvatu
- Slope
- HillSha
- TEMP
- PRES
- EVI
- BSI

## Target variable

SOC

## Coordinate system

EPSG:32717

## Raster resolution

30 m

## Final model

SOC_GWRCK = SOC_GWRC + Kriged_Residual_GWRC
