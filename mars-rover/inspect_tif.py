import sys
import rasterio
import numpy as np

path = sys.argv[1]
with rasterio.open(path) as s:
    a = s.read(1).astype(np.float64)
    print("File:", path)
    print("Shape (rows, cols):", s.shape, "| bands:", s.count)
    print("Data type:", s.dtypes[0])
    print("Pixel size:", s.res, "| CRS:", s.crs)
    print("Nodata value:", s.nodata)
    ok = np.isfinite(a) & (a != s.nodata)
    print("Min / max / mean:", a[ok].min(), a[ok].max(), a[ok].mean())