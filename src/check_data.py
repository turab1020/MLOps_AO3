"""Quick sanity check: print shape, dtype and value range of every saved array."""
import glob
import os

import numpy as np

for folder in (os.path.join("data", "raw"), os.path.join("data", "processed")):
    for path in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        arr = np.load(path)
        print(f"{path:32} shape={str(arr.shape):18} dtype={str(arr.dtype):8} min={arr.min():.3f} max={arr.max():.3f}")
