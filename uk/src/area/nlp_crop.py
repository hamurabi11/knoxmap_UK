"""Cut an area out of an EA National LiDAR Programme point cloud tile.

    python nlp_crop.py <tile.laz>[,<tile.laz>...] <out.npz> <e0> <n0> <e1> <n1>

Several tiles (comma-separated) for a box across a tile edge.

Keeps x, y, z (BNG metres), intensity, return number, number of returns and
class (1 unclassified, 2 ground, 3-5 vegetation, 6 building, 7 noise, 9 water)
so later steps don't re-read the 200 MB tile.
"""
import sys

import laspy
import numpy as np

CHUNK = 5_000_000


def main() -> int:
    src, out = sys.argv[1:3]
    e0, n0, e1, n1 = map(float, sys.argv[3:7])
    keep = {k: [] for k in ("x", "y", "z", "intensity", "ret", "nret", "cls")}
    for tile in src.split(","):
        with laspy.open(tile) as f:
            for pts in f.chunk_iterator(CHUNK):
                x, y = np.asarray(pts.x), np.asarray(pts.y)
                m = (x >= e0) & (x < e1) & (y >= n0) & (y < n1)
                if not m.any():
                    continue
                keep["x"].append(x[m])
                keep["y"].append(y[m])
                keep["z"].append(np.asarray(pts.z)[m])
                keep["intensity"].append(np.asarray(pts.intensity)[m])
                keep["ret"].append(np.asarray(pts.return_number)[m])
                keep["nret"].append(np.asarray(pts.number_of_returns)[m])
                keep["cls"].append(np.asarray(pts.classification)[m])
    arrays = {k: np.concatenate(v) for k, v in keep.items()}
    np.savez_compressed(out, box=np.array([e0, n0, e1, n1]), **arrays)
    n = len(arrays["x"])
    area = (e1 - e0) * (n1 - n0)
    cls, counts = np.unique(arrays["cls"], return_counts=True)
    print(f"wrote {out}: {n} points, {n / area:.2f} per m2")
    print("  classes:", dict(zip(cls.tolist(), counts.tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
