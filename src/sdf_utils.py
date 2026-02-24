import os
import sys
import numpy as np

# Make sure we can import dexnet/meshpy when this module is used standalone
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import meshpy.sdf_file as sdf_file


def _get_dims(sdf3d):
    if hasattr(sdf3d, "dims"):
        return tuple(int(x) for x in sdf3d.dims)
    if hasattr(sdf3d, "dimensions"):
        return tuple(int(x) for x in sdf3d.dimensions)
    raise AttributeError("Sdf3D has no dims/dimensions attribute")


def load_sdf(proc_sdf_path):
    """Load an existing *_proc.sdf file into a meshpy Sdf3D object."""
    if not os.path.exists(proc_sdf_path):
        raise FileNotFoundError(proc_sdf_path)
    sf = sdf_file.SdfFile(proc_sdf_path)
    return sf.read()


def sdf_world_trilinear(sdf3d, p_world):
    """
    Query SDF at a WORLD (x,y,z) point using trilinear interpolation.
    Returns float if inside grid, else None.
    """
    origin = np.array(sdf3d.origin, dtype=float)
    res = float(sdf3d.resolution)
    dims = np.array(_get_dims(sdf3d), dtype=int)

    g = (np.array(p_world, dtype=float) - origin) / res  # grid coords (float)

    if np.any(g < 0) or np.any(g > (dims - 1)):
        return None

    i0 = np.floor(g).astype(int)
    d = g - i0
    i1 = np.minimum(i0 + 1, dims - 1)

    def V(ii, jj, kk):
        return float(sdf3d[int(ii), int(jj), int(kk)])

    x0, y0, z0 = i0
    x1, y1, z1 = i1

    c000 = V(x0, y0, z0)
    c100 = V(x1, y0, z0)
    c010 = V(x0, y1, z0)
    c110 = V(x1, y1, z0)
    c001 = V(x0, y0, z1)
    c101 = V(x1, y0, z1)
    c011 = V(x0, y1, z1)
    c111 = V(x1, y1, z1)

    cx00 = c000 * (1 - d[0]) + c100 * d[0]
    cx10 = c010 * (1 - d[0]) + c110 * d[0]
    cx01 = c001 * (1 - d[0]) + c101 * d[0]
    cx11 = c011 * (1 - d[0]) + c111 * d[0]
    cxy0 = cx00 * (1 - d[1]) + cx10 * d[1]
    cxy1 = cx01 * (1 - d[1]) + cx11 * d[1]
    cxyz = cxy0 * (1 - d[2]) + cxy1 * d[2]
    return float(cxyz)