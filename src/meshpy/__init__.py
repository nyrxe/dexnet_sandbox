# Minimal meshpy init for sandbox (Python 3)
# Import only what Dex-Net mesh_processor needs.
from .obj_file import ObjFile
from .sdf_file import SdfFile
from .stp_file import StablePoseFile

__all__ = ['ObjFile', 'SdfFile', 'StablePoseFile']
