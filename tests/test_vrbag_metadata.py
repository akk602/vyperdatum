"""
VR BAG XML metadata.
"""
import h5py
import numpy as np
import pytest
from vyperdatum.drivers import vrbag

# UTM 17N corners (easting, northing), as a VR BAG's metadata stores them.
X1, Y1, X2, Y2 = 400000.0, 3500000.0, 400100.0, 3500100.0

METADATA = f"""<?xml version="1.0" encoding="UTF-8"?>
<gmi:MI_Metadata xmlns:gmi="http://www.isotc211.org/2005/gmi"
                 xmlns:gml="http://www.opengis.net/gml/3.2"
                 xmlns:gco="http://www.isotc211.org/2005/gco">
  <gml:coordinates>{X1},{Y1} {X2},{Y2}</gml:coordinates>
  {"".join(f"<gco:CharacterString>s{i}</gco:CharacterString>" for i in range(10))}
</gmi:MI_Metadata>
"""


def _write_fake_vrbag(path, metadata=METADATA, nul_terminated=True):
    raw = metadata.encode() + (b"\x00" if nul_terminated else b"")
    meta_dtype = np.dtype([("index", "<u4"), ("dimensions_x", "<u4"), ("dimensions_y", "<u4"),
                           ("resolution_x", "<f4"), ("resolution_y", "<f4"),
                           ("sw_corner_x", "<f4"), ("sw_corner_y", "<f4")])
    ref_dtype = np.dtype([("depth", "<f4"), ("depth_uncrt", "<f4")])
    with h5py.File(path, "w") as bag:
        root = bag.create_group("BAG_root")
        root.create_dataset("metadata", data=np.frombuffer(raw, dtype="S1"))
        root.create_dataset("elevation", data=np.full((2, 2), -5.0, dtype=np.float32))
        root.create_dataset("varres_metadata", data=np.zeros((1, 1), dtype=meta_dtype))
        refs = np.zeros((1, 4), dtype=ref_dtype)
        refs["depth"] = [-1.0, -2.0, -3.0, -4.0]
        root.create_dataset("varres_refinements", data=refs)


@pytest.mark.parametrize("nul_terminated", [True, False])
def test_corner_points_with_nul_terminated_metadata(nul_terminated, tmp_path):
    """Some writers end the metadata with a NUL byte, which lxml used to reject."""
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, nul_terminated=nul_terminated)
    assert vrbag.corner_points(str(fname)) == (X1, Y1, X2, Y2)

