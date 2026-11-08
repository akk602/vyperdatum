"""
VR BAG XML metadata: NUL-terminated metadata, and the corner points written back
after a transformation.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from vyperdatum.drivers import vrbag
from vyperdatum.transformer import Transformer

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


def test_update_vr_refinements_keeps_corner_axis_order(tmp_path, monkeypatch):
    """
    After a real horizontal transformation (UTM 17N on WGS 84 -> on NAD83), the corners
    written back must still be easting,northing and close to the originals. They used to
    be passed to PROJ as northing,easting, which swapped them.
    """
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, nul_terminated=False)
    fake_ds = SimpleNamespace(GetGeoTransform=lambda: (X1, 50.0, 0, Y2, 0, -50.0))
    monkeypatch.setattr(vrbag, "gdal", SimpleNamespace(Open=lambda fname: fake_ds))
    crs_from, crs_to = "EPSG:32617", "EPSG:26917"
    tf = Transformer(crs_from=crs_from, crs_to=crs_to,
                     steps=[{"crs_from": crs_from, "crs_to": crs_to, "v_shift": False}])
    vrbag.update_vr_refinements(str(fname), index=[0], arr=[np.array([-1.0, -2.0, -3.0, -4.0])],
                                tf=tf)
    x1, y1, x2, y2 = vrbag.corner_points(str(fname))
    for new, old in [(x1, X1), (y1, Y1), (x2, X2), (y2, Y2)]:
        assert new == pytest.approx(old, abs=5.0), (x1, y1, x2, y2)
