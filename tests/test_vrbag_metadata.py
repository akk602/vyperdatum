"""
VR BAG XML metadata and axis order: NUL-terminated metadata, the corner points and WKTs
written back after a transformation, and grid points of geographic VR BAGs.
"""
from io import BytesIO
from types import SimpleNamespace

import h5py
import numpy as np
import pyproj as pp
import pytest
from lxml import etree
from vyperdatum.drivers import vrbag
from vyperdatum.transformer import Transformer

# NAD83 / UTM 10N corners (easting, northing), as a VR BAG's metadata stores them.
X1, Y1, X2, Y2 = 500000.0, 4000000.0, 500100.0, 4000100.0
GMD = "http://www.isotc211.org/2005/gmd"
GCO = "http://www.isotc211.org/2005/gco"


def _metadata(corners=f"{X1},{Y1} {X2},{Y2}", reference_systems=True):
    """
    Metadata laid out like GDAL's BAG metadata: five other gco:CharacterString elements
    come before the horizontal reference system's code.
    """
    def text(tag, value):
        return f"<gmd:{tag}><gco:CharacterString>{value}</gco:CharacterString></gmd:{tag}>"

    def reference_system(wkt):
        return ("<gmd:referenceSystemInfo><gmd:MD_ReferenceSystem>"
                "<gmd:referenceSystemIdentifier><gmd:RS_Identifier>"
                f"{text('code', wkt)}{text('codeSpace', 'WKT')}"
                "</gmd:RS_Identifier></gmd:referenceSystemIdentifier>"
                "</gmd:MD_ReferenceSystem></gmd:referenceSystemInfo>")

    systems = (reference_system(pp.CRS("EPSG:26910").to_wkt(version="WKT1_GDAL"))
               + reference_system('VERT_CS["MLLW",VERT_DATUM["MLLW",2000]]')
               if reference_systems else "")
    return (f'<gmi:MI_Metadata xmlns:gmi="http://www.isotc211.org/2005/gmi" '
            f'xmlns:gmd="{GMD}" xmlns:gco="{GCO}" xmlns:gml="http://www.opengis.net/gml/3.2">'
            f"{text('fileIdentifier', '')}"
            f"<gmd:contact>{text('individualName', 'unknown')}"
            f"{text('positionName', 'unknown')}</gmd:contact>"
            f"{text('metadataStandardName', '')}{text('metadataStandardVersion', '')}"
            f"{systems}"
            f"{text('abstract', 'Survey abstract')}"
            f"<gml:coordinates>{corners}</gml:coordinates>"
            "</gmi:MI_Metadata>")


def _write_fake_vrbag(path, metadata, suffix=b"\x00", elevation=None):
    raw = metadata.encode() + suffix
    meta_dtype = np.dtype([("index", "<u4"), ("dimensions_x", "<u4"), ("dimensions_y", "<u4"),
                           ("resolution_x", "<f4"), ("resolution_y", "<f4"),
                           ("sw_corner_x", "<f4"), ("sw_corner_y", "<f4")])
    ref_dtype = np.dtype([("depth", "<f4"), ("depth_uncrt", "<f4")])
    if elevation is None:
        elevation = np.full((2, 2), -5.0, dtype=np.float32)
    with h5py.File(path, "w") as bag:
        root = bag.create_group("BAG_root")
        root.create_dataset("metadata", data=np.frombuffer(raw, dtype="S1"))
        root.create_dataset("elevation", data=elevation)
        root.create_dataset("varres_metadata", data=np.zeros((1, 1), dtype=meta_dtype))
        refs = np.zeros((1, 4), dtype=ref_dtype)
        refs["depth"] = [-1.0, -2.0, -3.0, -4.0]
        root.create_dataset("varres_refinements", data=refs)


def _read_metadata(fname):
    with h5py.File(fname) as bag:
        return etree.parse(BytesIO(np.asarray(bag["BAG_root/metadata"][()]).tobytes()))


def _stub_gdal(monkeypatch, geotransform):
    fake_ds = SimpleNamespace(GetGeoTransform=lambda: geotransform)
    monkeypatch.setattr(vrbag, "gdal", SimpleNamespace(Open=lambda fname: fake_ds))


@pytest.mark.parametrize("suffix", [b"", b"\x00", b"\x00\x00\x00", b"\x00\n"],
                         ids=["no NUL", "NUL", "several NULs", "NUL then newline"])
def test_corner_points_with_nul_terminated_metadata(suffix, tmp_path):
    """GDAL ends the metadata with a NUL byte, which lxml used to reject."""
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, _metadata(), suffix=suffix)
    assert vrbag.corner_points(str(fname)) == (X1, Y1, X2, Y2)


def test_update_vr_refinements_writes_corners_and_wkt(tmp_path, monkeypatch):
    """
    After a transformation that moves the points (UTM 10N -> 11N), the corners written back
    are easting,northing and match PROJ, and the new WKT replaces the horizontal reference
    system's code without touching other fields.
    """
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, _metadata())
    _stub_gdal(monkeypatch, (X1, 50.0, 0, Y2, 0, -50.0))
    crs_from, crs_to = "EPSG:26910", "EPSG:26911"
    tf = Transformer(crs_from=crs_from, crs_to=crs_to,
                     steps=[{"crs_from": crs_from, "crs_to": crs_to, "v_shift": False}])
    vrbag.update_vr_refinements(str(fname), index=[0], arr=[np.array([-1.0, -2.0, -3.0, -4.0])],
                                tf=tf)

    expected = pp.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    ex1, ey1 = expected.transform(X1, Y1)
    ex2, ey2 = expected.transform(X2, Y2)
    assert vrbag.corner_points(str(fname)) == pytest.approx((ex1, ey1, ex2, ey2), abs=1e-6)

    root = _read_metadata(fname).getroot()
    codes = [e[0].text for e in root.iter(f"{{{GMD}}}code")]
    code_spaces = [e[0].text for e in root.iter(f"{{{GMD}}}codeSpace")]
    assert "UTM zone 11N" in codes[0][:80]
    assert code_spaces == ["WKT", "WKT"]
    assert root.find(f".//{{{GMD}}}abstract")[0].text == "Survey abstract"


def test_update_vr_refinements_rejects_metadata_without_reference_systems(tmp_path,
                                                                          monkeypatch):
    """Metadata the WKT can't be written to fails before anything in the file changes."""
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, _metadata(reference_systems=False))
    _stub_gdal(monkeypatch, (X1, 50.0, 0, Y2, 0, -50.0))
    tf = Transformer(crs_from="EPSG:26910", crs_to="EPSG:26911",
                     steps=[{"crs_from": "EPSG:26910", "crs_to": "EPSG:26911",
                             "v_shift": False}])
    with h5py.File(fname) as bag:
        before = np.array(bag["BAG_root/varres_refinements"][0]["depth"])
    with pytest.raises(ValueError, match="reference systems"):
        vrbag.update_vr_refinements(str(fname), index=[0], arr=[np.array([9.0] * 4)], tf=tf)
    with h5py.File(fname) as bag:
        np.testing.assert_array_equal(bag["BAG_root/varres_refinements"][0]["depth"], before)


def test_base_grid_of_geographic_vrbag_is_transformed(tmp_path, monkeypatch):
    """
    Grid points are longitude, latitude. With the CRS's own axis order (latitude first)
    they used to reach PROJ swapped, so a geographic VR BAG's depths were not transformed.
    NAD83(2011) ellipsoid heights near Norfolk, VA -> MLLW (uses the MLLW grid).
    """
    fname = tmp_path / "vr.bag"
    _write_fake_vrbag(fname, _metadata(corners="-76.2,37.12 -76.19,37.13"),
                      elevation=np.zeros((2, 2), dtype=np.float32))
    _stub_gdal(monkeypatch, (-76.2, 0.005, 0, 37.13, 0, -0.005))
    tf = Transformer(crs_from="EPSG:6319", crs_to="EPSG:6318+NOAA:98",
                     steps=[{"crs_from": "EPSG:6319", "crs_to": "EPSG:6318+NOAA:98",
                             "v_shift": True}])
    zt = vrbag.base_grid_point_transform(str(fname), tf=tf,
                                         nodata_value=vrbag.vrb_enum.NDV_REF.value)
    assert np.all(np.isfinite(zt))
    assert np.all((zt > 30.0) & (zt < 45.0)), zt
