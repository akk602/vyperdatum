"""
Automatic steps for geographic and geocentric horizontal CRSs that are not NAD83.

WGS 84 (EPSG:4326, EPSG:4979) and CRSs that are their own geodetic CRS (WGS 84 (G2139),
ITRF2014, ITRF2020) used to get a first or last step that PROJ resolves to a no-op, so
`Transformer` rejected them with "Invalid transformation pipeline".
"""
import numpy as np
import pyproj as pp
import pytest
from vyperdatum.pipeline import nwld_NAD832011_steps
from vyperdatum.transformer import Transformer
from vyperdatum.utils.crs_utils import validate_transform_steps_dict

MLLW = "NOAA:98"
LON, LAT, H = -76.197, 37.13, 5.0  # Norfolk, VA: inside the MLLW grid


@pytest.mark.parametrize("h0, start", [
    ("EPSG:4326", {"crs_from": "EPSG:9755", "crs_to": "EPSG:6318", "v_shift": False}),
    ("EPSG:9755", {"crs_from": "EPSG:9755", "crs_to": "EPSG:6318", "v_shift": False}),
    ("EPSG:4979", {"crs_from": "EPSG:9754", "crs_to": "EPSG:6319", "v_shift": False}),
    ("EPSG:9754", {"crs_from": "EPSG:9754", "crs_to": "EPSG:6319", "v_shift": False}),
    ("EPSG:7912", {"crs_from": "EPSG:7912", "crs_to": "EPSG:6319", "v_shift": False}),
    ("EPSG:9989", {"crs_from": "EPSG:9989", "crs_to": "EPSG:6319", "v_shift": False}),
    ("EPSG:9988", {"crs_from": "EPSG:9988", "crs_to": "EPSG:6319", "v_shift": False}),
])
def test_geographic_input_steps(h0, start):
    steps = nwld_NAD832011_steps(h0, None, "EPSG:6318", MLLW)
    assert steps == [start,
                     {"crs_from": "EPSG:6319", "crs_to": f"EPSG:6318+{MLLW}", "v_shift": True}]
    assert validate_transform_steps_dict(steps)


@pytest.mark.parametrize("h0", ["EPSG:4326", "EPSG:9755"])
def test_geographic_input_with_vertical_crs(h0):
    steps = nwld_NAD832011_steps(h0, "EPSG:5703", "EPSG:6318", MLLW)
    assert steps[0] == {"crs_from": "EPSG:9755", "crs_to": "EPSG:6318", "v_shift": False}
    assert validate_transform_steps_dict(steps)


@pytest.mark.parametrize("h1, end", [
    ("EPSG:4326", {"crs_from": "EPSG:6318", "crs_to": "EPSG:9755", "v_shift": False}),
    ("EPSG:4979", {"crs_from": "EPSG:6318", "crs_to": "EPSG:9754", "v_shift": False}),
    ("EPSG:9755", {"crs_from": "EPSG:6318", "crs_to": "EPSG:9755", "v_shift": False}),
])
def test_geographic_output_steps(h1, end):
    steps = nwld_NAD832011_steps("EPSG:6318", MLLW, h1, None)
    assert steps[-1] == end
    assert validate_transform_steps_dict(steps)


@pytest.mark.parametrize("h0, first", [
    ("EPSG:32618", {"crs_from": "EPSG:32618", "crs_to": "EPSG:9755", "v_shift": False}),
    ("EPSG:26918", {"crs_from": "EPSG:26918", "crs_to": "EPSG:4269", "v_shift": False}),
    ("EPSG:6347", {"crs_from": "EPSG:6347", "crs_to": "EPSG:6318", "v_shift": False}),
])
def test_projected_input_steps_unchanged(h0, first):
    """Projected inputs, which already worked, keep their steps."""
    steps = nwld_NAD832011_steps(h0, "EPSG:5703", "EPSG:6318", MLLW)
    assert steps[0] == first
    assert validate_transform_steps_dict(steps)


def test_wgs84_matches_wgs84_utm():
    """A WGS 84 point gives the same result as the same point in WGS 84 / UTM 18N."""
    e, n = pp.Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform(LON, LAT)
    ok, x_ref, y_ref, z_ref = Transformer("EPSG:32618", f"EPSG:6318+{MLLW}").transform_points(
        np.array([e]), np.array([n]), np.array([H]), always_xy=True)
    assert ok
    for crs in ["EPSG:4326", "EPSG:9755"]:
        ok, x, y, z = Transformer(crs, f"EPSG:6318+{MLLW}").transform_points(
            np.array([LON]), np.array([LAT]), np.array([H]), always_xy=True)
        assert ok, crs
        assert x[0] == pytest.approx(x_ref[0], abs=1e-9)
        assert y[0] == pytest.approx(y_ref[0], abs=1e-9)
        assert z[0] == pytest.approx(z_ref[0], abs=1e-6)


def test_wgs84_round_trip():
    ok, x, y, z = Transformer("EPSG:4326", f"EPSG:6318+{MLLW}").transform_points(
        np.array([LON]), np.array([LAT]), np.array([H]), always_xy=True)
    assert ok
    ok, x, y, z = Transformer(f"EPSG:6318+{MLLW}", "EPSG:4326").transform_points(
        x, y, z, always_xy=True)
    assert ok
    assert (x[0], y[0], z[0]) == pytest.approx((LON, LAT, H), abs=1e-6)
