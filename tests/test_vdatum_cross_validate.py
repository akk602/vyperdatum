"""
`vdatum_cross_validate()` on point data, with the VDatum API replaced by a stub so the
tests don't need the network.
"""
import numpy as np
import pyproj as pp
import pytest
from vyperdatum.transformer import Transformer
from vyperdatum.utils import vdatum_rest_utils

S_WKT = pp.CRS("EPSG:6319").to_wkt()
T_WKT = pp.CRS("EPSG:6318+NOAA:98").to_wkt()


def _points(n):
    source = [[-76.2 + i * 0.001, 37.1, 5.0] for i in range(n)]
    target = [[x, y, z + 40.0] for x, y, z in source]
    return source, target


@pytest.fixture
def vdatum_agrees(monkeypatch):
    """Stub the VDatum API so it returns exactly the target heights."""
    def transform_point(s_x, s_y, s_z, region, *args, **kwargs):
        return (s_x, s_y, s_z + 40.0), {}
    monkeypatch.setattr(vdatum_rest_utils, "vdatum_transform_point", transform_point)


@pytest.mark.parametrize("n_points", [1, 3, 19, 20, 50])
def test_point_check_with_any_number_of_points(vdatum_agrees, n_points):
    """Fewer points than `n_sample` used to raise ValueError from random.sample()."""
    source, target = _points(n_points)
    passed, df = vdatum_rest_utils.vdatum_cross_validate(
        s_wkt=S_WKT, t_wkt=T_WKT, n_sample=20,
        s_raster_metadata=None, t_raster_metadata=None,
        s_point_samples=source, t_point_samples=target,
        pivot_h_crs="EPSG:6318")
    assert passed
    assert len(df) == min(n_points, 20)


def test_transform_points_runs_vdatum_check_for_few_points(monkeypatch):
    """
    With fewer than 20 points the VDatum check used to raise before comparing anything.
    On main that error was swallowed, so the check was silently skipped.
    """
    calls = []

    def transform_point(s_x, s_y, s_z, region, *args, **kwargs):
        calls.append((s_x, s_y))
        return (s_x, s_y, s_z + 37.23), {}  # about the MLLW shift near Norfolk, VA

    monkeypatch.setattr(vdatum_rest_utils, "vdatum_transform_point", transform_point)
    tf = Transformer(crs_from="EPSG:6319", crs_to="EPSG:6318+NOAA:98",
                     steps=[{"crs_from": "EPSG:6319", "crs_to": "EPSG:6318+NOAA:98",
                             "v_shift": True}])
    lon, lat = np.array([-76.197, -76.198, -76.199]), np.array([37.13, 37.131, 37.132])
    success, _, _, z = tf.transform_points(lon, lat, np.full(3, 5.0), always_xy=True,
                                           vdatum_check=True)
    assert len(calls) == 3, "the VDatum check did not run"
    assert success is True
    assert np.all(np.isfinite(z))
