"""
Grid path handling tests that do not need a VDatum install.  A tiny fake VDatum folder is written to a temp
directory and added to the pyproj data directory.  See noaa-ocs-hydrography/vyperdatum#44.
"""

import os
import struct

import pyproj
import pytest
from pyproj import Transformer

from vyperdatum.core import proj_data_dir_contains
from vyperdatum.pipeline import get_regional_pipeline
from vyperdatum.vypercrs import is_valid_regional_pipeline

GEOID = r'core\geoid12b\g2012bu0.gtx'
REGION = 'NCcoast11_8301'


def _write_gtx(path: str, value: float):
    # 10x10 GTX grid with a constant value, lower left at 34N 80W, 0.5 degree spacing
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = cols = 10
    with open(path, 'wb') as gfile:
        gfile.write(struct.pack('>4d2i', 34.0, -80.0, 0.5, 0.5, rows, cols))
        gfile.write(struct.pack(f'>{rows * cols}f', *([value] * rows * cols)))


@pytest.fixture
def fake_vdatum(tmp_path):
    vdatum = str(tmp_path / 'vdatum')
    _write_gtx(os.path.join(vdatum, 'core', 'geoid12b', 'g2012bu0.gtx'), 0.5)
    _write_gtx(os.path.join(vdatum, REGION, 'tss.gtx'), 1.0)
    _write_gtx(os.path.join(vdatum, REGION, 'mllw.gtx'), 2.0)
    orig_data_dir = pyproj.datadir.get_data_dir()
    pyproj.datadir.append_data_dir(vdatum)  # same as DatumData.set_vdatum_directory
    yield vdatum
    pyproj.datadir.set_data_dir(orig_data_dir)


def test_is_valid_regional_pipeline_finds_grids(fake_vdatum):
    pipeline = get_regional_pipeline('ellipse', 'mllw', REGION, GEOID)
    valid, pipeline = is_valid_regional_pipeline(pipeline)
    assert valid
    if os.sep != '\\':
        assert '\\' not in pipeline


def test_is_valid_regional_pipeline_runs_in_proj(fake_vdatum):
    pipeline = get_regional_pipeline('ellipse', 'mllw', REGION, GEOID)
    valid, pipeline = is_valid_regional_pipeline(pipeline)
    assert valid
    x, y, z = Transformer.from_pipeline(pipeline).transform(-76.19698, 37.1299, 10.5)
    assert z == pytest.approx(10.5 - 0.5 + 1.0 - 2.0)


def test_is_valid_regional_pipeline_missing_grid(fake_vdatum):
    pipeline = get_regional_pipeline('ellipse', 'mllw', 'NotARegion_8301', GEOID)
    valid, pipeline = is_valid_regional_pipeline(pipeline)
    assert not valid


def test_is_valid_regional_pipeline_alternate_extension(fake_vdatum):
    os.rename(os.path.join(fake_vdatum, REGION, 'mllw.gtx'), os.path.join(fake_vdatum, REGION, 'mllw.tif'))
    pipeline = get_regional_pipeline('ellipse', 'mllw', REGION, GEOID)
    valid, pipeline = is_valid_regional_pipeline(pipeline)
    assert valid
    assert 'mllw.tif' in pipeline
    assert 'mllw.gtx' not in pipeline


def test_proj_data_dir_contains(fake_vdatum):
    assert proj_data_dir_contains(fake_vdatum)
    assert proj_data_dir_contains(fake_vdatum + os.sep)
    # a path that is only a prefix of an existing entry is not a match
    assert not proj_data_dir_contains(fake_vdatum[:-1])
