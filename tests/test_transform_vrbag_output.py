"""
`Transformer.transform_vrbag()` must remove its output copy when the VR BAG
transformation fails, as the other `transform_*()` methods do.
"""
import h5py
import numpy as np
import pytest
from vyperdatum.drivers import vrbag
from vyperdatum.transformer import Transformer

STEPS = [{"crs_from": "EPSG:6318+EPSG:5703", "crs_to": "EPSG:6319", "v_shift": True}]


def _write_minimal_vrbag(path):
    """Just enough HDF5 layout for `vrbag.is_vr()`."""
    with h5py.File(path, "w") as bag:
        root = bag.create_group("BAG_root")
        root.create_dataset("elevation", data=np.zeros((2, 2), dtype=np.float32))
        root.create_dataset("varres_refinements", data=np.zeros((1, 4), dtype=np.float32))


@pytest.mark.parametrize("vrbag_result", [False, True], ids=["fails", "succeeds"])
def test_transform_vrbag_output(vrbag_result, tmp_path, monkeypatch):
    input_file, output_file = tmp_path / "in.bag", tmp_path / "out.bag"
    _write_minimal_vrbag(input_file)
    monkeypatch.setattr(vrbag, "transform", lambda **kwargs: vrbag_result)
    tf = Transformer(crs_from="EPSG:6318+EPSG:5703", crs_to="EPSG:6319", steps=STEPS)
    success = tf.transform_vrbag(input_file=str(input_file), output_file=str(output_file),
                                 pre_post_checks=False, vdatum_check=False)
    assert success is vrbag_result
    assert output_file.exists() is vrbag_result
