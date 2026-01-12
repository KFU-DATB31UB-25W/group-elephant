import zipfile
from pathlib import Path

import numpy as np

from group_elephant.workstreams.data import secom_preprocess as sp


def test_preprocess_from_zip(tmp_path: Path):
    secom_data = "1 2 3\n4 nan 6\n7 8 9\n"
    secom_labels = "-1 2008-07-16 00:00:00\n1 2008-07-16 00:00:01\n-1 2008-07-16 00:00:02\n"

    zip_path = tmp_path / "secom.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("secom.data", secom_data)
        zf.writestr("secom_labels.data", secom_labels)

    cfg = sp.PreprocessConfig(max_missing_fraction=1.0)
    artifacts, meta = sp.preprocess_secom(None, None, zip_path, cfg)

    out_dir = tmp_path / "artifacts"
    npz_path = sp.save_npz(artifacts, out_dir)

    d = np.load(npz_path, allow_pickle=True)
    assert d["X"].shape[0] == 3
    assert d["y"].shape == (3,)
    assert len(d["timestamps"]) == 3
