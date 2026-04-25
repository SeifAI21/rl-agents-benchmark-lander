from pathlib import Path

import torch

from DQN.utils import plot_training_curves, resolve_device


def test_resolve_device_returns_torch_device():
    d = resolve_device("cpu")
    assert isinstance(d, torch.device)
    assert d.type == "cpu"


def test_plot_training_curves_writes_png(tmp_path: Path):
    out = plot_training_curves(
        [1.0, 2.0, -1.0],
        [0.5, 0.4, 0.3],
        save_dir=str(tmp_path),
        filename="curves.png",
    )
    assert Path(out).is_file()
    assert Path(out).stat().st_size > 0
