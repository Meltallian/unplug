import pickle

import numpy as np
import pytest

from app.pipeline import btc


def test_checkpoint_with_numpy_numbers_is_read(tmp_path):
    import torch
    path = tmp_path / "model.pt"
    torch.save({"model": {"w": torch.ones(2), "output_layer.lstm.weight": torch.zeros(1)},
                "mean": np.float64(-2.5), "std": np.float32(1.5)}, path)
    state, mean, std = btc.read_checkpoint(path)
    assert list(state) == ["w"] and mean == -2.5 and std == 1.5


def test_checkpoint_carrying_code_is_refused(tmp_path):
    import torch

    class Evil:
        def __reduce__(self):
            return (print, ("this must never run",))

    path = tmp_path / "evil.pt"
    torch.save({"model": {}, "mean": Evil(), "std": 1.0}, path)
    with pytest.raises(pickle.UnpicklingError):
        btc.read_checkpoint(path)


def test_checkpoint_numbers_must_be_plain_numbers():
    with pytest.raises(ValueError):
        btc._plain_number(btc._NumberType("O"), b"\x80\x04N.")       # a pickled object posing as a number


def test_network_gives_one_answer_per_frame_and_chord():
    import torch
    net = btc._build().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, btc.TIMESTEP, btc.N_BINS))
    assert out.shape == (1, btc.TIMESTEP, btc.N_CHORDS)
    assert btc.chord_index(0, "min") == 0 and btc.chord_index(11, "sus4") == 167 and btc.N_INDEX == 169


def test_features_keep_the_frame_count_of_each_window():
    y = np.zeros(int(btc.SAMPLE_RATE * 25), dtype=np.float32)        # 10 s + 10 s + 5 s
    feats = btc.cqt_features(y, btc.SAMPLE_RATE)
    assert feats.shape == (108 + 108 + 54, btc.N_BINS)
