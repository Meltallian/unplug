"""Chord acoustic model: BTC, the Bi-directional Transformer for Chord recognition
(Park, Choi, Lee, Kim and Kim, ISMIR 2019), with its published large-vocabulary weights.

For every 93 ms frame it gives a probability for each of 170 chords: 12 roots × 14 qualities,
plus "X" (a chord outside the vocabulary) and "N" (no chord). The decoding (which chord on which
beat) is done in chords.py; here are only the network and its input features.

The network is re-written compactly from https://github.com/jayg996/BTC-ISMIR19 (MIT licence,
© 2019 Jonggwon Park) with the same parameter names, so the published checkpoint loads as is.
"""

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

CHECKPOINT = os.environ.get("BTC_CHECKPOINT", "btc_model_large_voca.pt")
SAMPLE_RATE = 22050
HOP = 2048
N_BINS, BINS_PER_OCTAVE = 144, 24
CHUNK_S = 10.0                    # features and the network work on 10 s windows ...
TIMESTEP = 108                    # ... of 108 frames
FRAME_S = CHUNK_S / TIMESTEP      # seconds per frame (as the authors count it)
HIDDEN, HEADS, LAYERS = 128, 4, 8

ROOTS = 12
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7",
             "dim7", "hdim7", "sus2", "sus4"]
N_CHORDS = ROOTS * len(QUALITIES) + 2
X_INDEX, N_INDEX = N_CHORDS - 2, N_CHORDS - 1


def chord_index(root: int, quality: str) -> int:
    return root * len(QUALITIES) + QUALITIES.index(quality)


def _build():
    import torch
    from torch import nn

    class LayerNorm(nn.Module):
        def __init__(self, features: int, eps: float = 1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            return self.gamma * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.beta

    class MultiHeadAttention(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.query_linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.key_linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.value_linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.output_linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
            self.mask = mask                                 # which frames each frame may look at
            self.scale = (HIDDEN // HEADS) ** -0.5

        def _split(self, x):
            b, t, d = x.shape
            return x.view(b, t, HEADS, d // HEADS).permute(0, 2, 1, 3)

        def forward(self, x):
            q = self._split(self.query_linear(x)) * self.scale
            k, v = self._split(self.key_linear(x)), self._split(self.value_linear(x))
            t = x.shape[1]
            weights = torch.softmax(q @ k.transpose(-1, -2) + self.mask[:, :, :t, :t], dim=-1)
            out = (weights @ v).permute(0, 2, 1, 3).contiguous().view(x.shape)
            return self.output_linear(out)

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((1, 1), 0)
            self.conv = nn.Conv1d(HIDDEN, HIDDEN, kernel_size=3)

        def forward(self, x):
            return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Conv(), Conv()])

        def forward(self, x):
            for layer in self.layers:        # ReLU after the last layer too, as in the trained model
                x = torch.relu(layer(x))
            return x

    class AttentionBlock(nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.multi_head_attention = MultiHeadAttention(mask)
            self.positionwise_convolution = FeedForward()
            self.layer_norm_mha = LayerNorm(HIDDEN)
            self.layer_norm_ffn = LayerNorm(HIDDEN)

        def forward(self, x):
            x = x + self.multi_head_attention(self.layer_norm_mha(x))
            return x + self.positionwise_convolution(self.layer_norm_ffn(x))

    class BiDirectional(nn.Module):
        def __init__(self, past, future):
            super().__init__()
            self.attn_block = AttentionBlock(past)                # looks back in time
            self.backward_attn_block = AttentionBlock(future)     # looks ahead
            self.linear = nn.Linear(HIDDEN * 2, HIDDEN)

        def forward(self, x):
            return self.linear(torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            past = torch.triu(torch.full((TIMESTEP, TIMESTEP), float("-inf")), 1)[None, None]
            self.embedding_proj = nn.Linear(N_BINS, HIDDEN, bias=False)
            self.self_attn_layers = nn.Sequential(*[BiDirectional(past, past.transpose(2, 3)) for _ in range(LAYERS)])
            self.layer_norm = LayerNorm(HIDDEN)
            position = np.arange(TIMESTEP)[:, None]
            scales = np.exp(np.arange(HIDDEN // 2) * -(math.log(1.0e4) / (HIDDEN // 2 - 1)))[None, :]
            signal = np.concatenate([np.sin(position * scales), np.cos(position * scales)], axis=1)
            self.timing_signal = torch.from_numpy(signal[None]).float()

        def forward(self, x):
            x = self.embedding_proj(x) + self.timing_signal[:, :x.shape[1]]
            return self.layer_norm(self.self_attn_layers(x))

    class Output(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_projection = nn.Linear(HIDDEN, N_CHORDS)

        def forward(self, x):
            return self.output_projection(x)

    class BTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn_layers = Encoder()
            self.output_layer = Output()

        def forward(self, x):
            return self.output_layer(self.self_attn_layers(x))

    return BTC()


class _NumberType:
    """Stand-in for a pickled numpy dtype when reading the checkpoint: it only remembers which
    plain number type was meant ("f8", "i4", ...) and its byte order."""

    def __init__(self, code="", *_):
        self.code, self.order = str(code), "<"

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) > 1 and state[1] in ("<", ">", "=", "|"):
            self.order = state[1]


def _plain_number(dtype, data):
    """Stand-in for a pickled numpy scalar when reading the checkpoint: plain numbers only."""
    if not isinstance(dtype, _NumberType) or dtype.code not in ("f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
        raise ValueError("unexpected value type in the chord model checkpoint")
    order = "" if dtype.order == "|" else dtype.order
    return np.frombuffer(data, dtype=np.dtype(order + dtype.code))[0]


def read_checkpoint(path: Path) -> tuple[dict, float, float]:
    """(network weights, feature mean, feature std) from the published checkpoint file. It is read
    with torch's restricted loader: tensors and plain numbers only, never arbitrary pickled code."""
    import torch

    allowed = [(_plain_number, "numpy.core.multiarray.scalar"), (_plain_number, "numpy._core.multiarray.scalar"),
               (_NumberType, "numpy.dtype")]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in checkpoint["model"].items() if not k.startswith("output_layer.lstm.")}  # unused layer
    return state, float(checkpoint["mean"]), float(checkpoint["std"])


@lru_cache(maxsize=1)
def _model():
    path = Path(CHECKPOINT)
    if not path.exists():
        raise FileNotFoundError(f"chord model weights not found at {path} (set BTC_CHECKPOINT)")
    state, mean, std = read_checkpoint(path)
    net = _build()
    net.load_state_dict(state)
    net.eval()
    return net, mean, std


def cqt_features(y: np.ndarray, sr: int, offset_s: float = 0.0) -> np.ndarray:
    """Log constant-Q spectrogram (frames × 144), computed window by window like the training
    data. With an offset the windows start that much later (used to cover the window joins)."""
    import librosa

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y[int(offset_s * SAMPLE_RATE):]
    size = int(SAMPLE_RATE * CHUNK_S)
    chunks = []
    for start in range(0, max(len(y), 1), size):
        chunk = y[start:start + size]
        frames = 1 + len(chunk) // HOP
        if len(chunk) < size:                      # the last window: pad with silence, keep its frames
            chunk = np.pad(chunk, (0, size - len(chunk)))
        c = librosa.cqt(chunk, sr=SAMPLE_RATE, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                        hop_length=HOP, pad_mode="reflect")
        chunks.append(np.abs(c[:, :frames]))
    return np.log(np.concatenate(chunks, axis=1) + 1e-6).T.astype(np.float32)


def _run(features: np.ndarray) -> np.ndarray:
    """Chord probabilities (frames × 170) for a (frames × 144) feature matrix."""
    import torch

    net, mean, std = _model()
    n = len(features)
    x = (features - mean) / std
    pad = (-n) % TIMESTEP
    x = np.pad(x, ((0, pad), (0, 0)))
    with torch.no_grad():
        windows = torch.from_numpy(x).view(-1, TIMESTEP, N_BINS)
        probs = torch.cat([torch.softmax(net(w[None]), dim=-1)[0] for w in windows])
    return probs.numpy()[:n]


def chord_probabilities(y: np.ndarray, sr: int) -> np.ndarray:
    """(frames × 170) chord probabilities; frame i covers [i, i+1) × FRAME_S seconds.

    The network sees 10 s windows and is least sure at their edges, so it runs twice, the second
    time on windows shifted by 5 s, and the two answers are averaged.
    """
    first = _run(cqt_features(y, sr))
    shift = TIMESTEP // 2
    second = _run(cqt_features(y, sr, offset_s=CHUNK_S / 2))
    out = first.copy()
    n = min(len(first) - shift, len(second))
    if n > 0:
        out[shift:shift + n] = (first[shift:shift + n] + second[:n]) / 2
    return out
