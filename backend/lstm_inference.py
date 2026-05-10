"""
lstm_inference.py — Pure NumPy LSTM inference.
Exact architectural replica of lstm_trend_model.keras:
  Input  → (batch, 12, 6)
  LSTM(64, return_sequences=True) + Dropout(0.2)
  LSTM(64, return_sequences=False) + Dropout(0.2)
  Dense(32, relu)
  Dense(1, sigmoid)

No TensorFlow, no Keras, no torch. Only numpy (already in requirements.txt).

Usage:
    from lstm_inference import LSTMModel
    model = LSTMModel()                        # loads lstm_weights.npz from same dir
    prob  = model.predict(x)                   # x: np.ndarray shape (batch, 12, 6)
"""

from __future__ import annotations
import os
import numpy as np

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ---------------------------------------------------------------------------
# Single LSTM cell step  (Keras default gate order: i, f, c, o)
# ---------------------------------------------------------------------------

def _lstm_step(
    x_t:   np.ndarray,   # (batch, input_dim)
    h_prev: np.ndarray,  # (batch, units)
    c_prev: np.ndarray,  # (batch, units)
    kernel:    np.ndarray,    # (input_dim, 4*units)
    recurrent: np.ndarray,    # (units,     4*units)
    bias:      np.ndarray,    # (4*units,)
) -> tuple[np.ndarray, np.ndarray]:
    units = h_prev.shape[-1]
    z = x_t @ kernel + h_prev @ recurrent + bias   # (batch, 4*units)

    i = _sigmoid(z[:, :units])           # input gate
    f = _sigmoid(z[:, units:2*units])    # forget gate
    c_cand = _tanh(z[:, 2*units:3*units])
    o = _sigmoid(z[:, 3*units:])         # output gate

    c = f * c_prev + i * c_cand
    h = o * _tanh(c)
    return h, c


# ---------------------------------------------------------------------------
# Full LSTM layer over a sequence
# ---------------------------------------------------------------------------

def _lstm_layer(
    x:         np.ndarray,   # (batch, timesteps, input_dim)
    kernel:    np.ndarray,
    recurrent: np.ndarray,
    bias:      np.ndarray,
    return_sequences: bool,
) -> np.ndarray:
    batch, timesteps, _ = x.shape
    units = kernel.shape[1] // 4

    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)

    outputs = []
    for t in range(timesteps):
        h, c = _lstm_step(x[:, t, :], h, c, kernel, recurrent, bias)
        if return_sequences:
            outputs.append(h)

    if return_sequences:
        return np.stack(outputs, axis=1)   # (batch, timesteps, units)
    return h                               # (batch, units)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    Loads lstm_weights.npz and runs forward inference.
    Dropout layers are identity at inference time (no randomness).
    """

    def __init__(self, weights_path: str | None = None):
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "lstm_weights.npz",
            )
        w = np.load(weights_path)

        self._lstm0_kernel    = w["lstm0_kernel"].astype(np.float32)
        self._lstm0_recurrent = w["lstm0_recurrent"].astype(np.float32)
        self._lstm0_bias      = w["lstm0_bias"].astype(np.float32)

        self._lstm1_kernel    = w["lstm1_kernel"].astype(np.float32)
        self._lstm1_recurrent = w["lstm1_recurrent"].astype(np.float32)
        self._lstm1_bias      = w["lstm1_bias"].astype(np.float32)

        self._dense0_kernel   = w["dense0_kernel"].astype(np.float32)
        self._dense0_bias     = w["dense0_bias"].astype(np.float32)
        self._dense1_kernel   = w["dense1_kernel"].astype(np.float32)
        self._dense1_bias     = w["dense1_bias"].astype(np.float32)

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (batch, 12, 6)  — 12 quarters, 6 features
        returns: shape (batch,)  — crisis probability in [0, 1]
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis]          # add batch dim if single sample

        # LSTM 0  (return_sequences=True → passes full seq to LSTM 1)
        out = _lstm_layer(
            x,
            self._lstm0_kernel, self._lstm0_recurrent, self._lstm0_bias,
            return_sequences=True,
        )
        # Dropout(0.2) — identity at inference

        # LSTM 1  (return_sequences=False → last hidden state)
        out = _lstm_layer(
            out,
            self._lstm1_kernel, self._lstm1_recurrent, self._lstm1_bias,
            return_sequences=False,
        )
        # Dropout(0.2) — identity at inference

        # Dense(32, relu)
        out = _relu(out @ self._dense0_kernel + self._dense0_bias)

        # Dense(1, sigmoid)
        out = _sigmoid(out @ self._dense1_kernel + self._dense1_bias)

        return out.squeeze(-1)         # (batch,)

    # convenience for single sample dict input (matches your trend.py style)
    def predict_proba(self, sequence: list[dict], feature_cols: list[str]) -> float:
        """
        sequence : list of 12 dicts, each with keys = feature_cols
        returns  : float crisis probability
        """
        arr = np.array(
            [[row[c] for c in feature_cols] for row in sequence],
            dtype=np.float32,
        )                              # (12, 6)
        return float(self.predict(arr[np.newaxis])[0])