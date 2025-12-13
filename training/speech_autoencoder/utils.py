import mlx.core as mx


def ensure_waveform_2d(waveform: mx.array) -> mx.array:
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim == 3 and waveform.shape[-1] == 1:
        waveform = waveform[..., 0]
    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform shaped (B, T) or (B, T, 1) (or (T,) for a single example)."
        )
    return waveform.astype(mx.float32)


def hann_window(length: int, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    if length <= 1:
        return mx.ones((length,), dtype=dtype)
    n = mx.arange(length, dtype=dtype)
    return 0.5 - 0.5 * mx.cos((2.0 * mx.array(3.141592653589793, dtype=dtype) * n) / (length - 1))


def reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int, *, axis: int = -1) -> mx.array:
    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative")
    if pad_left == 0 and pad_right == 0:
        return x

    axis = axis if axis >= 0 else x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError("axis out of range")

    n = int(x.shape[axis])
    if n <= 0:
        raise ValueError("Cannot pad an empty axis")
    if n == 1:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (pad_left, pad_right)
        return mx.pad(x, pad_width=pad_width, constant_values=float(x.reshape((-1,))[0].item()))

    period = 2 * (n - 1)
    positions = mx.arange(-pad_left, n + pad_right, dtype=mx.int32)
    pos_mod = mx.remainder(positions, period)
    indices = mx.where(pos_mod <= (n - 1), pos_mod, period - pos_mod)
    return mx.take(x, indices, axis=axis)
