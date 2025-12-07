import mlx.core as mx
import mlx.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        hidden_dim = expansion * dim
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dims=dim)
        self.pwconv1 = nn.Conv1d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,
        )
        self.gamma = (
            mx.full(shape=(1, 1, dim), vals=layer_scale_init_value, dtype=mx.float32)
            if layer_scale_init_value > 0.0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return shortcut + x


class CausalConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for causal padding")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        hidden_dim = expansion * dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dims=dim)
        self.pwconv1 = nn.Conv1d(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,
        )
        self.gamma = (
            mx.full(shape=(1, 1, dim), vals=layer_scale_init_value, dtype=mx.float32)
            if layer_scale_init_value > 0.0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        left_pad = (self.kernel_size - 1) * self.dilation
        pad_width = [(0, 0), (left_pad, 0), (0, 0)]
        x = mx.pad(x, pad_width=pad_width, constant_values=0)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return shortcut + x
