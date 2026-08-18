"""Small Fourier-operator components for the SST forecasting notebooks.

The implementation follows the Fourier Neural Operator formulation of Li et al.
(2021), but the code in this module is written specifically for this project.
See ``docs/PROVENANCE.md`` for the implementation provenance and comparison
against the official NeuralOperator reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TruncatedFourierMix2d(nn.Module):
    """Mix selected 2-D Fourier coefficients with trainable complex kernels.

    Parameters are stored as real tensors with an explicit real/imaginary axis.
    This keeps the parameter representation inspectable while allowing PyTorch to
    construct complex kernels only for the Fourier-domain contraction.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes_y, modes_x) < 1:
            raise ValueError("Channel and mode counts must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x

        # Two vertical frequency bands are required because rfft2 removes only
        # the redundant half of the final Fourier dimension for real inputs.
        kernel_shape = (2, out_channels, in_channels, modes_y, modes_x, 2)
        self.kernel_components = nn.Parameter(torch.empty(kernel_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize real and imaginary kernel components."""
        for band in range(self.kernel_components.shape[0]):
            for component in range(self.kernel_components.shape[-1]):
                nn.init.xavier_normal_(self.kernel_components[band, ..., component])

    def _complex_kernels(self) -> Tensor:
        """Return kernels with shape ``[band, out, in, ky, kx]``."""
        return torch.view_as_complex(self.kernel_components.contiguous())

    @staticmethod
    def _apply_kernel(coefficients: Tensor, kernel: Tensor) -> Tensor:
        """Contract input channels against a Fourier-domain kernel."""
        return torch.einsum("bihw,oihw->bohw", coefficients, kernel)

    def forward(self, field: Tensor) -> Tensor:
        """Apply the truncated Fourier mixing operation.

        Parameters
        ----------
        field:
            Real tensor with shape ``[batch, channels, height, width]``.
        """
        if field.ndim != 4:
            raise ValueError("field must have shape [batch, channels, height, width].")

        batch, _, height, width = field.shape
        retained_x = width // 2 + 1
        if 2 * self.modes_y > height or self.modes_x > retained_x:
            raise ValueError(
                "Requested Fourier modes overlap or exceed the available spectrum: "
                f"modes_y={self.modes_y}, modes_x={self.modes_x}, shape={(height, width)}"
            )

        spectrum = torch.fft.rfft2(field, dim=(-2, -1), norm="ortho")
        mixed_spectrum = torch.zeros(
            (batch, self.out_channels, height, retained_x),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        kernels = self._complex_kernels()

        positive_y = slice(0, self.modes_y)
        negative_y = slice(height - self.modes_y, height)
        low_x = slice(0, self.modes_x)

        mixed_spectrum[:, :, positive_y, low_x] = self._apply_kernel(
            spectrum[:, :, positive_y, low_x], kernels[0]
        )
        mixed_spectrum[:, :, negative_y, low_x] = self._apply_kernel(
            spectrum[:, :, negative_y, low_x], kernels[1]
        )

        reconstructed: Tensor = torch.fft.irfft2(
            mixed_spectrum,
            s=(height, width),
            dim=(-2, -1),
            norm="ortho",
        )
        return reconstructed


class OperatorBlock2d(nn.Module):
    """Combine global Fourier mixing with a local pointwise transformation."""

    def __init__(self, width: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.global_path = TruncatedFourierMix2d(width, width, modes_y, modes_x)
        self.local_path = nn.Conv2d(width, width, kernel_size=1, bias=True)
        self.normalization = nn.GroupNorm(num_groups=1, num_channels=width)

    def forward(self, state: Tensor, *, activate: bool = True) -> Tensor:
        combined = self.global_path(state) + self.local_path(state)
        normalized = self.normalization(combined)
        return F.gelu(normalized) if activate else normalized


class FNO2d(nn.Module):
    """Compact 2-D Fourier Neural Operator for SST field forecasting.

    The temporal lookback is encoded as channels. Two normalized spatial
    coordinates are appended before lifting the input into the latent width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        *,
        width: int = 48,
        modes_y: int = 16,
        modes_x: int = 16,
        depth: int = 4,
        padding: int = 8,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, width, depth) < 1:
            raise ValueError("Channels, width, and depth must be positive.")
        if padding < 0:
            raise ValueError("padding cannot be negative.")

        self.padding = padding
        self.encoder = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.operator_blocks = nn.ModuleList(
            OperatorBlock2d(width, modes_y, modes_x) for _ in range(depth)
        )
        projection_width = max(width // 2, out_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(projection_width, out_channels, kernel_size=1),
        )

    @staticmethod
    def _spatial_coordinates(field: Tensor) -> Tensor:
        """Create normalized ``[-1, 1]`` coordinates on the current grid."""
        batch, _, height, width = field.shape
        lat_axis = torch.linspace(-1.0, 1.0, height, device=field.device, dtype=field.dtype)
        lon_axis = torch.linspace(-1.0, 1.0, width, device=field.device, dtype=field.dtype)
        lat_grid, lon_grid = torch.meshgrid(lat_axis, lon_axis, indexing="ij")
        coordinates = torch.stack((lat_grid, lon_grid), dim=0).unsqueeze(0)
        return coordinates.expand(batch, -1, -1, -1)

    def forward(self, field: Tensor) -> Tensor:
        """Forecast one output field from a stack of input fields."""
        if field.ndim != 4:
            raise ValueError("field must have shape [batch, channels, height, width].")

        augmented = torch.cat((field, self._spatial_coordinates(field)), dim=1)
        state = self.encoder(augmented)

        if self.padding > 0:
            state = F.pad(state, (0, self.padding, 0, self.padding))

        final_block = len(self.operator_blocks) - 1
        for block_index, block in enumerate(self.operator_blocks):
            state = block(state, activate=block_index != final_block)

        if self.padding > 0:
            state = state[..., : -self.padding, : -self.padding]

        prediction: Tensor = self.decoder(state)
        return prediction
