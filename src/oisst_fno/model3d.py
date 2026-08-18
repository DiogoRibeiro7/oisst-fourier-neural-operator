"""Spatiotemporal Fourier operator, for comparison against the channel-stacked 2-D model.

``model.py`` is left untouched: :class:`~oisst_fno.model.FNO2d` remains the baseline, and
this module is an alternative, not a replacement.

What actually changes
---------------------

In the channel-stacked model the lookback is an unordered feature axis. Its spectral layer
learns one complex matrix per retained *spatial* frequency,

.. math::

    \\hat{v}_{out}(k_y, k_x) = R_\\theta(k_y, k_x)\\, \\hat{v}_{in}(k_y, k_x),

where :math:`R_\\theta(k_y,k_x) \\in \\mathbb{C}^{out \\times in}` mixes channels — and
therefore mixes days — with a dense, unstructured matrix. Permuting the input days permutes
channels; nothing in the architecture says day ``t-1`` neighbours day ``t-2``. The model can
learn that ordering, but it is not given it.

Transforming time explicitly replaces that dense mixing with multiplication in temporal
frequency,

.. math::

    \\hat{v}_{out}(k_t, k_y, k_x) = R_\\theta(k_t, k_y, k_x)\\, \\hat{v}_{in}(k_t, k_y, k_x),

which *is* a convolution along time. Three consequences follow, and they are the reason the
comparison is interesting rather than decorative:

1. **Translation equivariance in time.** A shifted input produces a correspondingly shifted
   response, by construction rather than by training.
2. **Truncation becomes a temporal smoothness prior.** Keeping ``modes_t`` temporal
   frequencies discards fast day-to-day variation in the same way spatial truncation
   discards small scales.
3. **Parameter scaling changes.** A 2-D spectral layer holds
   :math:`O(\\text{modes}_y\\,\\text{modes}_x\\,\\text{width}^2)` weights; the 3-D layer holds
   :math:`O(\\text{modes}_t\\,\\text{modes}_y\\,\\text{modes}_x\\,\\text{width}^2)` — a factor
   of ``modes_t`` more at equal width. Matching parameter counts therefore means running the
   3-D model at a smaller width, which is exactly the trade the comparison has to price.

The temporal FFT also assumes the time window is periodic, which a 14-day history is not.
The same padding argument used for the non-periodic spatial domain applies to time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TruncatedFourierMix3d(nn.Module):
    """Mix selected space-time Fourier coefficients with trainable complex kernels.

    Follows the same conventions as :class:`~oisst_fno.model.TruncatedFourierMix2d`:
    kernels are stored as real components with an explicit real/imaginary axis and
    ``[out_channels, in_channels, ...]`` ordering, and complex tensors are constructed only
    for the Fourier-domain contraction.

    A real input means ``rfftn`` halves the last axis only, so the retained region is four
    corner blocks: both signs of temporal frequency crossed with both signs of the
    meridional frequency, at low zonal frequency.
    """

    _BANDS = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_t: int,
        modes_y: int,
        modes_x: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes_t, modes_y, modes_x) < 1:
            raise ValueError("Channel and mode counts must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_y = modes_y
        self.modes_x = modes_x

        kernel_shape = (
            self._BANDS,
            out_channels,
            in_channels,
            modes_t,
            modes_y,
            modes_x,
            2,
        )
        self.kernel_components = nn.Parameter(torch.empty(kernel_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize real and imaginary kernel components."""
        # Scaled so the variance of the mixed spectrum does not grow with the number of
        # retained modes, which is far larger here than in the 2-D layer.
        scale = 1.0 / (self.in_channels * self.modes_t * self.modes_y * self.modes_x) ** 0.5
        with torch.no_grad():
            self.kernel_components.normal_(0.0, scale)

    def _complex_kernels(self) -> Tensor:
        """Return kernels with shape ``[band, out, in, kt, ky, kx]``."""
        return torch.view_as_complex(self.kernel_components.contiguous())

    @staticmethod
    def _apply_kernel(coefficients: Tensor, kernel: Tensor) -> Tensor:
        """Contract input channels against a Fourier-domain kernel."""
        return torch.einsum("bcthw,octhw->bothw", coefficients, kernel)

    def forward(self, field: Tensor) -> Tensor:
        """Apply truncated space-time Fourier mixing.

        Parameters
        ----------
        field:
            Real tensor with shape ``[batch, channels, time, height, width]``.
        """
        if field.ndim != 5:
            raise ValueError("field must have shape [batch, channels, time, height, width].")

        batch, _, steps, height, width = field.shape
        retained_x = width // 2 + 1
        if 2 * self.modes_t > steps or 2 * self.modes_y > height or self.modes_x > retained_x:
            raise ValueError(
                "Requested Fourier modes overlap or exceed the available spectrum: "
                f"modes_t={self.modes_t}, modes_y={self.modes_y}, modes_x={self.modes_x}, "
                f"shape={(steps, height, width)}"
            )

        spectrum = torch.fft.rfftn(field, dim=(-3, -2, -1), norm="ortho")
        mixed_spectrum = torch.zeros(
            (batch, self.out_channels, steps, height, retained_x),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        kernels = self._complex_kernels()

        positive_t = slice(0, self.modes_t)
        negative_t = slice(steps - self.modes_t, steps)
        positive_y = slice(0, self.modes_y)
        negative_y = slice(height - self.modes_y, height)
        low_x = slice(0, self.modes_x)

        bands = (
            (positive_t, positive_y),
            (positive_t, negative_y),
            (negative_t, positive_y),
            (negative_t, negative_y),
        )
        for index, (time_band, lat_band) in enumerate(bands):
            mixed_spectrum[:, :, time_band, lat_band, low_x] = self._apply_kernel(
                spectrum[:, :, time_band, lat_band, low_x], kernels[index]
            )

        reconstructed: Tensor = torch.fft.irfftn(
            mixed_spectrum,
            s=(steps, height, width),
            dim=(-3, -2, -1),
            norm="ortho",
        )
        return reconstructed


class OperatorBlock3d(nn.Module):
    """Combine global space-time Fourier mixing with a local pointwise transformation."""

    def __init__(self, width: int, modes_t: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.global_path = TruncatedFourierMix3d(width, width, modes_t, modes_y, modes_x)
        self.local_path = nn.Conv3d(width, width, kernel_size=1, bias=True)
        self.normalization = nn.GroupNorm(num_groups=1, num_channels=width)

    def forward(self, state: Tensor, *, activate: bool = True) -> Tensor:
        combined = self.global_path(state) + self.local_path(state)
        normalized = self.normalization(combined)
        return F.gelu(normalized) if activate else normalized


class FNO3d(nn.Module):
    """Spatiotemporal Fourier Neural Operator over a ``[time, height, width]`` domain.

    Consumes a history sequence of shape ``[batch, channels, time_in, height, width]`` and
    returns a forecast sequence of shape ``[batch, out_channels, time_out, height, width]``.
    Producing a sequence rather than a single field is what makes temporal coherence
    measurable and lets several lead times be read from one forecast.

    Three normalized coordinates (time, latitude, longitude) are appended before lifting,
    mirroring the two spatial coordinates the 2-D model uses.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        *,
        steps_out: int,
        width: int = 20,
        modes_t: int = 4,
        modes_y: int = 12,
        modes_x: int = 12,
        depth: int = 4,
        padding: int = 4,
        time_padding: int = 2,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, width, depth, steps_out) < 1:
            raise ValueError("Channels, width, depth, and steps_out must be positive.")
        if padding < 0 or time_padding < 0:
            raise ValueError("padding values cannot be negative.")

        self.padding = padding
        self.time_padding = time_padding
        self.steps_out = steps_out

        self.encoder = nn.Conv3d(in_channels + 3, width, kernel_size=1)
        self.operator_blocks = nn.ModuleList(
            OperatorBlock3d(width, modes_t, modes_y, modes_x) for _ in range(depth)
        )
        projection_width = max(width // 2, out_channels)
        self.decoder = nn.Sequential(
            nn.Conv3d(width, projection_width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(projection_width, out_channels, kernel_size=1),
        )

    @staticmethod
    def _coordinates(field: Tensor) -> Tensor:
        """Create normalized ``[-1, 1]`` time and space coordinates on the current grid."""
        batch, _, steps, height, width = field.shape
        device, dtype = field.device, field.dtype
        time_axis = torch.linspace(-1.0, 1.0, steps, device=device, dtype=dtype)
        lat_axis = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        lon_axis = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        time_grid, lat_grid, lon_grid = torch.meshgrid(time_axis, lat_axis, lon_axis, indexing="ij")
        coordinates = torch.stack((time_grid, lat_grid, lon_grid), dim=0).unsqueeze(0)
        return coordinates.expand(batch, -1, -1, -1, -1)

    def forward(self, field: Tensor) -> Tensor:
        """Forecast a sequence of future fields from a history sequence."""
        if field.ndim != 5:
            raise ValueError("field must have shape [batch, channels, time, height, width].")

        augmented = torch.cat((field, self._coordinates(field)), dim=1)
        state = self.encoder(augmented)

        # The time window is not periodic either, so it is padded like the spatial domain.
        if self.padding > 0 or self.time_padding > 0:
            state = F.pad(
                state,
                (0, self.padding, 0, self.padding, 0, self.time_padding),
            )

        final_block = len(self.operator_blocks) - 1
        for block_index, block in enumerate(self.operator_blocks):
            state = block(state, activate=block_index != final_block)

        if self.padding > 0 or self.time_padding > 0:
            time_stop = state.shape[-3] - self.time_padding if self.time_padding else None
            state = state[
                ...,
                :time_stop,
                : -self.padding if self.padding else None,
                : -self.padding if self.padding else None,
            ]

        decoded: Tensor = self.decoder(state)

        # Map the history length onto the requested forecast length. Interpolation keeps
        # the layer resolution-agnostic in time rather than hard-coding equal lengths.
        if decoded.shape[-3] != self.steps_out:
            decoded = F.interpolate(
                decoded,
                size=(self.steps_out, decoded.shape[-2], decoded.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return decoded
