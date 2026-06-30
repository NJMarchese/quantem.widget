import numpy as np

from quantem.widget.show_polymer_4dstem import (
    _DP_VIEW_PRESETS,
    _dp_view,
    _resolve_intensity_map,
)


class _Pattern:
    def __init__(self, array):
        self.array = array


class _Dataset:
    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        ry, rx = key
        return _Pattern(self._array[ry, rx])


def test_dp_view_zoom_crop_uses_supplied_center():
    data = np.arange(1 * 1 * 8 * 8, dtype=np.float32).reshape(1, 1, 8, 8)
    dataset = _Dataset(data)

    cropped, x0, y0 = _dp_view(
        dataset,
        0,
        0,
        norm_upper_quantile=None,
        norm_power=1.0,
        gaussian_filter_sigma=None,
        zoom=2.0,
        center=(2.0, 5.0),
    )

    assert (x0, y0) == (4, 0)
    np.testing.assert_array_equal(cropped, data[0, 0, 0:4, 4:8])


def test_resolve_intensity_map_scalar_is_not_rgb():
    arr = np.random.rand(16, 20).astype(np.float32)
    out, up, is_rgb = _resolve_intensity_map(None, arr, (16, 20))
    assert is_rgb is False
    assert up == 1
    assert out.shape == (16, 20)
    assert out.dtype == np.float32


def test_resolve_intensity_map_rgb_keeps_three_channels_and_clips():
    # Out-of-range RGB values must be clipped into [0, 1]; alpha dropped if present.
    rgba = np.random.rand(16, 20, 4).astype(np.float32) * 2.0 - 0.5
    out, up, is_rgb = _resolve_intensity_map(None, rgba, (16, 20))
    assert is_rgb is True
    assert up == 1
    assert out.shape == (16, 20, 3)
    assert out.flags["C_CONTIGUOUS"]
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resolve_intensity_map_rgb_supports_integer_upsample():
    rgb = np.random.rand(32, 40, 3).astype(np.float32)
    out, up, is_rgb = _resolve_intensity_map(None, rgb, (16, 20))
    assert is_rgb is True
    assert up == 2
    assert out.shape == (32, 40, 3)


def test_pipi_preset_matches_reference_image_map_settings():
    pipi = next(p for p in _DP_VIEW_PRESETS if p["key"] == "pipi")

    assert pipi["cmap"] == "turbo_black"
    assert pipi["norm_upper_quantile"] == 0.9999
    assert pipi["norm_power"] == 1.5
    assert pipi["gaussian_filter_sigma"] == 4.0
    assert pipi["vmin"] == 0.055
    assert pipi["vmax"] == 0.13
