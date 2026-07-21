import numpy as np

from quantem.widget.show_polymer_4dstem import (
    _DP_VIEW_PRESETS,
    _VIEW_SETTING_PANEL_LIST_TRAITS,
    _VIEW_SETTING_TRAITS,
    _apply_view_settings,
    _collect_view_settings,
    _dp_view,
    _resolve_intensity_map,
    ShowPolymer4DSTEM,
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


# --- subpanel display-settings save / restore -------------------------------


class _ScanDataset:
    """Minimal Dataset4dstem stand-in: (Ry, Rx, Qy, Qx) shape + [ry, rx].array."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape

    def __getitem__(self, key):
        ry, rx = key
        return _Pattern(self._array[ry, rx])


class _FakeBP:
    """Duck-typed BraggPeaksPolymer: enough surface to build the widget (no peaks/polar)."""

    def __init__(self, ry=3, rx=4, qy=16, qx=16):
        rng = np.random.default_rng(0)
        self.dataset_cartesian = _ScanDataset(
            rng.random((ry, rx, qy, qx)).astype(np.float32)
        )
        self.peak_coordinates_cartesian = None
        self.peak_intensities = None
        self.polar_data = None
        self.polar_peaks = None
        self.image_centers = None


class _TraitStub:
    """Bare object carrying the view-setting trait names, for helper unit tests."""

    def __init__(self):
        n = len(_DP_VIEW_PRESETS)
        for name in _VIEW_SETTING_PANEL_LIST_TRAITS:
            setattr(self, name, [0.0] * n)
        for name in _VIEW_SETTING_TRAITS:
            if not hasattr(self, name):
                setattr(self, name, "seed")


def test_collect_apply_view_settings_round_trip():
    src = _TraitStub()
    src.dp_view_cmaps = ["a"] * len(_DP_VIEW_PRESETS)
    src.map_cmap = "magma"
    snapshot = _collect_view_settings(src)

    # A deep copy: mutating the source afterwards must not change the snapshot.
    src.dp_view_cmaps[0] = "changed"
    assert snapshot["dp_view_cmaps"][0] == "a"

    dst = _TraitStub()
    _apply_view_settings(dst, snapshot)
    assert dst.dp_view_cmaps == ["a"] * len(_DP_VIEW_PRESETS)
    assert dst.map_cmap == "magma"


def test_apply_view_settings_skips_wrong_length_panel_lists():
    dst = _TraitStub()
    original = list(dst.dp_view_zooms)
    # A stale preset with too few panels must be ignored (guards the panel arrays),
    # while global settings still apply.
    _apply_view_settings(dst, {"dp_view_zooms": [9.0], "map_cmap": "cividis"})
    assert dst.dp_view_zooms == original
    assert dst.map_cmap == "cividis"


def test_save_view_request_persists_settings_on_bp():
    bp = _FakeBP()
    imap = np.random.default_rng(1).random((3, 4)).astype(np.float32)
    w = ShowPolymer4DSTEM(bp, intensity_map=imap, show_polar=False)

    w.dp_view_cmaps = ["twilight"] * len(_DP_VIEW_PRESETS)
    w.dp_view_zooms = [2.5] * len(_DP_VIEW_PRESETS)
    w.map_cmap = "inferno"
    w._on_save_view_request(None)

    saved = getattr(bp, "_widget_view_settings", None)
    assert saved is not None
    assert saved["dp_view_cmaps"] == ["twilight"] * len(_DP_VIEW_PRESETS)
    assert saved["dp_view_zooms"] == [2.5] * len(_DP_VIEW_PRESETS)
    assert saved["map_cmap"] == "inferno"
    assert "Saved" in w.view_settings_status


def test_new_widget_restores_saved_settings_and_can_opt_out():
    bp = _FakeBP()
    imap = np.random.default_rng(2).random((3, 4)).astype(np.float32)

    first = ShowPolymer4DSTEM(bp, intensity_map=imap, show_polar=False)
    first.dp_view_cmaps = ["twilight"] * len(_DP_VIEW_PRESETS)
    first.map_cmap = "inferno"
    first._on_save_view_request(None)

    restored = ShowPolymer4DSTEM(bp, intensity_map=imap, show_polar=False)
    assert restored.dp_view_cmaps == ["twilight"] * len(_DP_VIEW_PRESETS)
    assert restored.map_cmap == "inferno"

    fresh = ShowPolymer4DSTEM(
        bp, intensity_map=imap, show_polar=False, restore_view_settings=False
    )
    assert fresh.map_cmap != "inferno"  # back to the preset default
