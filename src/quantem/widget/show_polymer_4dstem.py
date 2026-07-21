"""
show_polymer_4dstem: Interactive Bragg-peak / polymer 4D-STEM viewer.

A React (anywidget) port of ``BraggPeaksPolymer.plot_interactive_peak_map`` /
``plot_interactive_image_map`` from core ``quantem``. Click a real-space
intensity map to browse the diffraction pattern at that scan position, with the
detected Bragg peaks overlaid (sized by intensity, central beam highlighted) and
an optional polar-transform panel.

This is a *live-kernel* widget: the running Python kernel recomputes the selected
diffraction pattern + peaks on every click and ships them over the Jupyter comm
channel to the browser. It does not need WebGPU and works with NumPy, PyTorch, or
CuPy-backed datasets.

Public entry point::

    from quantem.widget import show_polymer_4DSTEM
    w = show_polymer_4DSTEM(bragg_peaks)   # bragg_peaks: BraggPeaksPolymer
"""

import copy
import pathlib
import warnings

import anywidget
import numpy as np
import traitlets

from quantem.widget.utils.array import to_numpy

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - scipy is optional for downstream users.
    gaussian_filter = None

# Intensity field used by BraggPeaksPolymer.peak_intensities (matches the
# `intensity_field='intensities'` default in plot_interactive_peak_map).
_DEFAULT_INTENSITY_FIELD = "intensities"


# ---------------------------------------------------------------------------
# Small data helpers (re-implemented here so the widget stays self-contained
# and does not depend on private ``quantem.diffraction`` internals).
# ---------------------------------------------------------------------------
def _mean_intensity_map(dataset_cartesian, scan_shape):
    """Mean detector intensity per scan position -> (Ry, Rx) float array."""
    Ry, Rx = scan_shape
    return np.array(
        [
            [float(np.mean(to_numpy(dataset_cartesian[i, j].array))) for j in range(Rx)]
            for i in range(Ry)
        ],
        dtype=np.float32,
    )


def _resolve_intensity_map(dataset_cartesian, intensity_map, scan_shape):
    """Return (map_float32, upsample_factor, is_rgb). None -> mean map at upsample 1.

    A 2D array is sent as a scalar field (colormapped on the frontend via ``map_cmap``).
    A 3D ``(H, W, 3|4)`` array is treated as a true RGB(A) image: the first three
    channels are kept (clipped to ``[0, 1]``) and drawn directly, bypassing the colormap.
    """
    Ry, Rx = scan_shape
    if intensity_map is None:
        return _mean_intensity_map(dataset_cartesian, scan_shape), 1, False
    arr = np.asarray(to_numpy(intensity_map))
    is_rgb = arr.ndim == 3
    if is_rgb:
        # RGB(A) map: keep the first three channels and render without a colormap.
        arr = arr[..., :3]
    elif arr.ndim != 2:
        raise ValueError(
            f"intensity_map must be 2D or RGB ((H, W, 3)), got shape {arr.shape}"
        )
    up = arr.shape[0] // Ry
    if up < 1 or arr.shape[0] % Ry or arr.shape[1] % Rx or arr.shape[1] // Rx != up:
        raise ValueError(
            f"intensity_map shape {arr.shape} is not an integer multiple of "
            f"the scan grid ({Ry}, {Rx})"
        )
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if is_rgb:
        arr = np.clip(arr, 0.0, 1.0)
    return arr, int(up), is_rgb


def _display_limits(arr):
    """1st/99th percentile clip limits, robust to NaN/uniform data."""
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    lo, hi = np.quantile(arr[finite], [0.01, 0.99])
    if not (hi > lo):
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalized_dp(dataset_cartesian, ry, rx, *, norm_upper_quantile, norm_power):
    """Single diffraction pattern at (ry, rx) with optional clip/power scaling."""
    dp = np.asarray(to_numpy(dataset_cartesian[ry, rx].array), dtype=np.float32).copy()
    if norm_upper_quantile is not None:
        dp = np.clip(dp, 0, np.quantile(dp, norm_upper_quantile))
    if norm_power != 1.0:
        m = float(np.nanmax(dp))
        if np.isfinite(m) and m > 0:
            dp = (dp / m) ** norm_power * m
    return dp


def _dp_view(
    dataset_cartesian,
    ry,
    rx,
    *,
    norm_upper_quantile,
    norm_power,
    gaussian_filter_sigma,
    zoom,
    center=None,
):
    """Tutorial-style display transform for one DP panel."""
    dp = _normalized_dp(
        dataset_cartesian,
        ry,
        rx,
        norm_upper_quantile=norm_upper_quantile,
        norm_power=norm_power,
    )
    if gaussian_filter_sigma and gaussian_filter_sigma > 0 and gaussian_filter is not None:
        dp = gaussian_filter(dp, sigma=float(gaussian_filter_sigma)).astype(np.float32, copy=False)

    x0 = y0 = 0
    if zoom and zoom > 1:
        h, w = dp.shape
        crop_h = max(1, int(round(h / float(zoom))))
        crop_w = max(1, int(round(w / float(zoom))))
        if center is None:
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        else:
            cy, cx = center
        y0 = int(round(float(cy) - (crop_h - 1) / 2.0))
        x0 = int(round(float(cx) - (crop_w - 1) / 2.0))
        y0 = min(max(y0, 0), h - crop_h)
        x0 = min(max(x0, 0), w - crop_w)
        dp = dp[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return np.ascontiguousarray(dp, dtype=np.float32), int(x0), int(y0)


def _display_center(image_centers, ry, rx, dp_shape):
    """Center (y, x): stored beam center if valid, else geometric center."""
    cy, cx = dp_shape[0] / 2.0, dp_shape[1] / 2.0
    if image_centers is not None:
        c = to_numpy(image_centers)[:, ry, rx]
        if np.all(np.isfinite(c)) and not np.allclose(c, 0):
            cy, cx = float(c[0]), float(c[1])
    return cy, cx


def _has_peaks(px, py):
    return px is not None and py is not None and len(px) > 0 and len(py) > 0


def _central_beam_max_dist(dp_shape):
    """Pixel radius within which a detected peak counts as the central beam.

    Small enough that finite-q Bragg peaks are never mistaken for the beam, generous
    enough to absorb a few-pixel disagreement between center-finding and peak detection.
    """
    return max(4.0, 0.03 * min(dp_shape[0], dp_shape[1]))


def _central_peak_index(px, py, r_invA, center, max_dist=None):
    """Index of the detected central-beam peak, or -1.

    The peak nearest the calibrated ``center`` (from ``image_centers``), but only when
    within ``max_dist`` px. The filled central marker is always drawn at ``center``
    itself; this index only flags which peak, if any, to drop from the open-circle set so
    a ring is not drawn on the beam. ``r_invA`` is unused (kept for call-site
    compatibility): picking the smallest polar radius made the marker jump to an
    off-center low-q Bragg peak when the beam was not itself detected.
    """
    if not _has_peaks(px, py):
        return -1
    cy, cx = center
    d2 = (np.asarray(px) - cx) ** 2 + (np.asarray(py) - cy) ** 2
    idx = int(np.argmin(d2))
    if max_dist is not None and d2[idx] > max_dist ** 2:
        return -1
    return idx


def _as_float_list(arr):
    if arr is None:
        return []
    return [float(v) for v in np.asarray(arr).ravel()]


def _shifted_float_list(arr, offset, limit):
    if arr is None:
        return []
    values = []
    for v in np.asarray(arr).ravel():
        shifted = float(v) - float(offset)
        if 0 <= shifted < limit:
            values.append(shifted)
        else:
            values.append(float("nan"))
    return values


_DP_VIEW_PRESETS = (
    # First panel: current widget behavior.
    {
        "key": "current",
        "title": "Current",
        "cmap": None,
        "color": "#ff3b30",
        "central_color": "#00d5e8",
        "marker_scaled": True,
        "marker_size": 8.0,
        "marker_size_min": 4.0,
        "marker_size_max": 16.0,
        "show_central": True,
        "central_size": 5.0,
        "norm_upper_quantile": None,
        "norm_power": None,
        "gaussian_filter_sigma": None,
        "zoom": 1.0,
        "vmin": None,
        "vmax": None,
    },
    # These three mirror the tutorial_minimal plot_interactive_peak_map calls.
    {
        "key": "lamellar",
        "title": "Lamellar",
        "cmap": "inferno",
        "color": "#ff1f1f",
        "central_color": "#00d5e8",
        "marker_scaled": True,
        "marker_size": 8.0,
        "marker_size_min": 4.0,
        "marker_size_max": 16.0,
        "show_central": True,
        "central_size": 5.0,
        "norm_upper_quantile": 0.9999,
        "norm_power": 1.5,
        "gaussian_filter_sigma": 0.75,
        "zoom": 4.0,
        "vmin": 0.2,
        "vmax": 5.0,
    },
    {
        "key": "backbone",
        "title": "Backbone",
        "cmap": "gray",
        "color": "#7bdc3c",
        "central_color": "#00d5e8",
        "marker_scaled": True,
        "marker_size": 8.0,
        "marker_size_min": 4.0,
        "marker_size_max": 16.0,
        "show_central": True,
        "central_size": 5.0,
        "norm_upper_quantile": 0.9999,
        "norm_power": 1.5,
        "gaussian_filter_sigma": 1.5,
        "zoom": 2.0,
        "vmin": 0.035,
        "vmax": 0.11,
    },
    {
        "key": "pipi",
        "title": "pi-pi",
        "cmap": "turbo_black",
        "color": "#ff1f1f",
        "central_color": "#00d5e8",
        "marker_scaled": True,
        "marker_size": 8.0,
        "marker_size_min": 4.0,
        "marker_size_max": 16.0,
        "show_central": True,
        "central_size": 5.0,
        "norm_upper_quantile": 0.9999,
        "norm_power": 1.5,
        "gaussian_filter_sigma": 4.0,
        "zoom": 1.0,
        "vmin": 0.055,
        "vmax": 0.13,
    },
)


def _marker_px_to_mpl_s(radius_px):
    """Map the widget's on-screen circle radius (px) to matplotlib scatter ``s`` (points^2).

    The live overlay draws circles by pixel radius; ``save_peak_figures`` sizes them by
    scatter area. ~8 px maps to s=75 (the prior hard-coded save default), i.e.
    ``s = (radius * 1.08) ** 2``. Screen vs PDF differ in DPI/figsize, so parity is only
    approximate (a pre-existing display-vs-save limitation).
    """
    return float((max(0.0, float(radius_px)) * 1.08) ** 2)


def _central_px_to_scaling(radius_px):
    """Map central-beam radius (px) to ``crosshair_scaling_central_beam`` (s=120*scaling)."""
    return float(max(0.0, float(radius_px)) / 5.0)


# Display-only traits that make up a saved "subpanel visualization" preset. The
# Save-settings button snapshots these onto the source BraggPeaksPolymer, and the next
# widget built from that same object restores them (see _collect/_apply_view_settings),
# so re-opening with a different intensity_map keeps the hand-tuned look. Excludes data
# bytes, geometry, scan position, and the disk-save (save_*) traits.
_VIEW_SETTING_PANEL_LIST_TRAITS = (
    "dp_view_cmaps",
    "dp_view_colors",
    "dp_view_central_colors",
    "dp_view_marker_scaled",
    "dp_view_marker_sizes",
    "dp_view_marker_size_mins",
    "dp_view_marker_size_maxs",
    "dp_view_show_central",
    "dp_view_central_sizes",
    "dp_view_vmins",
    "dp_view_vmaxs",
    "dp_view_sigmas",
    "dp_view_powers",
    "dp_view_upper_quantiles",
    "dp_view_zooms",
)
_VIEW_SETTING_GLOBAL_TRAITS = (
    "map_cmap",
    "dp_cmap",
    "dp_vmin",
    "dp_vmax",
    "show_inset",
    "inset_size",
    "show_peaks",
    "show_polar",
    "peak_color",
    "central_color",
    "polar_cmap",
    "polar_display_vmin",
    "polar_display_vmax",
)
_VIEW_SETTING_TRAITS = _VIEW_SETTING_PANEL_LIST_TRAITS + _VIEW_SETTING_GLOBAL_TRAITS


def _collect_view_settings(widget):
    """Snapshot the widget's current subpanel display settings into a plain dict."""
    return {name: copy.deepcopy(getattr(widget, name)) for name in _VIEW_SETTING_TRAITS}


def _apply_view_settings(widget, settings):
    """Overlay a saved settings dict onto a widget's display traits, in place.

    Per-panel list traits are applied only when their length matches the widget's current
    panel count, so a stale preset (from a different panel set) can't corrupt the arrays.
    """
    n_panels = len(_DP_VIEW_PRESETS)
    panel_list = set(_VIEW_SETTING_PANEL_LIST_TRAITS)
    for name in _VIEW_SETTING_TRAITS:
        if name not in settings:
            continue
        value = settings[name]
        if name in panel_list and (value is None or len(value) != n_panels):
            continue
        setattr(widget, name, copy.deepcopy(value))


class ShowPolymer4DSTEM(anywidget.AnyWidget):
    """Live-kernel Bragg-peak / polymer 4D-STEM viewer.

    Renders a clickable real-space intensity map alongside the diffraction
    pattern at the selected scan position. When the source
    ``BraggPeaksPolymer`` has detected peaks, they are overlaid on the pattern
    (markers sized by intensity; central beam filled). When a polar transform is
    present, an optional third panel shows it with its polar peaks.

    Use the :func:`show_polymer_4DSTEM` factory rather than constructing this
    directly; the factory pulls the dataset and peak arrays off the
    ``BraggPeaksPolymer`` instance.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show_polymer_4dstem.js"

    # --- Static geometry / display config (set once) ---
    scan_height = traitlets.Int(1).tag(sync=True)        # Ry
    scan_width = traitlets.Int(1).tag(sync=True)         # Rx
    upsample_factor = traitlets.Int(1).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    map_bytes = traitlets.Bytes(b"").tag(sync=True)      # float32; (h, w) scalar or (h, w, 3) RGB
    map_height = traitlets.Int(1).tag(sync=True)
    map_width = traitlets.Int(1).tag(sync=True)
    map_vmin = traitlets.Float(0.0).tag(sync=True)
    map_vmax = traitlets.Float(1.0).tag(sync=True)
    map_cmap = traitlets.Unicode("viridis").tag(sync=True)
    map_is_rgb = traitlets.Bool(False).tag(sync=True)    # True -> draw RGB directly, skip cmap
    show_inset = traitlets.Bool(True).tag(sync=True)     # zoomed neighborhood of selected pixel
    inset_size = traitlets.Int(7).tag(sync=True)         # NxN inset window (odd)
    map_title = traitlets.Unicode("Intensity Map").tag(sync=True)

    dp_cmap = traitlets.Unicode("gray").tag(sync=True)
    dp_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    dp_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)

    has_peaks = traitlets.Bool(False).tag(sync=True)
    has_polar = traitlets.Bool(False).tag(sync=True)
    show_peaks = traitlets.Bool(True).tag(sync=True)
    show_polar = traitlets.Bool(True).tag(sync=True)

    # --- Live inference: run the model on the DP under the cursor on the fly ---
    live_inference = traitlets.Bool(False).tag(sync=True)
    # Detection threshold, exposed as a live slider in live-inference mode.
    threshold_peak = traitlets.Float(0.5).tag(sync=True)

    # --- Save current position's figures to disk (via save_peak_figures) ---
    save_dir_base = traitlets.Unicode("").tag(sync=True)          # read-only display
    save_subfolder = traitlets.Unicode("").tag(sync=True)         # user-editable, nestable ("a/b")
    save_include_map = traitlets.Bool(True).tag(sync=True)
    save_include_polar = traitlets.Bool(True).tag(sync=True)
    save_panels = traitlets.List(traitlets.Unicode()).tag(sync=True)  # which DP panels to save
    save_request = traitlets.Int(0).tag(sync=True)                # frontend increments to trigger
    save_status = traitlets.Unicode("").tag(sync=True)            # Python writes result/errors

    # --- Save/restore subpanel display settings (in-memory, on the source bp object) ---
    save_view_request = traitlets.Int(0).tag(sync=True)           # frontend increments to trigger
    view_settings_status = traitlets.Unicode("").tag(sync=True)   # Python writes confirmation/errors

    peak_color = traitlets.Unicode("#ff3b30").tag(sync=True)
    central_color = traitlets.Unicode("#ff3b30").tag(sync=True)
    peak_size_min = traitlets.Float(4.0).tag(sync=True)
    peak_size_max = traitlets.Float(16.0).tag(sync=True)

    # Polar geometry (for placing polar peaks in the polar image).
    polar_radial_bins = traitlets.Int(0).tag(sync=True)
    polar_annular_bins = traitlets.Int(0).tag(sync=True)
    max_radius_invA = traitlets.Float(0.0).tag(sync=True)
    two_fold_symmetry = traitlets.Bool(True).tag(sync=True)

    # --- Selected scan position (data coords). JS writes these on click. ---
    pos_ry = traitlets.Int(0).tag(sync=True)
    pos_rx = traitlets.Int(0).tag(sync=True)

    # --- Per-position payload (Python writes these in the observer) ---
    dp_bytes = traitlets.Bytes(b"").tag(sync=True)       # float32, dp_height x dp_width
    dp_height = traitlets.Int(1).tag(sync=True)
    dp_width = traitlets.Int(1).tag(sync=True)
    dp_data_vmin = traitlets.Float(0.0).tag(sync=True)   # auto contrast for this DP
    dp_data_vmax = traitlets.Float(1.0).tag(sync=True)
    # Monotonic counter so JS re-renders even when the new bytes hash-compare
    # equal (mirrors frame_seq in Show3D).
    payload_seq = traitlets.Int(0).tag(sync=True)

    peaks_x = traitlets.List(traitlets.Float()).tag(sync=True)
    peaks_y = traitlets.List(traitlets.Float()).tag(sync=True)
    peaks_intensity = traitlets.List(traitlets.Float()).tag(sync=True)
    central_idx = traitlets.Int(-1).tag(sync=True)
    center_y = traitlets.Float(0.0).tag(sync=True)
    center_x = traitlets.Float(0.0).tag(sync=True)

    polar_bytes = traitlets.Bytes(b"").tag(sync=True)    # float32, polar_height x polar_width
    polar_height = traitlets.Int(1).tag(sync=True)
    polar_width = traitlets.Int(1).tag(sync=True)
    polar_vmin = traitlets.Float(0.0).tag(sync=True)
    polar_vmax = traitlets.Float(1.0).tag(sync=True)
    polar_peaks_r_bin = traitlets.List(traitlets.Float()).tag(sync=True)
    polar_peaks_theta_bin = traitlets.List(traitlets.Float()).tag(sync=True)

    dp_view_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    dp_view_cmaps = traitlets.List(traitlets.Unicode()).tag(sync=True)
    dp_view_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    dp_view_central_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # Per-panel peak-marker + central-beam-marker controls (display-only; frontend applies
    # these live and the Save path forwards them to save_peak_figures). marker_scaled=True
    # sizes peak circles by intensity between marker_size_min/max; False draws them all at
    # the uniform marker_size. show_central toggles the filled central-beam dot.
    dp_view_marker_scaled = traitlets.List(traitlets.Bool()).tag(sync=True)
    dp_view_marker_sizes = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_view_marker_size_mins = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_view_marker_size_maxs = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_view_show_central = traitlets.List(traitlets.Bool()).tag(sync=True)
    dp_view_central_sizes = traitlets.List(traitlets.Float()).tag(sync=True)
    # Display-only (frontend applies these live; not observed for recompute):
    dp_view_vmins = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    dp_view_vmaxs = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    # Data-transform params (per DP panel). Editing these triggers a Python recompute
    # of that panel's float bytes (see _recompute_display); no model re-run.
    dp_view_sigmas = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    dp_view_powers = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    dp_view_upper_quantiles = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    dp_view_zooms = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)

    # Polar display-only overrides (empty/NaN sentinel = use auto). Applied on frontend.
    polar_cmap = traitlets.Unicode("").tag(sync=True)
    polar_display_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    polar_display_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)

    dp_current_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_current_height = traitlets.Int(1).tag(sync=True)
    dp_current_width = traitlets.Int(1).tag(sync=True)
    dp_current_data_vmin = traitlets.Float(0.0).tag(sync=True)
    dp_current_data_vmax = traitlets.Float(1.0).tag(sync=True)
    dp_current_peaks_x = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_current_peaks_y = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_current_peaks_intensity = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_current_central_idx = traitlets.Int(-1).tag(sync=True)
    dp_current_center_y = traitlets.Float(0.0).tag(sync=True)
    dp_current_center_x = traitlets.Float(0.0).tag(sync=True)

    dp_lamellar_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_lamellar_height = traitlets.Int(1).tag(sync=True)
    dp_lamellar_width = traitlets.Int(1).tag(sync=True)
    dp_lamellar_data_vmin = traitlets.Float(0.0).tag(sync=True)
    dp_lamellar_data_vmax = traitlets.Float(1.0).tag(sync=True)
    dp_lamellar_peaks_x = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_lamellar_peaks_y = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_lamellar_peaks_intensity = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_lamellar_central_idx = traitlets.Int(-1).tag(sync=True)
    dp_lamellar_center_y = traitlets.Float(0.0).tag(sync=True)
    dp_lamellar_center_x = traitlets.Float(0.0).tag(sync=True)

    dp_backbone_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_backbone_height = traitlets.Int(1).tag(sync=True)
    dp_backbone_width = traitlets.Int(1).tag(sync=True)
    dp_backbone_data_vmin = traitlets.Float(0.0).tag(sync=True)
    dp_backbone_data_vmax = traitlets.Float(1.0).tag(sync=True)
    dp_backbone_peaks_x = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_backbone_peaks_y = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_backbone_peaks_intensity = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_backbone_central_idx = traitlets.Int(-1).tag(sync=True)
    dp_backbone_center_y = traitlets.Float(0.0).tag(sync=True)
    dp_backbone_center_x = traitlets.Float(0.0).tag(sync=True)

    dp_pipi_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_pipi_height = traitlets.Int(1).tag(sync=True)
    dp_pipi_width = traitlets.Int(1).tag(sync=True)
    dp_pipi_data_vmin = traitlets.Float(0.0).tag(sync=True)
    dp_pipi_data_vmax = traitlets.Float(1.0).tag(sync=True)
    dp_pipi_peaks_x = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_pipi_peaks_y = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_pipi_peaks_intensity = traitlets.List(traitlets.Float()).tag(sync=True)
    dp_pipi_central_idx = traitlets.Int(-1).tag(sync=True)
    dp_pipi_center_y = traitlets.Float(0.0).tag(sync=True)
    dp_pipi_center_x = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        bragg_peaks,
        *,
        intensity_map=None,
        map_cmap="viridis",
        dp_cmap="gray",
        vmin_cartesian=None,
        vmax_cartesian=7.0,
        norm_upper_quantile=None,
        norm_power=1.0,
        intensity_field=_DEFAULT_INTENSITY_FIELD,
        show_polar=True,
        two_fold_symmetry=True,
        show_inset=True,
        inset_size=7,
        title="",
        ry=None,
        rx=None,
        live_inference=False,
        threshold_peak=0.5,
        sigma_peak_blur=1.0,
        infer_device=None,
        scan_mask=None,
        save_dir=None,
        restore_view_settings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Duck-typed: any object exposing the BraggPeaksPolymer surface works.
        if not hasattr(bragg_peaks, "dataset_cartesian"):
            raise TypeError(
                "show_polymer_4DSTEM expects a BraggPeaksPolymer (with a "
                "`dataset_cartesian` attribute); got "
                f"{type(bragg_peaks).__name__}"
            )

        self._bp = bragg_peaks
        self._intensity_field = intensity_field
        self._norm_upper_quantile = norm_upper_quantile
        self._norm_power = float(norm_power)

        # Live-inference config.
        self._live_inference = bool(live_inference)
        self._infer_device = infer_device
        self._sigma_peak_blur = float(sigma_peak_blur)

        # Cached per-position peaks + center, reused by _recompute_display so that
        # per-panel display-param edits don't re-fetch or re-run the model.
        self._px = self._py = self._ints = self._r_invA = None
        self._source_center = None

        # Save config: the raw intensity map (for the saved context panel), a title,
        # and the base output directory.
        self._intensity_map = intensity_map
        self._map_title = title or "Intensity Map"
        self._save_dir = (
            pathlib.Path(save_dir) if save_dir is not None
            else pathlib.Path.cwd() / "widget_saves"
        )

        dataset = bragg_peaks.dataset_cartesian
        Ry, Rx = int(dataset.shape[0]), int(dataset.shape[1])

        peak_coords = getattr(bragg_peaks, "peak_coordinates_cartesian", None)
        peak_ints = getattr(bragg_peaks, "peak_intensities", None)
        polar_data = getattr(bragg_peaks, "polar_data", None)
        polar_peaks = getattr(bragg_peaks, "polar_peaks", None)

        if self._live_inference:
            if getattr(bragg_peaks, "model", None) is None:
                raise ValueError(
                    "live_inference=True requires a model on the BraggPeaksPolymer "
                    "(set bragg_peaks.model and load weights first)."
                )
            # Restrict normalization + BN adaptation to the sample ROI: an explicit
            # scan_mask wins, else reuse whatever find_peaks_model stored on the object.
            mask = scan_mask if scan_mask is not None else getattr(bragg_peaks, "scan_mask", None)
            if scan_mask is not None and hasattr(type(bragg_peaks), "scan_mask"):
                bragg_peaks.scan_mask = scan_mask  # keep object + widget in agreement
            # Peaks are produced on the fly. Warm the input-normalization cache (median/iqr
            # over the ROI) so the first cursor move isn't slow. Live inference defaults to
            # bn_mode="train_batch" (runs each DP inside its find_peaks_model train-mode chunk,
            # matching the precomputed detection), so no eval-mode adapt_batchnorm is needed
            # here. Polar is a separate precompute, disabled here.
            bragg_peaks.ensure_normalization_params(device=infer_device, scan_mask=mask)
            has_peaks = True
            has_polar = False
        else:
            has_peaks = peak_coords is not None
            has_polar = bool(show_polar and polar_data is not None)

        imap, up, map_is_rgb = _resolve_intensity_map(dataset, intensity_map, (Ry, Rx))
        # RGB maps are drawn as-is; scalar maps get percentile display limits.
        mvmin, mvmax = (0.0, 1.0) if map_is_rgb else _display_limits(imap)

        with self.hold_sync():
            self.scan_height = Ry
            self.scan_width = Rx
            self.upsample_factor = up
            self.title = title
            self.map_bytes = np.ascontiguousarray(imap, dtype=np.float32).tobytes()
            self.map_height = int(imap.shape[0])
            self.map_width = int(imap.shape[1])
            self.map_vmin = mvmin
            self.map_vmax = mvmax
            self.map_cmap = map_cmap
            self.map_is_rgb = bool(map_is_rgb)
            self.show_inset = bool(show_inset)
            self.inset_size = max(1, int(inset_size) | 1)  # force odd so a center cell exists
            self.dp_cmap = dp_cmap
            self.dp_vmin = None if vmin_cartesian is None else float(vmin_cartesian)
            self.dp_vmax = None if vmax_cartesian is None else float(vmax_cartesian)
            self.dp_view_titles = [p["title"] for p in _DP_VIEW_PRESETS]
            self.dp_view_cmaps = [p["cmap"] or dp_cmap for p in _DP_VIEW_PRESETS]
            self.dp_view_colors = [p["color"] for p in _DP_VIEW_PRESETS]
            self.dp_view_central_colors = [p["central_color"] for p in _DP_VIEW_PRESETS]
            self.dp_view_marker_scaled = [bool(p["marker_scaled"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_marker_sizes = [float(p["marker_size"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_marker_size_mins = [float(p["marker_size_min"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_marker_size_maxs = [float(p["marker_size_max"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_show_central = [bool(p["show_central"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_central_sizes = [float(p["central_size"]) for p in _DP_VIEW_PRESETS]
            self.dp_view_vmins = [
                self.dp_vmin if p["vmin"] is None else float(p["vmin"])
                for p in _DP_VIEW_PRESETS
            ]
            self.dp_view_vmaxs = [
                self.dp_vmax if p["vmax"] is None else float(p["vmax"])
                for p in _DP_VIEW_PRESETS
            ]
            # Per-panel transform params. The "current" panel is seeded from the
            # widget-level norm settings (its preset leaves them None); the rest use
            # their preset values. These become editable via the per-panel modal.
            def _panel_qtl(p):
                return self._norm_upper_quantile if p["key"] == "current" else p["norm_upper_quantile"]

            def _panel_power(p):
                return self._norm_power if p["key"] == "current" else p["norm_power"]

            self.dp_view_sigmas = [
                None if p["gaussian_filter_sigma"] is None else float(p["gaussian_filter_sigma"])
                for p in _DP_VIEW_PRESETS
            ]
            self.dp_view_powers = [
                None if _panel_power(p) is None else float(_panel_power(p))
                for p in _DP_VIEW_PRESETS
            ]
            self.dp_view_upper_quantiles = [
                None if _panel_qtl(p) is None else float(_panel_qtl(p))
                for p in _DP_VIEW_PRESETS
            ]
            self.dp_view_zooms = [float(p["zoom"]) for p in _DP_VIEW_PRESETS]
            self.has_peaks = has_peaks
            self.has_polar = has_polar
            self.show_peaks = has_peaks
            self.show_polar = has_polar
            self.live_inference = self._live_inference
            self.threshold_peak = float(threshold_peak)
            self.save_dir_base = str(self._save_dir)
            self.save_panels = [p["key"] for p in _DP_VIEW_PRESETS]
            self.two_fold_symmetry = bool(two_fold_symmetry)
            if has_polar:
                self.polar_radial_bins = int(getattr(bragg_peaks, "num_radial_bins", 0) or 0)
                self.polar_annular_bins = int(getattr(bragg_peaks, "num_annular_bins", 0) or 0)
                self.max_radius_invA = float(getattr(bragg_peaks, "max_radius_invA", 0.0) or 0.0)

            # Restore previously-saved subpanel display settings (in-memory, from the
            # source bp). Applied after presets/kwargs so a saved look wins, and before
            # _update_payload so restored transform params (sigma/power/quantile/zoom)
            # feed the first recompute.
            saved = getattr(bragg_peaks, "_widget_view_settings", None)
            if restore_view_settings and saved:
                _apply_view_settings(self, saved)

            # Initial selected position (center of scan, in DATA coords).
            self.pos_ry = Ry // 2 if ry is None else int(ry)
            self.pos_rx = Rx // 2 if rx is None else int(rx)
            self._update_payload()

        self.observe(self._on_pos_change, names=["pos_ry", "pos_rx", "threshold_peak"])
        # Per-panel transform-param edits only need a display recompute (no re-fetch /
        # no model re-run). vmin/vmax/cmap are display-only (frontend applies them).
        self.observe(
            self._on_display_change,
            names=["dp_view_sigmas", "dp_view_powers", "dp_view_upper_quantiles", "dp_view_zooms"],
        )
        self.observe(self._on_save_request, names=["save_request"])
        self.observe(self._on_save_view_request, names=["save_view_request"])

    # -- live-kernel recompute ------------------------------------------------
    def _on_pos_change(self, _change):
        with self.hold_sync():
            self._update_payload()

    def _on_display_change(self, _change):
        with self.hold_sync():
            self._recompute_display()

    # -- save current position's figures -------------------------------------
    def _on_save_request(self, _change):
        try:
            self._save_current_position()
        except Exception as exc:  # surface the failure in the widget UI
            self.save_status = f"Save failed: {exc}"

    # -- save subpanel display settings (in-memory, on the source bp) ---------
    def _on_save_view_request(self, _change):
        try:
            self._bp._widget_view_settings = _collect_view_settings(self)
            self.view_settings_status = "Saved display settings ✓"
        except Exception as exc:  # surface the failure in the widget UI
            self.view_settings_status = f"Save failed: {exc}"

    def _save_current_position(self):
        bp = self._bp
        ry = max(0, min(int(self.pos_ry), self.scan_height - 1))
        rx = max(0, min(int(self.pos_rx), self.scan_width - 1))

        sub = (self.save_subfolder or "").strip().strip("/")
        dest = (self._save_dir / sub / f"ry{ry}_rx{rx}") if sub else (self._save_dir / f"ry{ry}_rx{rx}")

        # In live mode there are no scan-wide peak arrays; inject the cached live peaks.
        live_kw = {}
        if self._live_inference:
            live_kw = dict(
                peaks_x=self._px, peaks_y=self._py, peak_ints=self._ints, peaks_r_invA=None
            )

        preset_index = {p["key"]: i for i, p in enumerate(_DP_VIEW_PRESETS)}
        keys = list(self.save_panels) if self.save_panels else list(preset_index)

        def _panel_kwargs(i):
            power = self.dp_view_powers[i]
            # Mirror the frontend's effective-contrast fallback so the PDF matches what is
            # displayed. The live view resolves each limit as:
            #   per-panel override (dp_view_vmins[i]) -> global (dp_vmin) -> auto
            # where "auto" is the 1st/99th percentile of the processed DP
            # (dp_{key}_data_vmin/vmax, from _display_limits). A bare None here would leave
            # matplotlib to autoscale to full min/max, which the hot central beam compresses
            # into a dark image -- the source of the "save darker than widget" mismatch.
            key = _DP_VIEW_PRESETS[i]["key"]

            def _eff(per_panel, global_val, auto_attr):
                if per_panel is not None:
                    return per_panel
                if global_val is not None:
                    return global_val
                return getattr(self, auto_attr, None)

            vmin = _eff(self.dp_view_vmins[i], self.dp_vmin, f"dp_{key}_data_vmin")
            vmax = _eff(self.dp_view_vmaxs[i], self.dp_vmax, f"dp_{key}_data_vmax")
            kw = dict(
                dp_cmap=self.dp_view_cmaps[i],
                vmin_cartesian=vmin,
                vmax_cartesian=vmax,
                norm_upper_quantile=self.dp_view_upper_quantiles[i],
                norm_power=1.0 if power is None else float(power),
                gaussian_filter_sigma=self.dp_view_sigmas[i],
                zoom=self.dp_view_zooms[i],
                selected_peak_color=self.dp_view_colors[i],
                central_beam_color=self.dp_view_central_colors[i],
                crosshair_width_peaks=2,
                # Match the live overlay: rings/central dot are fully opaque, and the
                # central-beam dot is stroked at 1.5 px (see drawDot in the frontend).
                peak_alpha=1.0,
                central_linewidth=1.5,
                show_central_beam=bool(self.dp_view_show_central[i]),
                crosshair_scaling_central_beam=_central_px_to_scaling(self.dp_view_central_sizes[i]),
            )
            # Uniform vs intensity-scaled peak-marker sizing (mirrors the live overlay).
            if self.dp_view_marker_scaled[i]:
                kw["peak_marker_size"] = None
                kw["peak_size_range"] = (
                    _marker_px_to_mpl_s(self.dp_view_marker_size_mins[i]),
                    _marker_px_to_mpl_s(self.dp_view_marker_size_maxs[i]),
                )
            else:
                kw["peak_marker_size"] = _marker_px_to_mpl_s(self.dp_view_marker_sizes[i])
            return kw

        n_saved = 0
        for key in keys:
            i = preset_index.get(key)
            if i is None:
                continue
            bp.save_peak_figures(
                ry, rx,
                intensity_map=self._intensity_map,
                map_title=self._map_title,
                prefix=key,
                save_dir=str(dest),
                save_intensity_map=False,
                save_diffraction=True,
                save_polar=False,
                **_panel_kwargs(i),
                **live_kw,
            )
            n_saved += 1

        # Context map once (uses the map's own colormap; peaks not drawn on the map).
        if self.save_include_map:
            bp.save_peak_figures(
                ry, rx,
                intensity_map=self._intensity_map,
                map_title=self._map_title,
                prefix="context",
                save_dir=str(dest),
                map_cmap=self.map_cmap,
                save_intensity_map=True,
                save_diffraction=False,
                save_polar=False,
                **live_kw,
            )

        # Polar once (precomputed mode only; live mode has has_polar=False).
        if self.save_include_polar and self.has_polar:
            i = preset_index.get("current", 0)
            power = self.dp_view_powers[i]
            bp.save_peak_figures(
                ry, rx,
                intensity_map=self._intensity_map,
                map_title=self._map_title,
                prefix="polar",
                save_dir=str(dest),
                dp_cmap=self.dp_view_cmaps[i],
                norm_upper_quantile=self.dp_view_upper_quantiles[i],
                norm_power=1.0 if power is None else float(power),
                gaussian_filter_sigma=self.dp_view_sigmas[i],
                save_intensity_map=False,
                save_diffraction=False,
                save_polar=True,
            )

        self.save_status = f"Saved {n_saved} panel(s) to {dest.resolve()}"

    def _update_payload(self):
        """Fetch the position-dependent data (base DP + peaks, live or precomputed),
        cache the peaks, then render the panels via _recompute_display."""
        bp = self._bp
        dataset = bp.dataset_cartesian
        ry = max(0, min(int(self.pos_ry), self.scan_height - 1))
        rx = max(0, min(int(self.pos_rx), self.scan_width - 1))

        base_dp = _normalized_dp(
            dataset,
            ry,
            rx,
            norm_upper_quantile=self._norm_upper_quantile,
            norm_power=self._norm_power,
        )
        base_vmin, base_vmax = _display_limits(base_dp)
        self.dp_bytes = np.ascontiguousarray(base_dp, dtype=np.float32).tobytes()
        self.dp_height = int(base_dp.shape[0])
        self.dp_width = int(base_dp.shape[1])
        self.dp_data_vmin = base_vmin
        self.dp_data_vmax = base_vmax

        center = _display_center(
            getattr(bp, "image_centers", None), ry, rx, base_dp.shape
        )
        self._source_center = center
        self.center_y, self.center_x = float(center[0]), float(center[1])

        px = py = ints = r_invA = None
        if self._live_inference:
            # Run the model on this single DP; no precomputed peaks / polar peaks.
            res = bp.infer_peaks_single(
                ry,
                rx,
                device=self._infer_device,
                sigma_peak_blur=self._sigma_peak_blur,
                threshold_peak=float(self.threshold_peak),
            )
            px, py, ints = res["x_pixels"], res["y_pixels"], res["intensities"]
        elif self.has_peaks:
            px = bp.peak_coordinates_cartesian["x_pixels"][ry, rx]
            py = bp.peak_coordinates_cartesian["y_pixels"][ry, rx]
            if getattr(bp, "peak_intensities", None) is not None:
                ints = bp.peak_intensities[self._intensity_field][ry, rx]
            if getattr(bp, "polar_peaks", None) is not None:
                r_invA = bp.polar_peaks["r_invA"][ry, rx]

        # Cache the peaks so per-panel display-param edits can re-render without
        # re-fetching / re-running the model.
        self._px, self._py, self._ints, self._r_invA = px, py, ints, r_invA

        if px is not None and len(_as_float_list(px)) > 0:
            self.peaks_x = _as_float_list(px)
            self.peaks_y = _as_float_list(py)
            self.peaks_intensity = _as_float_list(ints)
            self.central_idx = _central_peak_index(
                px, py, r_invA, center, max_dist=_central_beam_max_dist(base_dp.shape)
            )
        else:
            self.peaks_x, self.peaks_y, self.peaks_intensity = [], [], []
            self.central_idx = -1

        self._recompute_display()

    def _recompute_display(self):
        """Rebuild the DP-view panels (and polar image) from the cached peaks + the
        current per-panel display params. Does NOT re-run inference or re-read peaks,
        so it's cheap enough to fire on every transform-param slider tick."""
        bp = self._bp
        dataset = bp.dataset_cartesian
        ry = max(0, min(int(self.pos_ry), self.scan_height - 1))
        rx = max(0, min(int(self.pos_rx), self.scan_width - 1))
        source_center = getattr(self, "_source_center", None)
        if source_center is None:
            source_center = _display_center(
                getattr(bp, "image_centers", None), ry, rx, (self.dp_height, self.dp_width)
            )

        self._update_dp_views(
            dataset, ry, rx, self._px, self._py, self._ints, self._r_invA, source_center
        )

        if self.has_polar:
            polar = np.asarray(
                to_numpy(bp.polar_data["intensity"][ry, rx]), dtype=np.float32
            ).T  # (theta, radius), matching plot_interactive_peak_map's `.T`
            pvmin, pvmax = _display_limits(polar)
            self.polar_bytes = np.ascontiguousarray(polar, dtype=np.float32).tobytes()
            self.polar_height = int(polar.shape[0])
            self.polar_width = int(polar.shape[1])
            self.polar_vmin = pvmin
            self.polar_vmax = pvmax
            self._update_polar_peaks(ry, rx)

        self.payload_seq += 1

    def _update_dp_views(self, dataset, ry, rx, px, py, ints, r_invA, source_center):
        for i, preset in enumerate(_DP_VIEW_PRESETS):
            key = preset["key"]
            # Per-panel transform params come from the (editable) trait lists, not the
            # frozen presets. zoom=1 + sigma=None reduces _dp_view to plain normalization
            # (so the "current" panel matches its old behavior until the user edits it).
            power = self.dp_view_powers[i]
            dp, x0, y0 = _dp_view(
                dataset,
                ry,
                rx,
                norm_upper_quantile=self.dp_view_upper_quantiles[i],
                norm_power=1.0 if power is None else float(power),
                gaussian_filter_sigma=self.dp_view_sigmas[i],
                zoom=self.dp_view_zooms[i],
                center=source_center,
            )

            dvmin, dvmax = _display_limits(dp)
            setattr(self, f"dp_{key}_bytes", np.ascontiguousarray(dp, dtype=np.float32).tobytes())
            setattr(self, f"dp_{key}_height", int(dp.shape[0]))
            setattr(self, f"dp_{key}_width", int(dp.shape[1]))
            setattr(self, f"dp_{key}_data_vmin", dvmin)
            setattr(self, f"dp_{key}_data_vmax", dvmax)
            setattr(self, f"dp_{key}_center_y", float(source_center[0]) - float(y0))
            setattr(self, f"dp_{key}_center_x", float(source_center[1]) - float(x0))

            if self.has_peaks:
                shifted_x = _shifted_float_list(px, x0, dp.shape[1])
                shifted_y = _shifted_float_list(py, y0, dp.shape[0])
                keep = [
                    i
                    for i, (x, y) in enumerate(zip(shifted_x, shifted_y))
                    if np.isfinite(x) and np.isfinite(y)
                ]
                peak_x = [shifted_x[i] for i in keep]
                peak_y = [shifted_y[i] for i in keep]
                peak_i_source = _as_float_list(ints)
                peak_i = [peak_i_source[i] if i < len(peak_i_source) else 1.0 for i in keep]
                central_source = _central_peak_index(
                    px, py, r_invA, source_center,
                    max_dist=_central_beam_max_dist((self.dp_height, self.dp_width)),
                )
                try:
                    central_idx = keep.index(central_source)
                except ValueError:
                    central_idx = -1
                setattr(self, f"dp_{key}_peaks_x", peak_x)
                setattr(self, f"dp_{key}_peaks_y", peak_y)
                setattr(self, f"dp_{key}_peaks_intensity", peak_i)
                setattr(self, f"dp_{key}_central_idx", central_idx)
            else:
                setattr(self, f"dp_{key}_peaks_x", [])
                setattr(self, f"dp_{key}_peaks_y", [])
                setattr(self, f"dp_{key}_peaks_intensity", [])
                setattr(self, f"dp_{key}_central_idx", -1)

    def _update_polar_peaks(self, ry, rx):
        bp = self._bp
        polar_peaks = getattr(bp, "polar_peaks", None)
        if polar_peaks is None or not self.max_radius_invA or not self.polar_radial_bins:
            self.polar_peaks_r_bin, self.polar_peaks_theta_bin = [], []
            return
        r = np.asarray(polar_peaks["r_invA"][ry, rx]).ravel()
        theta = np.asarray(polar_peaks["theta"][ry, rx]).ravel()
        if r.size == 0:
            self.polar_peaks_r_bin, self.polar_peaks_theta_bin = [], []
            return
        r_bins = r / self.max_radius_invA * self.polar_radial_bins
        period = np.pi if self.two_fold_symmetry else 2 * np.pi
        theta_bins = theta / period * self.polar_annular_bins
        self.polar_peaks_r_bin = _as_float_list(r_bins)
        self.polar_peaks_theta_bin = _as_float_list(theta_bins)


def show_polymer_4DSTEM(bragg_peaks, **kwargs):
    """Open the interactive Bragg-peak / polymer 4D-STEM viewer.

    Parameters
    ----------
    bragg_peaks : BraggPeaksPolymer
        A ``quantem.diffraction.BraggPeaksPolymer`` instance. Its
        ``dataset_cartesian`` is required; ``peak_coordinates_cartesian`` /
        ``peak_intensities`` enable the peak overlay, and ``polar_data`` /
        ``polar_peaks`` enable the polar panel. Any are optional.
    intensity_map : ndarray, optional
        Real-space map for the left panel (may be upsampled by an integer
        factor). A 2D ``(H, W)`` array is colormapped via ``map_cmap``; a 3D
        ``(H, W, 3)`` (or ``(H, W, 4)``) array is drawn as a true RGB(A) image
        (channels clipped to ``[0, 1]``, ``map_cmap`` ignored). Defaults to the
        mean detector intensity per scan position.
    map_cmap, dp_cmap : str
        Colormaps for the (scalar) intensity map and diffraction pattern.
        ``map_cmap`` is ignored when ``intensity_map`` is an RGB image.
    vmin_cartesian, vmax_cartesian : float, optional
        Fixed contrast for the diffraction pattern. ``vmax_cartesian`` defaults
        to 7.0 (matching ``plot_interactive_peak_map``); pass ``None`` for
        per-pattern auto-contrast.
    norm_upper_quantile, norm_power : float, optional
        Same diffraction-pattern normalization knobs as the matplotlib viewer.
    show_polar : bool, default True
        Show the polar-transform panel when polar data is available.
    show_inset : bool, default True
        Show a zoomed ``inset_size`` x ``inset_size`` neighborhood of the
        intensity map around the selected position, with the central (selected)
        pixel outlined in green. Stacked under the map; toggleable in the UI.
    inset_size : int, default 7
        Side length (in map pixels) of the square inset; forced odd so a single
        center pixel exists.
    ry, rx : int, optional
        Initial scan position (defaults to the scan center).
    live_inference : bool, default False
        Run the model on the diffraction pattern under the cursor on the fly
        (via ``bragg_peaks.infer_peaks_single``) instead of reading precomputed
        ``peak_coordinates_cartesian``. Requires ``bragg_peaks.model`` (with loaded
        weights). Peaks update live as you drag; the polar panel is disabled in this
        mode. Does not require having run ``find_peaks_model`` first.
    threshold_peak : float, default 0.5
        Peak-detection threshold. In live mode this is exposed as a slider that
        re-runs detection on the current pattern instantly.
    sigma_peak_blur : float, default 1.0
        Gaussian blur sigma applied to the model's position map before peak detection
        (live mode).
    infer_device : str, optional
        Device for live inference (defaults to the ``BraggPeaksPolymer`` device).
    scan_mask : ndarray, optional
        Boolean ``(Ry, Rx)`` region-of-interest mask for live-inference normalization +
        BatchNorm adaptation, so stats come from the sample region (not vacuum/edges). If
        omitted, reuses the mask ``find_peaks_model`` stored on the object; if none exists,
        the whole scan is used. Pass e.g. ``scan_mask=mask['mask']`` from
        ``create_interactive_circular_mask`` when using live mode without ``find_peaks_model``.
    restore_view_settings : bool, default True
        Restore subpanel display settings previously captured by the widget's
        "Save settings" button (stored in-memory on this ``bragg_peaks`` object). When
        settings have been saved they take precedence over the display kwargs above, so
        recalling the widget with a different ``intensity_map`` keeps the tuned look. Pass
        ``False`` for a fresh widget at preset defaults, or clear with
        ``bragg_peaks._widget_view_settings = None``.

    Returns
    -------
    ShowPolymer4DSTEM
        The widget; display it as the last expression in a notebook cell.

    Examples
    --------
    >>> from quantem.widget import show_polymer_4DSTEM
    >>> w = show_polymer_4DSTEM(bragg_peaks)            # doctest: +SKIP
    >>> w = show_polymer_4DSTEM(bragg_peaks, vmax_cartesian=None)  # doctest: +SKIP
    """
    if getattr(bragg_peaks, "image_centers", None) is None:
        warnings.warn(
            "bragg_peaks.image_centers is None: the central-beam marker will fall back to "
            "the geometric image center, not the calibrated beam center. Run "
            "bragg_peaks.process_polar(...) (or find_central_beams_4d / load_image_centers) "
            "before show_widget to use the pipeline (Karen's angular-uniformity) center.",
            stacklevel=2,
        )
    return ShowPolymer4DSTEM(bragg_peaks, **kwargs)
