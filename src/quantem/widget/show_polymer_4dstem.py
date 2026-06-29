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

import pathlib

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
    """Return (map_float32, upsample_factor). None -> mean map at upsample 1."""
    Ry, Rx = scan_shape
    if intensity_map is None:
        return _mean_intensity_map(dataset_cartesian, scan_shape), 1
    arr = np.asarray(to_numpy(intensity_map))
    if arr.ndim == 3:
        # RGB(A) map: collapse to luminance for the grayscale colormap path.
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"intensity_map must be 2D (or RGB), got shape {arr.shape}")
    up = arr.shape[0] // Ry
    if up < 1 or arr.shape[0] % Ry or arr.shape[1] % Rx or arr.shape[1] // Rx != up:
        raise ValueError(
            f"intensity_map shape {arr.shape} is not an integer multiple of "
            f"the scan grid ({Ry}, {Rx})"
        )
    return arr.astype(np.float32, copy=False), int(up)


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


def _dp_view(dataset_cartesian, ry, rx, *, norm_upper_quantile, norm_power, gaussian_filter_sigma, zoom):
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
        y0 = max(0, (h - crop_h) // 2)
        x0 = max(0, (w - crop_w) // 2)
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


def _central_peak_index(px, py, r_invA, center):
    """Index of the central beam: smallest r, else nearest to center."""
    if not _has_peaks(px, py):
        return -1
    if r_invA is not None and len(r_invA) > 0:
        return int(np.argmin(r_invA))
    cy, cx = center
    return int(np.argmin((np.asarray(px) - cx) ** 2 + (np.asarray(py) - cy) ** 2))


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
        "cmap": "gray",
        "color": "#ffee00",
        "central_color": "#00d5e8",
        "norm_upper_quantile": 0.9999,
        "norm_power": 1.0,
        "gaussian_filter_sigma": 4.0,
        "zoom": 1.0,
        "vmin": 0.1,
        "vmax": 1.0,
    },
)


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

    map_bytes = traitlets.Bytes(b"").tag(sync=True)      # float32, map_height x map_width
    map_height = traitlets.Int(1).tag(sync=True)
    map_width = traitlets.Int(1).tag(sync=True)
    map_vmin = traitlets.Float(0.0).tag(sync=True)
    map_vmax = traitlets.Float(1.0).tag(sync=True)
    map_cmap = traitlets.Unicode("viridis").tag(sync=True)
    map_title = traitlets.Unicode("Intensity Map").tag(sync=True)

    dp_cmap = traitlets.Unicode("gray").tag(sync=True)
    dp_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    dp_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)

    has_peaks = traitlets.Bool(False).tag(sync=True)
    has_polar = traitlets.Bool(False).tag(sync=True)
    show_peaks = traitlets.Bool(True).tag(sync=True)
    show_polar = traitlets.Bool(True).tag(sync=True)

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
    dp_view_vmins = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)
    dp_view_vmaxs = traitlets.List(traitlets.Float(allow_none=True), allow_none=False).tag(sync=True)

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
        title="",
        ry=None,
        rx=None,
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

        dataset = bragg_peaks.dataset_cartesian
        Ry, Rx = int(dataset.shape[0]), int(dataset.shape[1])

        peak_coords = getattr(bragg_peaks, "peak_coordinates_cartesian", None)
        peak_ints = getattr(bragg_peaks, "peak_intensities", None)
        polar_data = getattr(bragg_peaks, "polar_data", None)
        polar_peaks = getattr(bragg_peaks, "polar_peaks", None)

        has_peaks = peak_coords is not None
        has_polar = bool(show_polar and polar_data is not None)

        imap, up = _resolve_intensity_map(dataset, intensity_map, (Ry, Rx))
        mvmin, mvmax = _display_limits(imap)

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
            self.dp_cmap = dp_cmap
            self.dp_vmin = None if vmin_cartesian is None else float(vmin_cartesian)
            self.dp_vmax = None if vmax_cartesian is None else float(vmax_cartesian)
            self.dp_view_titles = [p["title"] for p in _DP_VIEW_PRESETS]
            self.dp_view_cmaps = [p["cmap"] or dp_cmap for p in _DP_VIEW_PRESETS]
            self.dp_view_colors = [p["color"] for p in _DP_VIEW_PRESETS]
            self.dp_view_central_colors = [p["central_color"] for p in _DP_VIEW_PRESETS]
            self.dp_view_vmins = [
                self.dp_vmin if p["vmin"] is None else float(p["vmin"])
                for p in _DP_VIEW_PRESETS
            ]
            self.dp_view_vmaxs = [
                self.dp_vmax if p["vmax"] is None else float(p["vmax"])
                for p in _DP_VIEW_PRESETS
            ]
            self.has_peaks = has_peaks
            self.has_polar = has_polar
            self.show_peaks = has_peaks
            self.show_polar = has_polar
            self.two_fold_symmetry = bool(two_fold_symmetry)
            if has_polar:
                self.polar_radial_bins = int(getattr(bragg_peaks, "num_radial_bins", 0) or 0)
                self.polar_annular_bins = int(getattr(bragg_peaks, "num_annular_bins", 0) or 0)
                self.max_radius_invA = float(getattr(bragg_peaks, "max_radius_invA", 0.0) or 0.0)

            # Initial selected position (center of scan, in DATA coords).
            self.pos_ry = Ry // 2 if ry is None else int(ry)
            self.pos_rx = Rx // 2 if rx is None else int(rx)
            self._update_payload()

        self.observe(self._on_pos_change, names=["pos_ry", "pos_rx"])

    # -- live-kernel recompute ------------------------------------------------
    def _on_pos_change(self, _change):
        with self.hold_sync():
            self._update_payload()

    def _update_payload(self):
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

        source_center = _display_center(
            getattr(bp, "image_centers", None), ry, rx, base_dp.shape
        )
        center = _display_center(
            getattr(bp, "image_centers", None), ry, rx, base_dp.shape
        )
        self.center_y, self.center_x = float(center[0]), float(center[1])

        px = py = ints = r_invA = None
        if self.has_peaks:
            px = bp.peak_coordinates_cartesian["x_pixels"][ry, rx]
            py = bp.peak_coordinates_cartesian["y_pixels"][ry, rx]
            if getattr(bp, "peak_intensities", None) is not None:
                ints = bp.peak_intensities[self._intensity_field][ry, rx]
            if getattr(bp, "polar_peaks", None) is not None:
                r_invA = bp.polar_peaks["r_invA"][ry, rx]
            self.peaks_x = _as_float_list(px)
            self.peaks_y = _as_float_list(py)
            self.peaks_intensity = _as_float_list(ints)
            self.central_idx = _central_peak_index(px, py, r_invA, center)
        else:
            self.peaks_x, self.peaks_y, self.peaks_intensity = [], [], []
            self.central_idx = -1

        self._update_dp_views(dataset, ry, rx, px, py, ints, r_invA, source_center)

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
        for preset in _DP_VIEW_PRESETS:
            key = preset["key"]
            if key == "current":
                dp = _normalized_dp(
                    dataset,
                    ry,
                    rx,
                    norm_upper_quantile=self._norm_upper_quantile,
                    norm_power=self._norm_power,
                )
                x0 = y0 = 0
            else:
                dp, x0, y0 = _dp_view(
                    dataset,
                    ry,
                    rx,
                    norm_upper_quantile=preset["norm_upper_quantile"],
                    norm_power=preset["norm_power"],
                    gaussian_filter_sigma=preset["gaussian_filter_sigma"],
                    zoom=preset["zoom"],
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
                central_source = _central_peak_index(px, py, r_invA, source_center)
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
        2D real-space map for the left panel (may be upsampled by an integer
        factor). Defaults to the mean detector intensity per scan position.
    map_cmap, dp_cmap : str
        Colormaps for the intensity map and diffraction pattern.
    vmin_cartesian, vmax_cartesian : float, optional
        Fixed contrast for the diffraction pattern. ``vmax_cartesian`` defaults
        to 7.0 (matching ``plot_interactive_peak_map``); pass ``None`` for
        per-pattern auto-contrast.
    norm_upper_quantile, norm_power : float, optional
        Same diffraction-pattern normalization knobs as the matplotlib viewer.
    show_polar : bool, default True
        Show the polar-transform panel when polar data is available.
    ry, rx : int, optional
        Initial scan position (defaults to the scan center).

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
    return ShowPolymer4DSTEM(bragg_peaks, **kwargs)
