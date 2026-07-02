// show_polymer_4dstem: live-kernel Bragg-peak / polymer 4D-STEM viewer.
//
// Port of BraggPeaksPolymer.plot_interactive_peak_map / plot_interactive_image_map.
// Left panel: drag-selectable real-space intensity map (drag or click selects a
// scan position). Middle panel: the diffraction pattern at that position, with the
// detected Bragg peaks overlaid (hollow markers sized by intensity, the central
// beam filled). Right panel (optional): the polar transform with its polar peaks.
//
// All heavy data is recomputed in the Python kernel on every scan-position
// update and shipped over the comm; this file only colormaps + draws.
import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import { COLORMAPS, applyColormap } from "../colormaps";

const { useRef, useEffect, useMemo, useCallback, useState } = React;

// Bytes traits arrive as a DataView (anywidget). Reinterpret as Float32Array.
function asFloat32(value: unknown): Float32Array {
  if (!value) return new Float32Array(0);
  if (value instanceof Float32Array) return value;
  if (value instanceof DataView) {
    return new Float32Array(value.buffer, value.byteOffset, Math.floor(value.byteLength / 4));
  }
  if (value instanceof ArrayBuffer) return new Float32Array(value);
  const v = value as { buffer?: ArrayBuffer; byteOffset?: number; byteLength?: number };
  if (v.buffer instanceof ArrayBuffer) {
    return new Float32Array(v.buffer, v.byteOffset ?? 0, Math.floor((v.byteLength ?? v.buffer.byteLength) / 4));
  }
  return new Float32Array(0);
}

function lutFor(name: string): Uint8Array {
  return COLORMAPS[name] ?? COLORMAPS["viridis"] ?? COLORMAPS["gray"];
}

// Colormap (data, h, w) into an offscreen canvas at native resolution.
function colormapToCanvas(
  data: Float32Array, w: number, h: number, lut: Uint8Array, vmin: number, vmax: number,
): HTMLCanvasElement | null {
  if (!data.length || w < 1 || h < 1 || data.length < w * h) return null;
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const ctx = off.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w, h);
  applyColormap(data.subarray(0, w * h), img.data, lut, vmin, vmax);
  ctx.putImageData(img, 0, 0);
  return off;
}

// Draw interleaved RGB float data (length w*h*3, channels in [0,1]) directly,
// bypassing the colormap. Values are scaled to bytes; img.data clamps to [0,255].
function rgbToCanvas(
  data: Float32Array, w: number, h: number,
): HTMLCanvasElement | null {
  const n = w * h;
  if (!data.length || w < 1 || h < 1 || data.length < n * 3) return null;
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const ctx = off.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w, h);
  const out = img.data;
  for (let i = 0; i < n; i++) {
    out[i * 4] = data[i * 3] * 255;
    out[i * 4 + 1] = data[i * 3 + 1] * 255;
    out[i * 4 + 2] = data[i * 3 + 2] * 255;
    out[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

interface ImagePanelProps {
  data: Float32Array;
  width: number;
  height: number;
  cmap: string;
  vmin: number;
  vmax: number;
  // When true, `data` is interleaved RGB (length w*h*3) drawn without a colormap.
  isRgb?: boolean;
  displayWidth: number;
  title: string;
  // Pixels-per-data-pixel marker overlay, drawn in NATIVE image coords.
  overlay?: (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => void;
  onPick?: (col: number, row: number) => void;
  cursor?: { col: number; row: number } | null;
  cursorColor?: string;
  aspectAuto?: boolean;
  // When provided, an "Adjust" button is shown next to the title to open the editor.
  onAdjust?: () => void;
}

interface DpView {
  key: string;
  title: string;
  data: Float32Array;
  width: number;
  height: number;
  cmap: string;
  vmin: number | null;
  vmax: number | null;
  dataVmin: number;
  dataVmax: number;
  peaksX: number[];
  peaksY: number[];
  peaksIntensity: number[];
  centralIdx: number;
  centerY: number;
  centerX: number;
  peakColor: string;
  centralColor: string;
}

function ImagePanel(props: ImagePanelProps) {
  const {
    data, width, height, cmap, vmin, vmax, isRgb, displayWidth, title,
    overlay, onPick, cursor, cursorColor, aspectAuto, onAdjust,
  } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isPickingRef = useRef(false);
  const lastPickRef = useRef<{ col: number; row: number } | null>(null);

  const dispW = displayWidth;
  const dispH = aspectAuto
    ? Math.round(displayWidth * 0.5)
    : Math.round((displayWidth * height) / Math.max(1, width));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(dispW * dpr);
    canvas.height = Math.round(dispH * dpr);
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.imageSmoothingEnabled = false;

    const off = isRgb
      ? rgbToCanvas(data, width, height)
      : colormapToCanvas(data, width, height, lutFor(cmap), vmin, vmax);
    const scaleX = dispW / Math.max(1, width);
    const scaleY = dispH / Math.max(1, height);
    if (off) ctx.drawImage(off, 0, 0, dispW, dispH);

    if (overlay) {
      ctx.save();
      overlay(ctx, scaleX, scaleY);
      ctx.restore();
    }

    if (cursor) {
      const cx = (cursor.col + 0.5) * scaleX;
      const cy = (cursor.row + 0.5) * scaleY;
      ctx.strokeStyle = cursorColor ?? "#ff3b30";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }, [data, width, height, cmap, vmin, vmax, isRgb, dispW, dispH, overlay, cursor, cursorColor]);

  const pickFromPointer = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!onPick) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const col = Math.max(
        0,
        Math.min(width - 1, Math.floor(((e.clientX - rect.left) / rect.width) * width)),
      );
      const row = Math.max(
        0,
        Math.min(height - 1, Math.floor(((e.clientY - rect.top) / rect.height) * height)),
      );
      const last = lastPickRef.current;
      if (last && last.col === col && last.row === row) return;
      lastPickRef.current = { col, row };
      onPick(col, row);
    },
    [onPick, width, height],
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!onPick || e.button !== 0) return;
      e.preventDefault();
      isPickingRef.current = true;
      lastPickRef.current = null;
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch {}
      pickFromPointer(e);
    },
    [onPick, pickFromPointer],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isPickingRef.current) return;
      e.preventDefault();
      pickFromPointer(e);
    },
    [pickFromPointer],
  );

  const stopPicking = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isPickingRef.current) return;
      isPickingRef.current = false;
      lastPickRef.current = null;
      try { e.currentTarget.releasePointerCapture(e.pointerId); } catch {}
    },
    [],
  );

  const pickFromMouse = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPick) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const col = Math.max(
        0,
        Math.min(width - 1, Math.floor(((e.clientX - rect.left) / rect.width) * width)),
      );
      const row = Math.max(
        0,
        Math.min(height - 1, Math.floor(((e.clientY - rect.top) / rect.height) * height)),
      );
      const last = lastPickRef.current;
      if (last && last.col === col && last.row === row) return;
      lastPickRef.current = { col, row };
      onPick(col, row);
    },
    [onPick, width, height],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPick || e.button !== 0) return;
      e.preventDefault();
      isPickingRef.current = true;
      lastPickRef.current = null;
      pickFromMouse(e);
    },
    [onPick, pickFromMouse],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isPickingRef.current) return;
      e.preventDefault();
      pickFromMouse(e);
    },
    [pickFromMouse],
  );

  const stopMousePicking = useCallback(() => {
    if (!isPickingRef.current) return;
    isPickingRef.current = false;
    lastPickRef.current = null;
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, fontFamily: "sans-serif" }}>{title}</span>
        {onAdjust ? (
          <button
            type="button"
            onClick={onAdjust}
            title="Adjust display settings"
            style={{
              fontSize: 11, lineHeight: 1, padding: "2px 6px", cursor: "pointer",
              border: "1px solid #bbb", borderRadius: 4, background: "#f5f5f5",
            }}
          >
            ⚙ Adjust
          </button>
        ) : null}
      </div>
      <canvas
        ref={canvasRef}
        onPointerDown={onPick ? handlePointerDown : undefined}
        onPointerMove={onPick ? handlePointerMove : undefined}
        onPointerUp={onPick ? stopPicking : undefined}
        onPointerCancel={onPick ? stopPicking : undefined}
        onMouseDown={onPick ? handleMouseDown : undefined}
        onMouseMove={onPick ? handleMouseMove : undefined}
        onMouseUp={onPick ? stopMousePicking : undefined}
        onMouseLeave={onPick ? stopMousePicking : undefined}
        style={{
          cursor: onPick ? "crosshair" : "default",
          imageRendering: "pixelated",
          border: "1px solid #ccc",
          touchAction: "none",
          userSelect: "none",
        }}
      />
    </div>
  );
}

// Zoomed NxN neighborhood of the map around the selected pixel, drawn as large
// cells so individual scan positions are visible. The center (selected) cell gets
// a green border. Out-of-bounds neighbors (near map edges) stay background-filled.
interface InsetPanelProps {
  data: Float32Array;
  width: number;
  height: number;
  cmap: string;
  vmin: number;
  vmax: number;
  isRgb?: boolean;
  centerCol: number;
  centerRow: number;
  size: number;     // NxN window (odd)
  cellPx?: number;  // displayed px per cell
  title?: string;
}

function InsetPanel(props: InsetPanelProps) {
  const {
    data, width, height, cmap, vmin, vmax, isRgb,
    centerCol, centerRow, size, cellPx = 18, title,
  } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const n = Math.max(1, size | 0);
  const half = Math.floor(n / 2);
  const dim = n * cellPx;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(dim * dpr);
    canvas.height = Math.round(dim * dpr);
    canvas.style.width = `${dim}px`;
    canvas.style.height = `${dim}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#222";  // out-of-bounds backdrop
    ctx.fillRect(0, 0, dim, dim);

    const lut = isRgb ? null : lutFor(cmap);
    const range = vmax > vmin ? vmax - vmin : 1;
    const uniform = !(vmax > vmin);
    const clamp255 = (x: number) => (x < 0 ? 0 : x > 255 ? 255 : x | 0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const col = centerCol - half + j;
        const row = centerRow - half + i;
        if (col < 0 || col >= width || row < 0 || row >= height) continue;
        const idx = row * width + col;
        let r: number, g: number, b: number;
        if (isRgb) {
          if (data.length < (idx + 1) * 3) continue;
          r = clamp255(data[idx * 3] * 255);
          g = clamp255(data[idx * 3 + 1] * 255);
          b = clamp255(data[idx * 3 + 2] * 255);
        } else if (lut) {
          if (data.length <= idx) continue;
          const v = uniform
            ? 128
            : Math.min(255, Math.max(0, Math.floor(((data[idx] - vmin) / range) * 255)));
          r = lut[v * 3]; g = lut[v * 3 + 1]; b = lut[v * 3 + 2];
        } else {
          r = 0; g = 0; b = 0;
        }
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(j * cellPx, i * cellPx, cellPx, cellPx);
      }
    }
    // faint cell grid
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    for (let k = 0; k <= n; k++) {
      ctx.beginPath(); ctx.moveTo(k * cellPx + 0.5, 0); ctx.lineTo(k * cellPx + 0.5, dim); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, k * cellPx + 0.5); ctx.lineTo(dim, k * cellPx + 0.5); ctx.stroke();
    }
    // green border on the selected (center) cell
    ctx.strokeStyle = "#00e000";
    ctx.lineWidth = 2;
    ctx.strokeRect(half * cellPx + 1, half * cellPx + 1, cellPx - 2, cellPx - 2);
  }, [data, width, height, cmap, vmin, vmax, isRgb, centerCol, centerRow, n, half, dim, cellPx]);

  return (
    <div>
      {title ? (
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, fontFamily: "sans-serif" }}>
          {title}
        </div>
      ) : null}
      <canvas ref={canvasRef} style={{ imageRendering: "pixelated", border: "1px solid #ccc" }} />
    </div>
  );
}

// Colormap choices for the per-panel editor dropdown.
const CMAP_OPTIONS = Object.keys(COLORMAPS);

// One labeled slider + number input row. `nullable` adds an "auto" button that clears
// the value (null = use auto / no transform).
function EditorRow(props: {
  label: string;
  value: number | null;
  min: number;
  max: number;
  step: number;
  onChange: (v: number | null) => void;
  nullable?: boolean;
}) {
  const { label, value, min, max, step, onChange, nullable } = props;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0", fontSize: 12 }}>
      <span style={{ width: 110 }}>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value == null ? min : value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1 }}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value == null ? "" : value}
        placeholder={nullable ? "auto" : ""}
        onChange={(e) => {
          const s = e.target.value;
          if (s === "") { onChange(nullable ? null : min); return; }
          const v = parseFloat(s);
          if (Number.isFinite(v)) onChange(v);
        }}
        style={{ width: 64 }}
      />
      {nullable ? (
        <button
          type="button"
          onClick={() => onChange(null)}
          title="Auto / none"
          style={{ fontSize: 10, padding: "1px 5px", cursor: "pointer" }}
        >
          auto
        </button>
      ) : null}
    </div>
  );
}

interface PanelEditorProps {
  title: string;
  isPolar: boolean;
  preview: React.ReactNode;
  cmap: string;
  vmin: number | null;
  vmax: number | null;
  dataVmin: number;
  dataVmax: number;
  sigma: number | null;
  power: number | null;
  upperQuantile: number | null;
  zoom: number | null;
  onCmap: (v: string) => void;
  onVmin: (v: number | null) => void;
  onVmax: (v: number | null) => void;
  onSigma: (v: number | null) => void;
  onPower: (v: number | null) => void;
  onQuantile: (v: number | null) => void;
  onZoom: (v: number | null) => void;
  onSave?: () => void;
  onClose: () => void;
}

// Modal overlay: large preview of one panel + its display controls. Display-only
// controls (colormap / vmin / vmax) re-render instantly; the transform controls
// (sigma / power / quantile / zoom) recompute the panel in the kernel.
function PanelEditor(props: PanelEditorProps) {
  const {
    title, isPolar, preview, cmap, vmin, vmax, dataVmin, dataVmax,
    sigma, power, upperQuantile, zoom,
    onCmap, onVmin, onVmax, onSigma, onPower, onQuantile, onZoom, onSave, onClose,
  } = props;
  const vspan = Math.max(1e-6, dataVmax - dataVmin || 1);
  const vlo = dataVmin - vspan;
  const vhi = dataVmax + vspan;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)",
        display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#fff", borderRadius: 8, padding: 16, maxWidth: "92vw", maxHeight: "92vh",
          overflow: "auto", boxShadow: "0 8px 40px rgba(0,0,0,0.3)", display: "flex", gap: 16,
          fontFamily: "sans-serif", alignItems: "flex-start",
        }}
      >
        <div>{preview}</div>
        <div style={{ minWidth: 300 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontWeight: 700, fontSize: 14 }}>Adjust: {title}</span>
            <div style={{ display: "flex", gap: 6 }}>
              {onSave ? (
                <button type="button" onClick={onSave} title="Save this panel to disk"
                  style={{ cursor: "pointer" }}>Save this panel</button>
              ) : null}
              <button type="button" onClick={onClose} style={{ cursor: "pointer" }}>Done</button>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0", fontSize: 12 }}>
            <span style={{ width: 110 }}>Colormap</span>
            <select value={cmap} onChange={(e) => onCmap(e.target.value)} style={{ flex: 1 }}>
              {CMAP_OPTIONS.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <EditorRow label="vmin" value={vmin} min={vlo} max={vhi} step={vspan / 200} onChange={onVmin} nullable />
          <EditorRow label="vmax" value={vmax} min={vlo} max={vhi} step={vspan / 200} onChange={onVmax} nullable />
          {!isPolar ? (
            <>
              <EditorRow label="sigma blur" value={sigma} min={0} max={8} step={0.05} onChange={onSigma} nullable />
              <EditorRow label="power law" value={power} min={0.1} max={3} step={0.05} onChange={onPower} />
              <EditorRow label="upper quantile" value={upperQuantile} min={0.9} max={1} step={0.0001} onChange={onQuantile} nullable />
              <EditorRow label="zoom" value={zoom} min={1} max={8} step={0.25} onChange={onZoom} />
              <div style={{ fontSize: 10, color: "#888", marginTop: 6 }}>
                sigma / power / quantile / zoom recompute the panel in the kernel; colormap / vmin / vmax are instant.
              </div>
            </>
          ) : (
            <div style={{ fontSize: 10, color: "#888", marginTop: 6 }}>
              Polar panel: display-only controls.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ShowPolymer4DSTEM() {
  const model = useModel();
  const [scanHeight] = useModelState<number>("scan_height");
  const [scanWidth] = useModelState<number>("scan_width");
  const [upsample] = useModelState<number>("upsample_factor");
  const [title] = useModelState<string>("title");

  const [mapBytes] = useModelState<unknown>("map_bytes");
  const [mapHeight] = useModelState<number>("map_height");
  const [mapWidth] = useModelState<number>("map_width");
  const [mapVmin] = useModelState<number>("map_vmin");
  const [mapVmax] = useModelState<number>("map_vmax");
  const [mapCmap] = useModelState<string>("map_cmap");
  const [mapIsRgb] = useModelState<boolean>("map_is_rgb");
  const [mapTitle] = useModelState<string>("map_title");
  const [insetSize] = useModelState<number>("inset_size");
  const [showInset, setShowInset] = useModelState<boolean>("show_inset");

  const [dpCmap] = useModelState<string>("dp_cmap");
  const [dpVmin] = useModelState<number | null>("dp_vmin");
  const [dpVmax] = useModelState<number | null>("dp_vmax");

  const [hasPeaks] = useModelState<boolean>("has_peaks");
  const [hasPolar] = useModelState<boolean>("has_polar");
  const [showPeaks, setShowPeaks] = useModelState<boolean>("show_peaks");
  const [showPolar, setShowPolar] = useModelState<boolean>("show_polar");

  const [liveInference] = useModelState<boolean>("live_inference");
  // Read-only here; the slider writes via model.set (through the throttle), not this setter,
  // so slider drags don't each trigger their own save_changes.
  const [thresholdPeak] = useModelState<number>("threshold_peak");

  const [peakColor] = useModelState<string>("peak_color");
  const [centralColor] = useModelState<string>("central_color");
  const [peakSizeMin] = useModelState<number>("peak_size_min");
  const [peakSizeMax] = useModelState<number>("peak_size_max");

  const [posRy] = useModelState<number>("pos_ry");
  const [posRx] = useModelState<number>("pos_rx");

  const [dpBytes] = useModelState<unknown>("dp_bytes");
  const [dpHeight] = useModelState<number>("dp_height");
  const [dpWidth] = useModelState<number>("dp_width");
  const [dpDataVmin] = useModelState<number>("dp_data_vmin");
  const [dpDataVmax] = useModelState<number>("dp_data_vmax");
  const [payloadSeq] = useModelState<number>("payload_seq");

  const [peaksX] = useModelState<number[]>("peaks_x");
  const [peaksY] = useModelState<number[]>("peaks_y");
  const [peaksIntensity] = useModelState<number[]>("peaks_intensity");
  const [centralIdx] = useModelState<number>("central_idx");
  const [centerY] = useModelState<number>("center_y");
  const [centerX] = useModelState<number>("center_x");

  const [polarBytes] = useModelState<unknown>("polar_bytes");
  const [polarHeight] = useModelState<number>("polar_height");
  const [polarWidth] = useModelState<number>("polar_width");
  const [polarVmin] = useModelState<number>("polar_vmin");
  const [polarVmax] = useModelState<number>("polar_vmax");
  const [polarPeaksR] = useModelState<number[]>("polar_peaks_r_bin");
  const [polarPeaksTheta] = useModelState<number[]>("polar_peaks_theta_bin");

  const [dpViewTitles] = useModelState<string[]>("dp_view_titles");
  // Display-only params (cmap/vmin/vmax): setters re-render locally + persist to
  // Python (which ignores them — no recompute).
  const [dpViewCmaps, setDpViewCmaps] = useModelState<string[]>("dp_view_cmaps");
  const [dpViewColors] = useModelState<string[]>("dp_view_colors");
  const [dpViewCentralColors] = useModelState<string[]>("dp_view_central_colors");
  const [dpViewVmins, setDpViewVmins] = useModelState<Array<number | null>>("dp_view_vmins");
  const [dpViewVmaxs, setDpViewVmaxs] = useModelState<Array<number | null>>("dp_view_vmaxs");
  // Data-transform params (sigma/power/upper-quantile/zoom): editing these must trigger
  // a Python recompute (routed through the throttle), so write via model.set, not here.
  const [dpViewSigmas] = useModelState<Array<number | null>>("dp_view_sigmas");
  const [dpViewPowers] = useModelState<Array<number | null>>("dp_view_powers");
  const [dpViewUpperQuantiles] = useModelState<Array<number | null>>("dp_view_upper_quantiles");
  const [dpViewZooms] = useModelState<Array<number | null>>("dp_view_zooms");
  // Polar display-only overrides.
  const [polarCmap, setPolarCmap] = useModelState<string>("polar_cmap");
  const [polarDisplayVmin, setPolarDisplayVmin] = useModelState<number | null>("polar_display_vmin");
  const [polarDisplayVmax, setPolarDisplayVmax] = useModelState<number | null>("polar_display_vmax");

  // Save-to-disk config.
  const [saveDirBase] = useModelState<string>("save_dir_base");
  const [saveSubfolder, setSaveSubfolder] = useModelState<string>("save_subfolder");
  const [saveIncludeMap, setSaveIncludeMap] = useModelState<boolean>("save_include_map");
  const [saveIncludePolar, setSaveIncludePolar] = useModelState<boolean>("save_include_polar");
  const [savePanels] = useModelState<string[]>("save_panels");
  const [saveStatus] = useModelState<string>("save_status");

  const [dpCurrentBytes] = useModelState<unknown>("dp_current_bytes");
  const [dpCurrentHeight] = useModelState<number>("dp_current_height");
  const [dpCurrentWidth] = useModelState<number>("dp_current_width");
  const [dpCurrentDataVmin] = useModelState<number>("dp_current_data_vmin");
  const [dpCurrentDataVmax] = useModelState<number>("dp_current_data_vmax");
  const [dpCurrentPeaksX] = useModelState<number[]>("dp_current_peaks_x");
  const [dpCurrentPeaksY] = useModelState<number[]>("dp_current_peaks_y");
  const [dpCurrentPeaksIntensity] = useModelState<number[]>("dp_current_peaks_intensity");
  const [dpCurrentCentralIdx] = useModelState<number>("dp_current_central_idx");
  const [dpCurrentCenterY] = useModelState<number>("dp_current_center_y");
  const [dpCurrentCenterX] = useModelState<number>("dp_current_center_x");

  const [dpLamellarBytes] = useModelState<unknown>("dp_lamellar_bytes");
  const [dpLamellarHeight] = useModelState<number>("dp_lamellar_height");
  const [dpLamellarWidth] = useModelState<number>("dp_lamellar_width");
  const [dpLamellarDataVmin] = useModelState<number>("dp_lamellar_data_vmin");
  const [dpLamellarDataVmax] = useModelState<number>("dp_lamellar_data_vmax");
  const [dpLamellarPeaksX] = useModelState<number[]>("dp_lamellar_peaks_x");
  const [dpLamellarPeaksY] = useModelState<number[]>("dp_lamellar_peaks_y");
  const [dpLamellarPeaksIntensity] = useModelState<number[]>("dp_lamellar_peaks_intensity");
  const [dpLamellarCentralIdx] = useModelState<number>("dp_lamellar_central_idx");
  const [dpLamellarCenterY] = useModelState<number>("dp_lamellar_center_y");
  const [dpLamellarCenterX] = useModelState<number>("dp_lamellar_center_x");

  const [dpBackboneBytes] = useModelState<unknown>("dp_backbone_bytes");
  const [dpBackboneHeight] = useModelState<number>("dp_backbone_height");
  const [dpBackboneWidth] = useModelState<number>("dp_backbone_width");
  const [dpBackboneDataVmin] = useModelState<number>("dp_backbone_data_vmin");
  const [dpBackboneDataVmax] = useModelState<number>("dp_backbone_data_vmax");
  const [dpBackbonePeaksX] = useModelState<number[]>("dp_backbone_peaks_x");
  const [dpBackbonePeaksY] = useModelState<number[]>("dp_backbone_peaks_y");
  const [dpBackbonePeaksIntensity] = useModelState<number[]>("dp_backbone_peaks_intensity");
  const [dpBackboneCentralIdx] = useModelState<number>("dp_backbone_central_idx");
  const [dpBackboneCenterY] = useModelState<number>("dp_backbone_center_y");
  const [dpBackboneCenterX] = useModelState<number>("dp_backbone_center_x");

  const [dpPipiBytes] = useModelState<unknown>("dp_pipi_bytes");
  const [dpPipiHeight] = useModelState<number>("dp_pipi_height");
  const [dpPipiWidth] = useModelState<number>("dp_pipi_width");
  const [dpPipiDataVmin] = useModelState<number>("dp_pipi_data_vmin");
  const [dpPipiDataVmax] = useModelState<number>("dp_pipi_data_vmax");
  const [dpPipiPeaksX] = useModelState<number[]>("dp_pipi_peaks_x");
  const [dpPipiPeaksY] = useModelState<number[]>("dp_pipi_peaks_y");
  const [dpPipiPeaksIntensity] = useModelState<number[]>("dp_pipi_peaks_intensity");
  const [dpPipiCentralIdx] = useModelState<number>("dp_pipi_central_idx");
  const [dpPipiCenterY] = useModelState<number>("dp_pipi_center_y");
  const [dpPipiCenterX] = useModelState<number>("dp_pipi_center_x");

  const mapData = useMemo(() => asFloat32(mapBytes), [mapBytes]);
  const dpData = useMemo(() => asFloat32(dpBytes), [dpBytes, payloadSeq]);
  const polarData = useMemo(() => asFloat32(polarBytes), [polarBytes, payloadSeq]);
  const dpCurrentData = useMemo(() => asFloat32(dpCurrentBytes), [dpCurrentBytes, payloadSeq]);
  const dpLamellarData = useMemo(() => asFloat32(dpLamellarBytes), [dpLamellarBytes, payloadSeq]);
  const dpBackboneData = useMemo(() => asFloat32(dpBackboneBytes), [dpBackboneBytes, payloadSeq]);
  const dpPipiData = useMemo(() => asFloat32(dpPipiBytes), [dpPipiBytes, payloadSeq]);

  // Cursor on the (possibly upsampled) map, in map pixel coords.
  const mapCursor = useMemo(
    () => ({ col: posRx * upsample + Math.floor(upsample / 2), row: posRy * upsample + Math.floor(upsample / 2) }),
    [posRx, posRy, upsample],
  );

  // Drag throttle: the crosshair + inset are driven by pos_ry/pos_rx local model
  // state and update on every move (no kernel involvement), but each
  // `save_changes()` triggers a full kernel recompute of every DP panel. Firing
  // one per pixel during a drag floods the single-threaded kernel and builds a
  // backlog (the ~0.5 s DP lag). Instead we keep at most one request in flight,
  // coalescing intermediate positions and always sending the final one.
  // Coalesce recomputes by a *signature* of every kernel-recompute-affecting value
  // (position + threshold + — in Phase 2 — the per-panel transform params). Callers
  // set the relevant traits on the model, then call requestRecompute(sig); at most one
  // save_changes() is in flight, and the trailing state is always flushed afterward.
  const pendingSigRef = useRef<string | null>(null);
  const lastSentSigRef = useRef<string | null>(null);
  const inFlightRef = useRef(false);
  const watchdogRef = useRef<number | null>(null);
  const firstSeqRef = useRef(true);

  const flushPending = useCallback(() => {
    const sig = pendingSigRef.current;
    if (sig == null || lastSentSigRef.current === sig) return;
    inFlightRef.current = true;
    lastSentSigRef.current = sig;
    if (watchdogRef.current != null) clearTimeout(watchdogRef.current);
    // Safety net: if a recompute response is ever lost, don't stall forever.
    watchdogRef.current = window.setTimeout(() => {
      watchdogRef.current = null;
      inFlightRef.current = false;
      flushPending();
    }, 3000);
    model.save_changes(); // flushes the traits already set on the model
  }, [model]);

  // Signature over every kernel-recompute-affecting trait, read straight off the model
  // (model.set updates it synchronously, so callers just set traits then call this).
  const recomputeSig = useCallback(() => {
    const g = (n: string) => JSON.stringify(model.get(n));
    return [
      g("pos_ry"), g("pos_rx"), g("threshold_peak"),
      g("dp_view_sigmas"), g("dp_view_powers"),
      g("dp_view_upper_quantiles"), g("dp_view_zooms"),
    ].join("|");
  }, [model]);

  const requestRecompute = useCallback(() => {
    pendingSigRef.current = recomputeSig();
    // Only kick a kernel recompute if none is in flight; otherwise the trailing
    // flush (on the next payload response) will pick up this latest state.
    if (!inFlightRef.current) flushPending();
  }, [recomputeSig, flushPending]);

  // Single entry point for every position change (map click/drag, arrow keys,
  // typed X/Y): clamp to the scan grid, update the model locally so the crosshair
  // + inset track immediately, then coalesce kernel recomputes.
  const commitPosition = useCallback(
    (ry: number, rx: number) => {
      const nextRy = Math.max(0, Math.min(scanHeight - 1, Math.round(ry)));
      const nextRx = Math.max(0, Math.min(scanWidth - 1, Math.round(rx)));
      if (nextRy === posRy && nextRx === posRx) return;
      model.set("pos_ry", nextRy);
      model.set("pos_rx", nextRx);
      requestRecompute();
    },
    [scanHeight, scanWidth, posRy, posRx, model, requestRecompute],
  );

  // Threshold slider (live-inference mode): update the model locally so the label
  // tracks instantly, and coalesce the kernel re-detection through the same throttle.
  const commitThreshold = useCallback(
    (v: number) => {
      model.set("threshold_peak", v);
      requestRecompute();
    },
    [model, requestRecompute],
  );

  // Set one entry of a per-panel transform list and trigger a coalesced recompute.
  const commitPanelTransform = useCallback(
    (traitName: string, current: Array<number | null> | undefined, idx: number, v: number | null) => {
      const next = [...(current ?? [])];
      next[idx] = v;
      model.set(traitName, next);
      requestRecompute();
    },
    [model, requestRecompute],
  );

  // Set one entry of a per-panel display-only list (cmap/vmin/vmax): re-renders locally
  // and persists to Python without a recompute.
  const setPanelDisplay = useCallback(
    <T,>(setter: (v: T[]) => void, current: T[] | undefined, idx: number, v: T) => {
      const next = [...(current ?? [])];
      next[idx] = v;
      setter(next);
    },
    [],
  );

  // Trigger a save on the kernel: set the selection traits, then bump save_request.
  const fireSave = useCallback(
    (panels: string[], includeMap: boolean, includePolar: boolean) => {
      model.set("save_panels", panels);
      model.set("save_include_map", includeMap);
      model.set("save_include_polar", includePolar);
      model.set("save_status", "Saving…");
      model.set("save_request", ((model.get("save_request") as number) ?? 0) + 1);
      model.save_changes();
    },
    [model],
  );

  // Root container is focusable so it can receive arrow-key events; clicking the
  // map focuses it so keyboard nav works without an extra tab/click.
  const containerRef = useRef<HTMLDivElement | null>(null);

  const handleMapPick = useCallback(
    (col: number, row: number) => {
      containerRef.current?.focus();
      commitPosition(
        Math.floor(row / Math.max(1, upsample)),
        Math.floor(col / Math.max(1, upsample)),
      );
    },
    [upsample, commitPosition],
  );

  // Arrow keys nudge the selected position by one scan pixel (10 with Shift).
  // Ignored while a text field is focused so typing in the X/Y boxes still works.
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      let dRy = 0;
      let dRx = 0;
      switch (e.key) {
        case "ArrowUp": dRy = -1; break;
        case "ArrowDown": dRy = 1; break;
        case "ArrowLeft": dRx = -1; break;
        case "ArrowRight": dRx = 1; break;
        default: return;
      }
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      commitPosition(posRy + dRy * step, posRx + dRx * step);
    },
    [posRy, posRx, commitPosition],
  );

  // Editable X (Rx) / Y (Ry) text fields. Local string state lets the user type
  // freely; we re-sync from the model whenever the committed position changes
  // (e.g. after a click, drag, or arrow-key move).
  // Which panel's editor modal is open ("current"/"lamellar"/"backbone"/"pipi"/"polar"), or null.
  const [editingPanel, setEditingPanel] = useState<string | null>(null);
  const [showSaveModal, setShowSaveModal] = useState<boolean>(false);

  const [ryInput, setRyInput] = useState<string>(String(posRy));
  const [rxInput, setRxInput] = useState<string>(String(posRx));
  useEffect(() => { setRyInput(String(posRy)); }, [posRy]);
  useEffect(() => { setRxInput(String(posRx)); }, [posRx]);

  const commitInputs = useCallback(() => {
    const ry = parseInt(ryInput, 10);
    const rx = parseInt(rxInput, 10);
    commitPosition(
      Number.isFinite(ry) ? ry : posRy,
      Number.isFinite(rx) ? rx : posRx,
    );
  }, [ryInput, rxInput, posRy, posRx, commitPosition]);

  // A new payload (payload_seq bump) means the in-flight recompute finished;
  // clear the gate and send whatever position the drag has since landed on.
  useEffect(() => {
    if (firstSeqRef.current) {
      firstSeqRef.current = false;
      return;
    }
    inFlightRef.current = false;
    if (watchdogRef.current != null) {
      clearTimeout(watchdogRef.current);
      watchdogRef.current = null;
    }
    flushPending();
  }, [payloadSeq, flushPending]);

  useEffect(
    () => () => {
      if (watchdogRef.current != null) clearTimeout(watchdogRef.current);
    },
    [],
  );

  // Polar peaks overlay (r_bin -> x, theta_bin -> y).
  const polarOverlay = useCallback(
    (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => {
      if (!showPeaks || !polarPeaksR || polarPeaksR.length === 0) return;
      for (let i = 0; i < polarPeaksR.length; i++) {
        const x = (polarPeaksR[i] + 0.5) * scaleX;
        const y = (polarPeaksTheta[i] + 0.5) * scaleY;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = peakColor;
        ctx.stroke();
      }
    },
    [showPeaks, polarPeaksR, polarPeaksTheta, peakColor],
  );

  const dpUseVmin = dpVmin == null ? dpDataVmin : dpVmin;
  const dpUseVmax = dpVmax == null ? dpDataVmax : dpVmax;

  const dpViews: DpView[] = useMemo(
    () => [
      {
        key: "current",
        title: dpViewTitles?.[0] ?? "Current",
        data: dpCurrentData.length ? dpCurrentData : dpData,
        width: dpCurrentWidth || dpWidth,
        height: dpCurrentHeight || dpHeight,
        cmap: dpViewCmaps?.[0] ?? dpCmap,
        vmin: dpViewVmins?.[0] ?? dpUseVmin,
        vmax: dpViewVmaxs?.[0] ?? dpUseVmax,
        dataVmin: dpCurrentDataVmin || dpDataVmin,
        dataVmax: dpCurrentDataVmax || dpDataVmax,
        peaksX: dpCurrentPeaksX?.length ? dpCurrentPeaksX : peaksX,
        peaksY: dpCurrentPeaksY?.length ? dpCurrentPeaksY : peaksY,
        peaksIntensity: dpCurrentPeaksIntensity?.length ? dpCurrentPeaksIntensity : peaksIntensity,
        centralIdx: dpCurrentCentralIdx ?? centralIdx,
        centerY: dpCurrentCenterY || centerY,
        centerX: dpCurrentCenterX || centerX,
        peakColor: dpViewColors?.[0] ?? peakColor,
        centralColor: dpViewCentralColors?.[0] ?? centralColor,
      },
      {
        key: "lamellar",
        title: dpViewTitles?.[1] ?? "Lamellar",
        data: dpLamellarData,
        width: dpLamellarWidth,
        height: dpLamellarHeight,
        cmap: dpViewCmaps?.[1] ?? "inferno",
        vmin: dpViewVmins?.[1] ?? null,
        vmax: dpViewVmaxs?.[1] ?? null,
        dataVmin: dpLamellarDataVmin,
        dataVmax: dpLamellarDataVmax,
        peaksX: dpLamellarPeaksX ?? [],
        peaksY: dpLamellarPeaksY ?? [],
        peaksIntensity: dpLamellarPeaksIntensity ?? [],
        centralIdx: dpLamellarCentralIdx,
        centerY: dpLamellarCenterY,
        centerX: dpLamellarCenterX,
        peakColor: dpViewColors?.[1] ?? "#ff1f1f",
        centralColor: dpViewCentralColors?.[1] ?? "#00d5e8",
      },
      {
        key: "backbone",
        title: dpViewTitles?.[2] ?? "Backbone",
        data: dpBackboneData,
        width: dpBackboneWidth,
        height: dpBackboneHeight,
        cmap: dpViewCmaps?.[2] ?? "gray",
        vmin: dpViewVmins?.[2] ?? null,
        vmax: dpViewVmaxs?.[2] ?? null,
        dataVmin: dpBackboneDataVmin,
        dataVmax: dpBackboneDataVmax,
        peaksX: dpBackbonePeaksX ?? [],
        peaksY: dpBackbonePeaksY ?? [],
        peaksIntensity: dpBackbonePeaksIntensity ?? [],
        centralIdx: dpBackboneCentralIdx,
        centerY: dpBackboneCenterY,
        centerX: dpBackboneCenterX,
        peakColor: dpViewColors?.[2] ?? "#7bdc3c",
        centralColor: dpViewCentralColors?.[2] ?? "#00d5e8",
      },
      {
        key: "pipi",
        title: dpViewTitles?.[3] ?? "pi-pi",
        data: dpPipiData,
        width: dpPipiWidth,
        height: dpPipiHeight,
        cmap: dpViewCmaps?.[3] ?? "gray",
        vmin: dpViewVmins?.[3] ?? null,
        vmax: dpViewVmaxs?.[3] ?? null,
        dataVmin: dpPipiDataVmin,
        dataVmax: dpPipiDataVmax,
        peaksX: dpPipiPeaksX ?? [],
        peaksY: dpPipiPeaksY ?? [],
        peaksIntensity: dpPipiPeaksIntensity ?? [],
        centralIdx: dpPipiCentralIdx,
        centerY: dpPipiCenterY,
        centerX: dpPipiCenterX,
        peakColor: dpViewColors?.[3] ?? "#ffee00",
        centralColor: dpViewCentralColors?.[3] ?? "#00d5e8",
      },
    ],
    [
      dpViewTitles, dpViewCmaps, dpViewVmins, dpViewVmaxs, dpViewColors, dpViewCentralColors,
      dpCurrentData, dpCurrentWidth, dpCurrentHeight, dpCurrentDataVmin, dpCurrentDataVmax,
      dpCurrentPeaksX, dpCurrentPeaksY, dpCurrentPeaksIntensity, dpCurrentCentralIdx,
      dpCurrentCenterY, dpCurrentCenterX, dpData, dpWidth, dpHeight, dpCmap, dpUseVmin,
      dpUseVmax, dpDataVmin, dpDataVmax, peaksX, peaksY, peaksIntensity, centralIdx,
      centerY, centerX, peakColor, centralColor, dpLamellarData, dpLamellarWidth,
      dpLamellarHeight, dpLamellarDataVmin, dpLamellarDataVmax, dpLamellarPeaksX,
      dpLamellarPeaksY, dpLamellarPeaksIntensity, dpLamellarCentralIdx, dpLamellarCenterY,
      dpLamellarCenterX, dpBackboneData, dpBackboneWidth, dpBackboneHeight,
      dpBackboneDataVmin, dpBackboneDataVmax, dpBackbonePeaksX, dpBackbonePeaksY,
      dpBackbonePeaksIntensity, dpBackboneCentralIdx, dpBackboneCenterY, dpBackboneCenterX,
      dpPipiData, dpPipiWidth, dpPipiHeight, dpPipiDataVmin, dpPipiDataVmax, dpPipiPeaksX,
      dpPipiPeaksY, dpPipiPeaksIntensity, dpPipiCentralIdx, dpPipiCenterY, dpPipiCenterX,
    ],
  );

  const makeDpOverlay = useCallback(
    (view: DpView) => (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => {
      const drawDot = (x: number, y: number, r: number, fill: string) => {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        ctx.beginPath();
        ctx.arc((x + 0.5) * scaleX, (y + 0.5) * scaleY, r, 0, 2 * Math.PI);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#000";
        ctx.stroke();
      };
      if (!showPeaks || !view.peaksX || view.peaksX.length === 0) {
        drawDot(view.centerX, view.centerY, 5, view.centralColor);
        return;
      }
      let imin = Infinity;
      let imax = -Infinity;
      for (let i = 0; i < view.peaksIntensity.length; i++) {
        if (i === view.centralIdx) continue;
        imin = Math.min(imin, view.peaksIntensity[i]);
        imax = Math.max(imax, view.peaksIntensity[i]);
      }
      const range = imax > imin ? imax - imin : 1;
      for (let i = 0; i < view.peaksX.length; i++) {
        const x = view.peaksX[i];
        const y = view.peaksY[i];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        if (i === view.centralIdx) {
          drawDot(x, y, 5, view.centralColor);
          continue;
        }
        const norm = view.peaksIntensity.length ? (view.peaksIntensity[i] - imin) / range : 0.5;
        const r = peakSizeMin + (Number.isFinite(norm) ? norm : 0.5) * (peakSizeMax - peakSizeMin);
        ctx.beginPath();
        ctx.arc((x + 0.5) * scaleX, (y + 0.5) * scaleY, r, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = view.peakColor;
        ctx.stroke();
      }
    },
    [showPeaks, peakSizeMin, peakSizeMax],
  );

  const checkbox = (label: string, checked: boolean, onChange: (v: boolean) => void) => (
    <label style={{ fontSize: 12, fontFamily: "sans-serif", display: "flex", alignItems: "center", gap: 4 }}>
      <input type="checkbox" checked={!!checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );

  const posField = (
    label: string,
    value: string,
    setValue: (v: string) => void,
    max: number,
  ) => (
    <label style={{ fontSize: 12, fontFamily: "sans-serif", display: "flex", alignItems: "center", gap: 4 }}>
      {label}
      <input
        type="number"
        min={0}
        max={Math.max(0, max)}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commitInputs();
            (e.target as HTMLInputElement).blur();
          }
        }}
        onBlur={commitInputs}
        style={{ width: 56, fontSize: 12 }}
      />
    </label>
  );

  // Effective polar display params (overrides win; empty/null => auto).
  const effPolarCmap = polarCmap && polarCmap.length ? polarCmap : dpCmap;
  const effPolarVmin = polarDisplayVmin == null ? polarVmin : polarDisplayVmin;
  const effPolarVmax = polarDisplayVmax == null ? polarVmax : polarDisplayVmax;

  // Build the per-panel editor modal for the panel currently being edited.
  let panelEditor: React.ReactNode = null;
  if (editingPanel === "polar") {
    panelEditor = (
      <PanelEditor
        title="Polar"
        isPolar
        preview={
          <ImagePanel
            data={polarData}
            width={polarWidth}
            height={polarHeight}
            cmap={effPolarCmap}
            vmin={effPolarVmin}
            vmax={effPolarVmax}
            displayWidth={480}
            title="Polar (preview)"
            overlay={polarOverlay}
            aspectAuto
          />
        }
        cmap={effPolarCmap}
        vmin={polarDisplayVmin}
        vmax={polarDisplayVmax}
        dataVmin={polarVmin}
        dataVmax={polarVmax}
        sigma={null}
        power={null}
        upperQuantile={null}
        zoom={null}
        onCmap={(v) => setPolarCmap(v)}
        onVmin={(v) => setPolarDisplayVmin(v)}
        onVmax={(v) => setPolarDisplayVmax(v)}
        onSigma={() => {}}
        onPower={() => {}}
        onQuantile={() => {}}
        onZoom={() => {}}
        onSave={() => fireSave([], false, true)}
        onClose={() => setEditingPanel(null)}
      />
    );
  } else if (editingPanel != null) {
    const i = dpViews.findIndex((v) => v.key === editingPanel);
    if (i >= 0) {
      const view = dpViews[i];
      panelEditor = (
        <PanelEditor
          title={view.title}
          isPolar={false}
          preview={
            <ImagePanel
              data={view.data}
              width={view.width}
              height={view.height}
              cmap={view.cmap}
              vmin={view.vmin == null ? view.dataVmin : view.vmin}
              vmax={view.vmax == null ? view.dataVmax : view.vmax}
              displayWidth={480}
              title={`${view.title} (preview)`}
              overlay={makeDpOverlay(view)}
            />
          }
          cmap={dpViewCmaps?.[i] ?? view.cmap}
          vmin={dpViewVmins?.[i] ?? null}
          vmax={dpViewVmaxs?.[i] ?? null}
          dataVmin={view.dataVmin}
          dataVmax={view.dataVmax}
          sigma={dpViewSigmas?.[i] ?? null}
          power={dpViewPowers?.[i] ?? null}
          upperQuantile={dpViewUpperQuantiles?.[i] ?? null}
          zoom={dpViewZooms?.[i] ?? null}
          onCmap={(v) => setPanelDisplay(setDpViewCmaps, dpViewCmaps, i, v)}
          onVmin={(v) => setPanelDisplay(setDpViewVmins, dpViewVmins, i, v)}
          onVmax={(v) => setPanelDisplay(setDpViewVmaxs, dpViewVmaxs, i, v)}
          onSigma={(v) => commitPanelTransform("dp_view_sigmas", dpViewSigmas, i, v)}
          onPower={(v) => commitPanelTransform("dp_view_powers", dpViewPowers, i, v)}
          onQuantile={(v) => commitPanelTransform("dp_view_upper_quantiles", dpViewUpperQuantiles, i, v)}
          onZoom={(v) => commitPanelTransform("dp_view_zooms", dpViewZooms, i, v)}
          onSave={() => fireSave([view.key], false, false)}
          onClose={() => setEditingPanel(null)}
        />
      );
    }
  }

  // Save modal: destination + which panels, then fire a save on the kernel.
  const pathPreview =
    `${saveDirBase || "widget_saves"}/` +
    (saveSubfolder ? `${saveSubfolder.replace(/^\/+|\/+$/g, "")}/` : "") +
    `ry${posRy}_rx${posRx}/`;
  const saveModal = showSaveModal ? (
    <div
      onClick={() => setShowSaveModal(false)}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)",
        display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#fff", borderRadius: 8, padding: 16, minWidth: 380, maxWidth: "92vw",
          maxHeight: "92vh", overflow: "auto", boxShadow: "0 8px 40px rgba(0,0,0,0.3)",
          fontFamily: "sans-serif", fontSize: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>Save figures — Ry={posRy}, Rx={posRx}</span>
          <button type="button" onClick={() => setShowSaveModal(false)} style={{ cursor: "pointer" }}>Close</button>
        </div>
        <div style={{ marginBottom: 6 }}>
          <span style={{ color: "#555" }}>Base:</span> <code>{saveDirBase || "widget_saves"}</code>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
          Subfolder
          <input
            type="text"
            value={saveSubfolder ?? ""}
            placeholder="e.g. dataset1/run5"
            onChange={(e) => setSaveSubfolder(e.target.value)}
            style={{ flex: 1, fontSize: 12 }}
          />
        </label>
        <div style={{ color: "#777", marginBottom: 10 }}>
          → <code>{pathPreview}</code>
        </div>
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Panels</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {dpViews.map((v) => {
              const checked = (savePanels ?? []).includes(v.key);
              return (
                <label key={v.key} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => {
                      const set = new Set(savePanels ?? []);
                      if (set.has(v.key)) set.delete(v.key); else set.add(v.key);
                      model.set("save_panels", dpViews.map((d) => d.key).filter((k) => set.has(k)));
                      model.save_changes();
                    }}
                  />
                  {v.title}
                </label>
              );
            })}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          {checkbox("Context map", !!saveIncludeMap, setSaveIncludeMap)}
          {hasPolar ? checkbox("Polar", !!saveIncludePolar, setSaveIncludePolar) : null}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            type="button"
            onClick={() => fireSave(savePanels ?? [], !!saveIncludeMap, !!saveIncludePolar && hasPolar)}
            style={{ cursor: "pointer", fontWeight: 600, padding: "4px 12px" }}
          >
            Save
          </button>
          <span style={{ color: "#555" }}>{saveStatus}</span>
        </div>
      </div>
    </div>
  ) : null;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      style={{ fontFamily: "sans-serif", outline: "none" }}
    >
      {title ? <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{title}</div> : null}
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        {posField(`Ry (Y, 0-${Math.max(0, scanHeight - 1)})`, ryInput, setRyInput, scanHeight - 1)}
        {posField(`Rx (X, 0-${Math.max(0, scanWidth - 1)})`, rxInput, setRxInput, scanWidth - 1)}
        <span style={{ fontSize: 11, color: "#888" }}>
          click map or use arrow keys (Shift = ×10)
        </span>
        {hasPeaks ? checkbox("Show peaks", showPeaks, setShowPeaks) : null}
        {hasPolar ? checkbox("Show polar", showPolar, setShowPolar) : null}
        {checkbox(`Show ${insetSize || 7}x${insetSize || 7} inset`, showInset, setShowInset)}
        <button
          type="button"
          onClick={() => setShowSaveModal(true)}
          title="Save this position's panels to disk"
          style={{
            fontSize: 12, padding: "2px 10px", cursor: "pointer",
            border: "1px solid #bbb", borderRadius: 4, background: "#f5f5f5",
          }}
        >
          Save…
        </button>
        {liveInference ? (
          <label style={{ fontSize: 12, fontFamily: "sans-serif", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ color: "#0a7", fontWeight: 600 }}>● live</span>
            Threshold: {(thresholdPeak ?? 0.5).toFixed(2)}
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.01}
              value={thresholdPeak ?? 0.5}
              onChange={(e) => commitThreshold(parseFloat(e.target.value))}
              style={{ width: 120 }}
            />
          </label>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* Map + its zoomed inset stacked vertically so the inset never overlaps the map. */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start" }}>
          <ImagePanel
            data={mapData}
            width={mapWidth}
            height={mapHeight}
            cmap={mapCmap}
            vmin={mapVmin}
            vmax={mapVmax}
            isRgb={mapIsRgb}
            displayWidth={260}
            title={mapTitle}
            onPick={handleMapPick}
            cursor={mapCursor}
            cursorColor="#ff3b30"
          />
          {showInset ? (
            <InsetPanel
              data={mapData}
              width={mapWidth}
              height={mapHeight}
              cmap={mapCmap}
              vmin={mapVmin}
              vmax={mapVmax}
              isRgb={mapIsRgb}
              centerCol={mapCursor.col}
              centerRow={mapCursor.row}
              size={insetSize || 7}
              title={`Inset (Ry=${posRy}, Rx=${posRx})`}
            />
          ) : null}
        </div>
        {dpViews.map((view) => (
          <ImagePanel
            key={view.key}
            data={view.data}
            width={view.width}
            height={view.height}
            cmap={view.cmap}
            vmin={view.vmin == null ? view.dataVmin : view.vmin}
            vmax={view.vmax == null ? view.dataVmax : view.vmax}
            displayWidth={view.key === "current" ? 300 : 220}
            title={`${view.title} DP (Ry=${posRy}, Rx=${posRx})`}
            overlay={makeDpOverlay(view)}
            onAdjust={() => setEditingPanel(view.key)}
          />
        ))}
        {hasPolar && showPolar ? (
          <ImagePanel
            data={polarData}
            width={polarWidth}
            height={polarHeight}
            cmap={effPolarCmap}
            vmin={effPolarVmin}
            vmax={effPolarVmax}
            displayWidth={320}
            title={`Polar (Ry=${posRy}, Rx=${posRx})`}
            overlay={polarOverlay}
            aspectAuto
            onAdjust={() => setEditingPanel("polar")}
          />
        ) : null}
      </div>
      {hasPeaks && showPeaks && (!peaksX || peaksX.length === 0) ? (
        <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>No peaks at this scan position.</div>
      ) : null}
      {panelEditor}
      {saveModal}
    </div>
  );
}

export const render = createRender(ShowPolymer4DSTEM);
