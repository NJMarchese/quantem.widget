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

const { useRef, useEffect, useMemo, useCallback } = React;

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
    overlay, onPick, cursor, cursorColor, aspectAuto,
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
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, fontFamily: "sans-serif" }}>{title}</div>
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
  const [dpViewCmaps] = useModelState<string[]>("dp_view_cmaps");
  const [dpViewColors] = useModelState<string[]>("dp_view_colors");
  const [dpViewCentralColors] = useModelState<string[]>("dp_view_central_colors");
  const [dpViewVmins] = useModelState<Array<number | null>>("dp_view_vmins");
  const [dpViewVmaxs] = useModelState<Array<number | null>>("dp_view_vmaxs");

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

  const handleMapPick = useCallback(
    (col: number, row: number) => {
      const ry = Math.floor(row / Math.max(1, upsample));
      const rx = Math.floor(col / Math.max(1, upsample));
      const nextRy = Math.max(0, Math.min(scanHeight - 1, ry));
      const nextRx = Math.max(0, Math.min(scanWidth - 1, rx));
      if (nextRy === posRy && nextRx === posRx) return;
      model.set("pos_ry", nextRy);
      model.set("pos_rx", nextRx);
      model.save_changes();
    },
    [upsample, scanHeight, scanWidth, posRy, posRx, model],
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

  return (
    <div style={{ fontFamily: "sans-serif" }}>
      {title ? <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{title}</div> : null}
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: "#555" }}>
          Scan position: Ry={posRy}, Rx={posRx}
        </span>
        {hasPeaks ? checkbox("Show peaks", showPeaks, setShowPeaks) : null}
        {hasPolar ? checkbox("Show polar", showPolar, setShowPolar) : null}
        {checkbox(`Show ${insetSize || 7}x${insetSize || 7} inset`, showInset, setShowInset)}
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
          />
        ))}
        {hasPolar && showPolar ? (
          <ImagePanel
            data={polarData}
            width={polarWidth}
            height={polarHeight}
            cmap={dpCmap}
            vmin={polarVmin}
            vmax={polarVmax}
            displayWidth={320}
            title={`Polar (Ry=${posRy}, Rx=${posRx})`}
            overlay={polarOverlay}
            aspectAuto
          />
        ) : null}
      </div>
      {hasPeaks && showPeaks && (!peaksX || peaksX.length === 0) ? (
        <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>No peaks at this scan position.</div>
      ) : null}
    </div>
  );
}

export const render = createRender(ShowPolymer4DSTEM);
