// show_polymer_4dstem: live-kernel Bragg-peak / polymer 4D-STEM viewer.
//
// Port of BraggPeaksPolymer.plot_interactive_peak_map / plot_interactive_image_map.
// Left panel: clickable real-space intensity map (a click selects a scan
// position). Middle panel: the diffraction pattern at that position, with the
// detected Bragg peaks overlaid (hollow markers sized by intensity, the central
// beam filled). Right panel (optional): the polar transform with its polar peaks.
//
// All heavy data is recomputed in the Python kernel on every click and shipped
// over the comm; this file only colormaps + draws.
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
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

interface ImagePanelProps {
  data: Float32Array;
  width: number;
  height: number;
  cmap: string;
  vmin: number;
  vmax: number;
  displayWidth: number;
  title: string;
  // Pixels-per-data-pixel marker overlay, drawn in NATIVE image coords.
  overlay?: (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => void;
  onPick?: (col: number, row: number) => void;
  cursor?: { col: number; row: number } | null;
  cursorColor?: string;
  aspectAuto?: boolean;
}

function ImagePanel(props: ImagePanelProps) {
  const {
    data, width, height, cmap, vmin, vmax, displayWidth, title,
    overlay, onPick, cursor, cursorColor, aspectAuto,
  } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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

    const off = colormapToCanvas(data, width, height, lutFor(cmap), vmin, vmax);
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
  }, [data, width, height, cmap, vmin, vmax, dispW, dispH, overlay, cursor, cursorColor]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPick) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const col = Math.floor(((e.clientX - rect.left) / rect.width) * width);
      const row = Math.floor(((e.clientY - rect.top) / rect.height) * height);
      onPick(
        Math.max(0, Math.min(width - 1, col)),
        Math.max(0, Math.min(height - 1, row)),
      );
    },
    [onPick, width, height],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, fontFamily: "sans-serif" }}>{title}</div>
      <canvas
        ref={canvasRef}
        onClick={onPick ? handleClick : undefined}
        style={{ cursor: onPick ? "crosshair" : "default", imageRendering: "pixelated", border: "1px solid #ccc" }}
      />
    </div>
  );
}

function ShowPolymer4DSTEM() {
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
  const [mapTitle] = useModelState<string>("map_title");

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

  const [posRy, setPosRy] = useModelState<number>("pos_ry");
  const [posRx, setPosRx] = useModelState<number>("pos_rx");

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

  const mapData = useMemo(() => asFloat32(mapBytes), [mapBytes]);
  const dpData = useMemo(() => asFloat32(dpBytes), [dpBytes, payloadSeq]);
  const polarData = useMemo(() => asFloat32(polarBytes), [polarBytes, payloadSeq]);

  // Cursor on the (possibly upsampled) map, in map pixel coords.
  const mapCursor = useMemo(
    () => ({ col: posRx * upsample + Math.floor(upsample / 2), row: posRy * upsample + Math.floor(upsample / 2) }),
    [posRx, posRy, upsample],
  );

  const handleMapPick = useCallback(
    (col: number, row: number) => {
      const ry = Math.floor(row / Math.max(1, upsample));
      const rx = Math.floor(col / Math.max(1, upsample));
      setPosRy(Math.max(0, Math.min(scanHeight - 1, ry)));
      setPosRx(Math.max(0, Math.min(scanWidth - 1, rx)));
    },
    [upsample, scanHeight, scanWidth, setPosRy, setPosRx],
  );

  // Peak overlay on the diffraction pattern (native DP coords -> display).
  const dpOverlay = useCallback(
    (ctx: CanvasRenderingContext2D, scaleX: number, scaleY: number) => {
      // Center / central beam marker (filled).
      const drawDot = (x: number, y: number, r: number, fill: string) => {
        ctx.beginPath();
        ctx.arc((x + 0.5) * scaleX, (y + 0.5) * scaleY, r, 0, 2 * Math.PI);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#000";
        ctx.stroke();
      };
      if (!showPeaks || !peaksX || peaksX.length === 0) {
        if (centerX || centerY) drawDot(centerX, centerY, 5, centralColor);
        return;
      }
      // Size non-central peaks by normalized intensity.
      let imin = Infinity;
      let imax = -Infinity;
      for (let i = 0; i < peaksIntensity.length; i++) {
        if (i === centralIdx) continue;
        imin = Math.min(imin, peaksIntensity[i]);
        imax = Math.max(imax, peaksIntensity[i]);
      }
      const range = imax > imin ? imax - imin : 1;
      for (let i = 0; i < peaksX.length; i++) {
        const px = (peaksX[i] + 0.5) * scaleX;
        const py = (peaksY[i] + 0.5) * scaleY;
        if (i === centralIdx) {
          drawDot(peaksX[i], peaksY[i], 5, centralColor);
          continue;
        }
        const norm = peaksIntensity.length ? (peaksIntensity[i] - imin) / range : 0.5;
        const r = peakSizeMin + (Number.isFinite(norm) ? norm : 0.5) * (peakSizeMax - peakSizeMin);
        ctx.beginPath();
        ctx.arc(px, py, r, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = peakColor;
        ctx.stroke();
      }
    },
    [showPeaks, peaksX, peaksY, peaksIntensity, centralIdx, centerX, centerY, peakColor, centralColor, peakSizeMin, peakSizeMax],
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
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
        <ImagePanel
          data={mapData}
          width={mapWidth}
          height={mapHeight}
          cmap={mapCmap}
          vmin={mapVmin}
          vmax={mapVmax}
          displayWidth={260}
          title={mapTitle}
          onPick={handleMapPick}
          cursor={mapCursor}
          cursorColor="#ff3b30"
        />
        <ImagePanel
          data={dpData}
          width={dpWidth}
          height={dpHeight}
          cmap={dpCmap}
          vmin={dpUseVmin}
          vmax={dpUseVmax}
          displayWidth={320}
          title={`Diffraction Pattern (Ry=${posRy}, Rx=${posRx})`}
          overlay={dpOverlay}
        />
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
