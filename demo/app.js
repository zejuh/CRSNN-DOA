const state = {
  summary: null,
  models: [],
  selectedModel: null,
  modelMenuOpen: false,
  selectedSample: "left",
  targetAngle: 20,
  predAngle: null,
  distribution: Array(36).fill(0),
  phase: 0,
  ringLayers: [],
  sparkField: [],
  ripples: [],
  lastWidth: 0,
  lastHeight: 0,
};

const canvas = document.getElementById("soundstage");
const ctx = canvas.getContext("2d");
const angleSlider = document.getElementById("angle-slider");
const modelPicker = document.getElementById("model-picker");
const modelSelectTrigger = document.getElementById("model-select-trigger");
const modelSelectLabel = document.getElementById("model-select-label");
const modelSelectSub = document.getElementById("model-select-sub");
const modelSelectMenu = document.getElementById("model-select-menu");
const sampleButtons = document.getElementById("sample-buttons");
const gtReadout = document.getElementById("gt-readout");
const predReadout = document.getElementById("pred-readout");
const confidenceReadout = document.getElementById("confidence-readout");
const errorReadout = document.getElementById("error-readout");
const deviceReadout = document.getElementById("device-readout");
const topbinReadout = document.getElementById("topbin-readout");
const barsEl = document.getElementById("bars");
const chartBestAcc = document.getElementById("chart-best-acc");
const chartBestMae = document.getElementById("chart-best-mae");
const chartLambdaAcc = document.getElementById("chart-lambda-acc");
const chartLambdaSynops = document.getElementById("chart-lambda-synops");
const chartSnrAcc = document.getElementById("chart-snr-acc");
const chartSnrMae = document.getElementById("chart-snr-mae");

let currentAudio = null;
let currentRequest = 0;
let predictTimer = null;
const modelPalette = {
  ConvRecSNN: "#7cc6ff",
  FlatLIFSNN: "#8dd694",
  CRNNBaseline: "#ffb86b",
  GCCPHATLSBaseline: "#ff7f50",
};

function degToRad(angle) {
  return (angle * Math.PI) / 180;
}

function normalizeDeg(angle) {
  return ((angle + 180) % 360 + 360) % 360 - 180;
}

function polarPoint(cx, cy, radiusX, radiusY, angleDeg) {
  const theta = degToRad(angleDeg - 90);
  return [cx + Math.cos(theta) * radiusX, cy + Math.sin(theta) * radiusY];
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function svgText(svg, x, y, text, attrs = {}) {
  const el = svgEl("text", { x, y, ...attrs });
  el.textContent = text;
  svg.appendChild(el);
  return el;
}

function clearSvg(svg) {
  svg.innerHTML = "";
}

function drawPlotFrame(svg, yTicks, yFormatter) {
  const width = 560;
  const height = 344;
  const m = { top: 12, right: 14, bottom: 68, left: 50 };
  const innerW = width - m.left - m.right;
  const innerH = height - m.top - m.bottom;

  svg.appendChild(svgEl("rect", {
    x: 0, y: 0, width, height, rx: 18,
    fill: "rgba(255,255,255,0.015)",
    stroke: "rgba(255,255,255,0.06)",
  }));

  for (const tick of yTicks) {
    const y = m.top + innerH * tick.ratio;
    svg.appendChild(svgEl("line", {
      x1: m.left, y1: y, x2: width - m.right, y2: y,
      stroke: "rgba(255,255,255,0.10)", "stroke-width": 1,
    }));
    svgText(svg, m.left - 8, y + 4, yFormatter(tick.value), {
      fill: "rgba(243,243,242,0.62)",
      "font-size": 18,
      "text-anchor": "end",
    });
  }

  svg.appendChild(svgEl("line", {
    x1: m.left, y1: m.top + innerH, x2: width - m.right, y2: m.top + innerH,
    stroke: "rgba(255,255,255,0.18)", "stroke-width": 1.2,
  }));

  return { width, height, m, innerW, innerH };
}

function renderBarChart(svg, rows, valueKey, errorKey, yMin, yMax, formatter) {
  clearSvg(svg);
  const layout = drawPlotFrame(
    svg,
    Array.from({ length: 5 }, (_, i) => ({
      value: yMax - ((yMax - yMin) * i) / 4,
      ratio: i / 4,
    })),
    formatter,
  );
  const { m, innerW, innerH } = layout;
  const band = innerW / rows.length;
  const barW = band * 0.62;

  const yPos = (value) => m.top + innerH * (1 - (value - yMin) / (yMax - yMin));

  rows.forEach((row, index) => {
    const x = m.left + index * band + (band - barW) / 2;
    const y = yPos(row[valueKey]);
    const zeroY = yPos(yMin);
    const color = modelPalette[row.model] || "#bbbbbb";

    svg.appendChild(svgEl("rect", {
      x, y, width: barW, height: Math.max(0, zeroY - y), rx: 10,
      fill: color, opacity: 0.88,
    }));

    const errTop = yPos(row[valueKey] + row[errorKey]);
    const errBottom = yPos(row[valueKey] - row[errorKey]);
    const cx = x + barW / 2;
    svg.appendChild(svgEl("line", { x1: cx, y1: errTop, x2: cx, y2: errBottom, stroke: "rgba(255,255,255,0.82)", "stroke-width": 1.6 }));
    svg.appendChild(svgEl("line", { x1: cx - 8, y1: errTop, x2: cx + 8, y2: errTop, stroke: "rgba(255,255,255,0.82)", "stroke-width": 1.6 }));
    svg.appendChild(svgEl("line", { x1: cx - 8, y1: errBottom, x2: cx + 8, y2: errBottom, stroke: "rgba(255,255,255,0.82)", "stroke-width": 1.6 }));

    svgText(svg, m.left + index * band + band / 2, m.top + innerH + 24, row.model.replace("Baseline", ""), {
      fill: "rgba(243,243,242,0.76)",
      "font-size": 16,
      "text-anchor": "middle",
    });
  });
}

function renderLineChart(svg, seriesMap, valueKey, errorKey, yMin, yMax, yFormatter) {
  clearSvg(svg);
  const lambdaOrder = [0.0, 0.03, 0.1, 0.3, 1.0];
  const lambdaLabels = ["0", "3e-2", "1e-1", "3e-1", "1"];
  const layout = drawPlotFrame(
    svg,
    Array.from({ length: 5 }, (_, i) => ({
      value: yMax - ((yMax - yMin) * i) / 4,
      ratio: i / 4,
    })),
    yFormatter,
  );
  const { width, m, innerW, innerH } = layout;
  const xPos = (index) => m.left + (innerW * index) / (lambdaOrder.length - 1);
  const yPos = (value) => m.top + innerH * (1 - (value - yMin) / (yMax - yMin));

  lambdaLabels.forEach((label, index) => {
    svgText(svg, xPos(index), m.top + innerH + 26, label, {
      fill: "rgba(243,243,242,0.76)",
      "font-size": 16,
      "text-anchor": "middle",
    });
  });

  const legendX = width - 150;
  let legendY = 28;

  Object.entries(seriesMap).forEach(([model, rows]) => {
    const color = modelPalette[model] || "#bbbbbb";
    const ordered = lambdaOrder.map((lambda) => rows.find((row) => Number(row.lambda) === lambda)).filter(Boolean);
    const upperPoints = ordered.map((row, index) => `${xPos(index)},${yPos(row[valueKey] + row[errorKey])}`);
    const lowerPoints = [...ordered]
      .reverse()
      .map((row) => `${xPos(lambdaOrder.indexOf(Number(row.lambda)))},${yPos(row[valueKey] - row[errorKey])}`);
    svg.appendChild(svgEl("polygon", {
      points: [...upperPoints, ...lowerPoints].join(" "),
      fill: color,
      opacity: 0.14,
    }));

    const points = ordered.map((row, index) => `${xPos(index)},${yPos(row[valueKey])}`).join(" ");
    svg.appendChild(svgEl("polyline", {
      points,
      fill: "none",
      stroke: color,
      "stroke-width": 3,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));

    ordered.forEach((row, index) => {
      const x = xPos(index);
      const y = yPos(row[valueKey]);
      const errTop = yPos(row[valueKey] + row[errorKey]);
      const errBottom = yPos(row[valueKey] - row[errorKey]);

      svg.appendChild(svgEl("line", { x1: x, y1: errTop, x2: x, y2: errBottom, stroke: color, "stroke-width": 1.4, opacity: 0.7 }));
      svg.appendChild(svgEl("circle", { cx: x, cy: y, r: 4.8, fill: color }));
    });

    svg.appendChild(svgEl("line", { x1: legendX, y1: legendY, x2: legendX + 24, y2: legendY, stroke: color, "stroke-width": 3, "stroke-linecap": "round" }));
    svgText(svg, legendX + 30, legendY + 4, model, {
      fill: "rgba(243,243,242,0.78)",
      "font-size": 13,
    });
    legendY += 18;
  });
}

function renderTrendChart(svg, seriesMap, xKey, valueKey, yMin, yMax, yFormatter, showLegend = true) {
  clearSvg(svg);
  const xValues = [...new Set(Object.values(seriesMap).flat().map((row) => row[xKey]))].sort((a, b) => a - b);
  if (xValues.length === 0) return;

  const xLabels = xValues.map((value) => `${value}`);
  const layout = drawPlotFrame(
    svg,
    Array.from({ length: 5 }, (_, i) => ({
      value: yMax - ((yMax - yMin) * i) / 4,
      ratio: i / 4,
    })),
    yFormatter,
  );
  const { width, m, innerW, innerH } = layout;
  const xPos = (index) => m.left + (innerW * index) / Math.max(xValues.length - 1, 1);
  const yPos = (value) => m.top + innerH * (1 - (value - yMin) / (yMax - yMin));

  xLabels.forEach((label, index) => {
    svgText(svg, xPos(index), m.top + innerH + 26, label, {
      fill: "rgba(243,243,242,0.76)",
      "font-size": 16,
      "text-anchor": "middle",
    });
  });

  const legendX = width - 190;
  let legendY = 20;

  Object.entries(seriesMap).forEach(([model, rows]) => {
    const color = modelPalette[model] || "#bbbbbb";
    const ordered = [...rows].sort((a, b) => a[xKey] - b[xKey]);
    const points = ordered.map((row) => `${xPos(xValues.indexOf(row[xKey]))},${yPos(row[valueKey])}`).join(" ");
    svg.appendChild(svgEl("polyline", {
      points,
      fill: "none",
      stroke: color,
      "stroke-width": 2,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));

    ordered.forEach((row) => {
      const x = xPos(xValues.indexOf(row[xKey]));
      const y = yPos(row[valueKey]);
      svg.appendChild(svgEl("circle", { cx: x, cy: y, r: 4.8, fill: color }));
    });

    if (showLegend) {
      svg.appendChild(svgEl("line", { x1: legendX, y1: legendY, x2: legendX + 24, y2: legendY, stroke: color, "stroke-width": 3, "stroke-linecap": "round" }));
      svgText(svg, legendX + 30, legendY + 4, model, {
        fill: "rgba(243,243,242,0.78)",
        "font-size": 13,
      });
      legendY += 26;
    }
  });
}

function renderBenchmarkCharts(benchmark) {
  if (!benchmark) return;

  const bestRows = benchmark.best_noisy || [];
  renderBarChart(chartBestAcc, bestRows, "acc_mean", "acc_std", 0.6, 0.86, (v) => v.toFixed(2));
  renderBarChart(chartBestMae, bestRows, "ang_mae_deg_mean", "ang_mae_deg_std", 0.0, 4.8, (v) => v.toFixed(1));

  const lambdaStudy = benchmark.lambda_study || {};
  renderLineChart(chartLambdaAcc, lambdaStudy, "acc_mean", "acc_std", 0.68, 0.89, (v) => v.toFixed(2));
  renderLineChart(
    chartLambdaSynops,
    Object.fromEntries(
      Object.entries(lambdaStudy).map(([model, rows]) => [
        model,
        rows.map((row) => ({
          ...row,
          synops_per_sample_mean: row.synops_per_sample_mean / 1e3,
          synops_per_sample_std: row.synops_per_sample_std / 1e3,
        })),
      ]),
    ),
    "synops_per_sample_mean",
    "synops_per_sample_std",
    120,
    1400,
    (v) => `${Math.round(v)}`,
  );

  const snrStudy = benchmark.snr_robustness || {};
  renderTrendChart(chartSnrAcc, snrStudy, "snr_db", "acc", 0.1, 1.0, (v) => v.toFixed(2), false);
  renderTrendChart(chartSnrMae, snrStudy, "snr_db", "ang_mae_deg", 0.0, 40.0, (v) => v.toFixed(0), true);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = rect.width;
  const height = rect.height;
  if (width !== state.lastWidth || height !== state.lastHeight) {
    state.lastWidth = width;
    state.lastHeight = height;
    rebuildParticleField(width, height);
  }
}

function rebuildParticleField(width, height) {
  state.ringLayers = Array.from({ length: 7 }, (_, index) => ({
    count: 140 + index * 80,
    radiusX: width * (0.1 + index * 0.115),
    radiusY: height * (0.06 + index * 0.073),
    drift: (index % 2 === 0 ? 1 : -1) * (0.00022 + index * 0.00005),
    scatter: 8 + index * 3.6,
    alpha: 0.09 + index * 0.023,
    size: 0.6 + index * 0.12,
    offsetY: -height * 0.06 + index * height * 0.012,
  }));
  state.sparkField = [];
}

function setMetricText(id, value) {
  document.getElementById(id).textContent = value;
}

function getSelectedModel() {
  return state.models.find((item) => item.id === state.selectedModel) || null;
}

function formatModelSubLabel(model) {
  if (!model) return "";
  if (model.family === "classical") {
    return "classical baseline";
  }
  if (model.model === "CRNNBaseline") {
    return "learned baseline";
  }
  if (String(model.label).includes("noisy-best")) {
    return `noisy best / lambda=${Number(model.lambda).toExponential(0)}`;
  }
  if (String(model.label).includes("val-selected")) {
    return `val selected / lambda=${Number(model.lambda).toExponential(0)}`;
  }
  return `${model.family} / lambda=${Number(model.lambda).toExponential(0)}`;
}

function setModelMenuOpen(isOpen) {
  state.modelMenuOpen = isOpen;
  modelPicker.classList.toggle("open", isOpen);
  modelSelectTrigger.setAttribute("aria-expanded", String(isOpen));
}

function selectModel(modelId, shouldPredict = true) {
  state.selectedModel = modelId;
  updateModelMetrics();
  highlightModelMenu();
  if (shouldPredict) {
    schedulePrediction();
  }
}

function updateModelMetrics() {
  const model = getSelectedModel();
  if (!model) return;

  modelSelectLabel.textContent = model.model;
  modelSelectSub.textContent = formatModelSubLabel(model);
  setMetricText("metric-model", model.model);
  setMetricText("metric-model-sub", formatModelSubLabel(model));
  setMetricText("metric-acc", `${(model.metrics.noisy_acc * 100).toFixed(2)}%`);
  setMetricText("metric-angle", `${model.metrics.noisy_mae_deg.toFixed(2)} deg`);
}

function highlightModelMenu() {
  for (const option of modelSelectMenu.querySelectorAll(".model-option")) {
    option.classList.toggle("active", option.dataset.id === state.selectedModel);
  }
}

function buildModelSelect(models) {
  modelSelectMenu.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "model-option";
    option.dataset.id = model.id;
    option.innerHTML = `
      <span class="model-option-label">${model.model}</span>
      <span class="model-option-meta">${formatModelSubLabel(model)}</span>
    `;
    option.addEventListener("click", () => {
      setModelMenuOpen(false);
      selectModel(model.id);
    });
    modelSelectMenu.appendChild(option);
  }
  highlightModelMenu();
}

function buildSampleButtons(samples) {
  sampleButtons.innerHTML = "";
  for (const sample of samples) {
    const button = document.createElement("button");
    button.className = "sample-button";
    button.dataset.key = sample.key;
    button.textContent = sample.label;
    button.addEventListener("click", () => {
      state.selectedSample = sample.key;
      updateSampleButtons();
      schedulePrediction();
    });
    sampleButtons.appendChild(button);
  }
  updateSampleButtons();
}

function updateSampleButtons() {
  for (const button of sampleButtons.querySelectorAll(".sample-button")) {
    button.classList.toggle("active", button.dataset.key === state.selectedSample);
  }
}

function initBars(count) {
  barsEl.innerHTML = "";
  for (let i = 0; i < count; i += 1) {
    const bar = document.createElement("div");
    bar.className = "bar";
    barsEl.appendChild(bar);
  }
}

function updateBars(distribution, topBins) {
  const topAngles = new Set((topBins || []).map((item) => Math.round(item.angle_deg)));
  const max = Math.max(...distribution, 1e-6);
  [...barsEl.children].forEach((bar, index) => {
    const value = distribution[index] / max;
    const angle = Math.round(state.summary.binCentersDeg[index]);
    bar.style.height = `${10 + value * 124}px`;
    bar.classList.toggle("active", topAngles.has(angle));
  });
}

function angleDistance(a, b) {
  return Math.abs(normalizeDeg(a - b));
}

function drawBackground(width, height, cx, cy) {
  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#1f1f21");
  bg.addColorStop(0.34, "#111113");
  bg.addColorStop(1, "#060607");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const halo = ctx.createRadialGradient(cx, cy - height * 0.18, width * 0.03, cx, cy - height * 0.18, width * 0.55);
  halo.addColorStop(0, "rgba(255,255,255,0.20)");
  halo.addColorStop(0.28, "rgba(255,255,255,0.08)");
  halo.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, width, height);
}

function drawRingLayer(layer, cx, cy, width, activeAngles) {
  const time = state.phase;
  for (let i = 0; i < layer.count; i += 1) {
    const ratio = i / layer.count;
    const theta = ratio * Math.PI * 2 + time * layer.drift;
    const jitter = Math.sin(theta * (2.2 + layer.size) + time * 0.0026 + i * 0.07) * layer.scatter;
    const rx = layer.radiusX + jitter;
    const ry = layer.radiusY + jitter * 0.68;
    const x = cx + Math.cos(theta) * rx;
    const y = cy + Math.sin(theta) * ry + layer.offsetY;
    const renderAngle = normalizeDeg((theta * 180) / Math.PI + 90);

    let alpha = layer.alpha;
    let size = layer.size;
    for (const activeAngle of activeAngles) {
      const diff = angleDistance(renderAngle, activeAngle.angle);
      if (diff < 22) {
        const boost = (22 - diff) / 22;
        alpha += boost * activeAngle.alpha;
        size += boost * activeAngle.size;
      }
    }

    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${Math.min(alpha, 0.95)})`;
    ctx.fill();
  }
}

function drawDistributionBeams(cx, cy, outerX, outerY) {
  const distribution = state.distribution || [];
  distribution.forEach((value, index) => {
    if (!value) return;
    const angle = state.summary.binCentersDeg[index];
    const [x1, y1] = polarPoint(cx, cy, outerX * 0.88, outerY * 0.88, angle);
    const [x2, y2] = polarPoint(cx, cy, outerX * (0.92 + value * 0.12), outerY * (0.92 + value * 0.08), angle);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = `rgba(255,255,255,${0.04 + value * 0.22})`;
    ctx.lineWidth = 1 + value * 2;
    ctx.stroke();
  });
}

function drawPulsePath(cx, cy, target, pred) {
  const points = [];
  for (let i = 0; i <= 28; i += 1) {
    const t = i / 28;
    const x = target[0] * (1 - t) + cx * t + Math.sin(state.phase * 0.02 + i * 0.6) * (1 - t) * 8;
    const y = target[1] * (1 - t) + cy * t + Math.cos(state.phase * 0.018 + i * 0.55) * (1 - t) * 8;
    points.push([x, y]);
  }

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.24)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  points.forEach((point, index) => {
    const glow = (Math.sin(state.phase * 0.05 - index * 0.55) + 1) * 0.5;
    ctx.beginPath();
    ctx.arc(point[0], point[1], 1.2 + glow * 2.6, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${0.06 + glow * 0.25})`;
    ctx.fill();
  });

  if (pred) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(pred[0], pred[1]);
    ctx.strokeStyle = "rgba(255,209,138,0.5)";
    ctx.lineWidth = 2.2;
    ctx.stroke();
  }
}

function drawRipples(cx, cy, target) {
  state.ripples = state.ripples.filter((ripple) => state.phase - ripple.birth < 136);

  state.ripples.forEach((ripple) => {
    const age = (state.phase - ripple.birth) / 136;
    const fade = 1 - age;
    const point = polarPoint(cx, cy, ripple.radiusX, ripple.radiusY, ripple.angle);
    const direction = Math.atan2(point[1] - cy, point[0] - cx);
    const flashRadius = 10 + age * 28;

    const flash = ctx.createRadialGradient(point[0], point[1], 0, point[0], point[1], flashRadius);
    flash.addColorStop(0, `rgba(255,255,255,${fade * 0.56})`);
    flash.addColorStop(0.45, `rgba(255,255,255,${fade * 0.18})`);
    flash.addColorStop(1, "rgba(255,255,255,0)");
    ctx.beginPath();
    ctx.arc(point[0], point[1], flashRadius, 0, Math.PI * 2);
    ctx.fillStyle = flash;
    ctx.fill();

    ripple.bands.forEach((band, bandIndex) => {
      const t = (age - band.delay) / band.duration;
      if (t <= 0 || t >= 1) return;

      const bandFade = (1 - t) * fade;
      const spread = band.spread;
      const radius = band.startRadius + t * band.travel;

      for (let i = 0; i < band.count; i += 1) {
        const ratio = band.count === 1 ? 0.5 : i / (band.count - 1);
        const localAngle = -spread / 2 + ratio * spread;
        const jitter = Math.sin(ripple.seed + bandIndex * 1.9 + i * 0.37) * band.jitter;
        const theta = direction + localAngle + jitter * 0.02;
        const shellX = radius * (1.04 + 0.1 * Math.cos(localAngle * 1.6));
        const shellY = radius * (0.82 + 0.18 * Math.cos(localAngle * 2.1));
        const px = point[0] + Math.cos(theta) * shellX;
        const py = point[1] + Math.sin(theta) * shellY;
        const centerWeight = 1 - Math.abs(localAngle) / (spread / 2);
        const alpha = Math.max(0, bandFade * band.alpha * (0.35 + centerWeight * 1.15));
        const size = band.size * (1.2 + centerWeight * 1.35);

        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.fill();
      }
    });
  });

  if (target) {
    ctx.beginPath();
    ctx.arc(target[0], target[1], 11, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.09)";
    ctx.fill();
  }
}

function drawCenterOrb(cx, cy) {
  const pulse = (Math.sin(state.phase * 0.03) + 1) * 0.5;
  for (let i = 0; i < 3; i += 1) {
    ctx.beginPath();
    ctx.ellipse(cx, cy, 60 + i * 30 + pulse * 10, 42 + i * 20 + pulse * 6, 0, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${0.12 - i * 0.02})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  const orb = ctx.createRadialGradient(cx - 8, cy - 12, 6, cx, cy, 58);
  orb.addColorStop(0, "rgba(255,255,255,0.92)");
  orb.addColorStop(0.35, "rgba(228,228,228,0.70)");
  orb.addColorStop(0.7, "rgba(88,88,88,0.92)");
  orb.addColorStop(1, "rgba(28,28,28,1)");
  ctx.beginPath();
  ctx.arc(cx, cy, 58, 0, Math.PI * 2);
  ctx.fillStyle = orb;
  ctx.shadowColor = "rgba(255,255,255,0.24)";
  ctx.shadowBlur = 28;
  ctx.fill();
  ctx.shadowBlur = 0;

  const micAngles = [-135, -45, 45, 135];
  micAngles.forEach((angle) => {
    const [mx, my] = polarPoint(cx, cy, 86, 64, angle);
    ctx.beginPath();
    ctx.arc(mx, my, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.42)";
    ctx.fill();
  });
}

function drawLabels(cx, cy, outerX, outerY) {
  ctx.fillStyle = "rgba(255,255,255,0.46)";
  ctx.font = "13px Aptos Display, Bahnschrift, sans-serif";
  [-180, -120, -60, 0, 60, 120, 180].forEach((angle) => {
    const [lx, ly] = polarPoint(cx, cy, outerX * 1.02, outerY * 1.02, angle);
    ctx.fillText(`${angle}`, lx - 12, ly + 4);
  });
}

function drawStage() {
  const width = canvas.getBoundingClientRect().width;
  const height = canvas.getBoundingClientRect().height;
  const cx = width * 0.5;
  const cy = height * 0.42;
  const outerX = width * 0.42;
  const outerY = height * 0.33;

  drawBackground(width, height, cx, cy);

  const activeAngles = [
    { angle: state.targetAngle, alpha: 0.2, size: 1.1 },
    { angle: state.predAngle ?? state.targetAngle, alpha: 0.28, size: 1.35 },
  ];

  state.ringLayers.forEach((layer) => {
    drawRingLayer(layer, cx, cy, width, activeAngles);
  });

  drawDistributionBeams(cx, cy, outerX, outerY);

  const targetPoint = polarPoint(cx, cy, outerX * 0.86, outerY * 0.86, state.targetAngle);
  const predPoint = state.predAngle === null ? null : polarPoint(cx, cy, outerX * 0.72, outerY * 0.72, state.predAngle);
  drawRipples(cx, cy, targetPoint);
  drawPulsePath(cx, cy, targetPoint, predPoint);
  drawCenterOrb(cx, cy);

  ctx.beginPath();
  ctx.arc(targetPoint[0], targetPoint[1], 10, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fill();

  if (predPoint) {
    ctx.beginPath();
    ctx.arc(predPoint[0], predPoint[1], 7.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,209,138,0.98)";
    ctx.fill();
  }

  drawLabels(cx, cy, outerX, outerY);
}

function animate() {
  state.phase += 1;
  if (state.summary) {
    drawStage();
  }
  requestAnimationFrame(animate);
}

async function requestPrediction() {
  const requestId = ++currentRequest;
  gtReadout.textContent = `${Math.round(state.targetAngle)} deg`;
  predReadout.textContent = "...";

  try {
    const response = await fetch(
      `/api/predict?model=${encodeURIComponent(state.selectedModel)}&sample=${encodeURIComponent(state.selectedSample)}&angle=${encodeURIComponent(state.targetAngle)}`,
    );
    const payload = await response.json();
    if (requestId !== currentRequest || payload.error) return;

    state.predAngle = payload.predAngleDeg;
    state.distribution = payload.distribution;

    predReadout.textContent = `${payload.predAngleDeg.toFixed(1)} deg`;
    confidenceReadout.textContent = `${(payload.confidence * 100).toFixed(1)}%`;
    errorReadout.textContent = `${payload.errorDeg.toFixed(2)} deg`;
    topbinReadout.textContent = `${payload.topBins[0].angle_deg.toFixed(0)} deg`;
    updateBars(payload.distribution, payload.topBins);
  } catch (error) {
    console.error(error);
  }
}

function schedulePrediction() {
  if (predictTimer) {
    clearTimeout(predictTimer);
  }
  predictTimer = setTimeout(() => {
    predictTimer = null;
    requestPrediction();
  }, 120);
}

function spawnRipple(angle) {
  const width = canvas.getBoundingClientRect().width;
  const height = canvas.getBoundingClientRect().height;
  state.ripples.push({
    angle,
    birth: state.phase,
    radiusX: width * 0.42 * 0.86,
    radiusY: height * 0.33 * 0.86,
    seed: Math.random() * Math.PI * 2,
    bands: [
      { delay: 0.0, duration: 0.72, startRadius: 10, travel: 170, spread: Math.PI * 1.28, count: 96, alpha: 0.22, size: 1.35, jitter: 0.9 },
      { delay: 0.06, duration: 0.78, startRadius: 22, travel: 230, spread: Math.PI * 1.56, count: 132, alpha: 0.17, size: 1.12, jitter: 1.15 },
      { delay: 0.14, duration: 0.84, startRadius: 34, travel: 290, spread: Math.PI * 1.82, count: 164, alpha: 0.12, size: 1.0, jitter: 1.4 },
      { delay: 0.24, duration: 0.9, startRadius: 50, travel: 340, spread: Math.PI * 2.02, count: 188, alpha: 0.08, size: 0.9, jitter: 1.65 },
    ],
  });
}

async function loadSummary() {
  const response = await fetch("/api/summary");
  const summary = await response.json();
  summary.binCentersDeg = Array.from(
    { length: summary.config.doa_bins },
    (_, index) => normalizeDeg(-180 + (index + 0.5) * (360 / summary.config.doa_bins)),
  );

  state.summary = summary;
  state.models = summary.models || [];
  state.selectedModel = summary.defaultModel || (state.models[0] && state.models[0].id);
  state.distribution = Array(summary.config.doa_bins).fill(0);

  buildModelSelect(state.models);
  updateModelMetrics();
  renderBenchmarkCharts(summary.benchmark);
  deviceReadout.textContent = summary.device.toUpperCase();
  buildSampleButtons(summary.samples);
  initBars(summary.config.doa_bins);
}

async function playCurrentSample() {
  if (!state.summary) return;
  const sample = state.summary.samples.find((item) => item.key === state.selectedSample);
  if (!sample) return;

  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  const audio = new Audio(sample.audio);
  audio.crossOrigin = "anonymous";

  const context = new (window.AudioContext || window.webkitAudioContext)();
  const source = context.createMediaElementSource(audio);
  const panner = context.createStereoPanner();
  const gain = context.createGain();

  panner.pan.value = Math.sin(degToRad(state.targetAngle));
  gain.gain.value = 1;

  source.connect(panner).connect(gain).connect(context.destination);
  await context.resume();
  currentAudio = audio;
  await audio.play();
}

function updateAngle(value) {
  state.targetAngle = normalizeDeg(Number(value));
  angleSlider.value = state.targetAngle;
  gtReadout.textContent = `${Math.round(state.targetAngle)} deg`;
  schedulePrediction();
}

function attachStageInteraction() {
  const angleFromPointer = (clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height * 0.42;
    const angle = Math.atan2(clientY - cy, clientX - cx);
    return normalizeDeg((angle * 180) / Math.PI + 90);
  };

  canvas.addEventListener("pointerdown", (event) => {
    const angle = angleFromPointer(event.clientX, event.clientY);
    state.targetAngle = angle;
    angleSlider.value = angle;
    gtReadout.textContent = `${Math.round(angle)} deg`;
    spawnRipple(angle);
    requestPrediction();
    playCurrentSample();
  });
}

async function bootstrap() {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  angleSlider.addEventListener("input", (event) => updateAngle(event.target.value));
  modelSelectTrigger.addEventListener("click", () => {
    setModelMenuOpen(!state.modelMenuOpen);
  });
  document.addEventListener("click", (event) => {
    if (!modelPicker.contains(event.target)) {
      setModelMenuOpen(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setModelMenuOpen(false);
    }
  });
  attachStageInteraction();
  await loadSummary();
  await requestPrediction();
  animate();
}

bootstrap();
