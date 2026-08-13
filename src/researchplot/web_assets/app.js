"use strict";

const params = new URLSearchParams(location.search);
const token = params.get("token") || "";
history.replaceState({}, "", "/");

const elements = Object.fromEntries([
  "profile", "width", "role", "content", "profile-detail", "artifact", "dropzone",
  "status", "results", "verdict", "summary", "coverage-note", "findings", "command",
  "download", "copy-command", "preview-card", "preview", "preview-buttons",
].map((id) => [id, document.getElementById(id)]));

let profiles = [];
let currentPayload = null;
let sourceImage = null;

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-ResearchPlot-Token", token);
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function selectedProfile() {
  return profiles.find((item) => item.coordinate === elements.profile.value);
}

function updateProfile() {
  const profile = selectedProfile();
  elements.width.replaceChildren();
  if (!profile) return;
  const widths = profile.widths.length ? profile.widths : [""];
  for (const width of widths) {
    const option = document.createElement("option");
    option.value = width;
    option.textContent = width || "Not specified";
    option.selected = width === profile.default_width;
    elements.width.append(option);
  }
  const caveat = profile.caveats.length ? ` Caveat: ${profile.caveats[0]}` : "";
  elements["profile-detail"].textContent = `${profile.coordinate} · verified ${profile.verified_on}.${caveat}`;
}

async function loadProfiles() {
  try {
    const payload = await api("/api/profiles");
    profiles = payload.profiles;
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.coordinate;
      option.textContent = `${profile.name} (${profile.coordinate})`;
      elements.profile.append(option);
    }
    updateProfile();
  } catch (error) {
    elements.status.textContent = error.message;
  }
}

function metric(label, value) {
  const node = document.createElement("div");
  node.className = "metric";
  const strong = document.createElement("strong");
  strong.textContent = String(value ?? 0);
  node.append(strong, document.createTextNode(label));
  return node;
}

function render(payload) {
  currentPayload = payload;
  const report = payload.report;
  elements.results.classList.remove("hidden");
  elements.verdict.textContent = report.verdict.replace("_", " ");
  elements.verdict.className = `verdict ${report.verdict}`;
  elements.summary.replaceChildren(
    metric(" findings", report.summary.findings),
    metric(" failures", report.summary.failures),
    metric(" warnings", report.summary.warnings),
    metric(" unresolved", report.summary.unresolved),
  );
  elements["coverage-note"].textContent = report.verdict === "indeterminate"
    ? "Required evidence is unresolved. This result is not a compliance pass."
    : "The verdict reflects the evidence phases listed in this report.";
  elements.findings.replaceChildren();
  for (const finding of report.findings) {
    const row = document.createElement("tr");
    const status = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `status ${finding.outcome}`;
    badge.textContent = finding.outcome;
    status.append(badge);
    const rule = document.createElement("td");
    const code = document.createElement("code");
    code.textContent = finding.rule_id;
    rule.append(code);
    const evidence = document.createElement("td");
    evidence.textContent = finding.message;
    const action = document.createElement("td");
    action.textContent = finding.suggestion || "Review the cited venue source.";
    row.append(status, rule, evidence, action);
    elements.findings.append(row);
  }
  elements.command.textContent = payload.command;
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

const matrices = {
  original: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  grayscale: [.2126, .7152, .0722, .2126, .7152, .0722, .2126, .7152, .0722],
  protanopia: [.567, .433, 0, .558, .442, 0, 0, .242, .758],
  deuteranopia: [.625, .375, 0, .7, .3, 0, 0, .3, .7],
  tritanopia: [.95, .05, 0, 0, .433, .567, 0, .475, .525],
};

function drawPreview(mode = "original") {
  if (!sourceImage) return;
  const canvas = elements.preview;
  const scale = Math.min(1, 1000 / sourceImage.width, 620 / sourceImage.height);
  canvas.width = Math.max(1, Math.round(sourceImage.width * scale));
  canvas.height = Math.max(1, Math.round(sourceImage.height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
  const matrix = matrices[mode];
  if (mode !== "original") {
    const image = context.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 0; index < image.data.length; index += 4) {
      const r = image.data[index], g = image.data[index + 1], b = image.data[index + 2];
      image.data[index] = Math.min(255, matrix[0] * r + matrix[1] * g + matrix[2] * b);
      image.data[index + 1] = Math.min(255, matrix[3] * r + matrix[4] * g + matrix[5] * b);
      image.data[index + 2] = Math.min(255, matrix[6] * r + matrix[7] * g + matrix[8] * b);
    }
    context.putImageData(image, 0, 0);
  }
}

function preparePreview(file) {
  const isRaster = /image\/(png|jpeg|tiff)/.test(file.type) || /\.(png|jpe?g|tiff?)$/i.test(file.name);
  if (!isRaster) {
    elements["preview-card"].classList.add("hidden");
    sourceImage = null;
    return;
  }
  const image = new Image();
  const url = URL.createObjectURL(file);
  image.onload = () => {
    sourceImage = image;
    elements["preview-card"].classList.remove("hidden");
    drawPreview();
    URL.revokeObjectURL(url);
  };
  image.src = url;
}

async function audit(file) {
  if (!file) return;
  elements.status.textContent = `Auditing ${file.name} locally…`;
  elements.results.classList.add("hidden");
  preparePreview(file);
  const query = new URLSearchParams({
    profile: elements.profile.value,
    width: elements.width.value,
    role: elements.role.value,
    content: elements.content.value,
    filename: file.name,
  });
  try {
    const payload = await api(`/api/audit?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    elements.status.textContent = `Finished ${file.name}; ${payload.bytes.toLocaleString()} bytes processed locally.`;
    render(payload);
  } catch (error) {
    elements.status.textContent = `Audit failed: ${error.message}`;
  }
}

elements.profile.addEventListener("change", updateProfile);
elements.artifact.addEventListener("change", () => audit(elements.artifact.files[0]));
elements.dropzone.addEventListener("dragover", (event) => { event.preventDefault(); elements.dropzone.classList.add("drag"); });
elements.dropzone.addEventListener("dragleave", () => elements.dropzone.classList.remove("drag"));
elements.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("drag");
  audit(event.dataTransfer.files[0]);
});
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") elements.artifact.click();
});
elements["preview-buttons"].addEventListener("click", (event) => {
  if (!(event.target instanceof HTMLButtonElement)) return;
  for (const button of elements["preview-buttons"].querySelectorAll("button")) button.classList.remove("active");
  event.target.classList.add("active");
  drawPreview(event.target.dataset.mode);
});
elements.download.addEventListener("click", () => {
  if (!currentPayload) return;
  const blob = new Blob([JSON.stringify(currentPayload.report, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "researchplot-report.json";
  link.click();
  URL.revokeObjectURL(link.href);
});
elements["copy-command"].addEventListener("click", async () => {
  if (!currentPayload) return;
  await navigator.clipboard.writeText(currentPayload.command);
  elements["copy-command"].textContent = "Copied";
});

loadProfiles();
