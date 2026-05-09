/**
 * Image Recognition System — Frontend Logic
 * Handles uploads, API calls, result rendering, and Chart.js histograms.
 */

// ============================================================
// State
// ============================================================
let currentFile = null;
let filterFile = null;
let compareFile1 = null;
let compareFile2 = null;
let histogramChart = null;

// ============================================================
// DOM Ready
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUploadZone('upload-zone', 'image-input', handleMainUpload);
    initUploadZone('filter-upload-zone', 'filter-image-input', handleFilterUpload);
    initCompareUploads();
    initButtons();
});

// ============================================================
// Tab Navigation
// ============================================================
function initTabs() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
        });
    });

    // Result sub-tabs
    document.querySelectorAll('.result-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.dataset.result;
            document.querySelectorAll('.result-tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.result-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`panel-${panel}`).classList.add('active');
        });
    });
}

// ============================================================
// Upload Zone (Drag & Drop + Click)
// ============================================================
function initUploadZone(zoneId, inputId, callback) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            // FIX #3: ORIGINAL BUG: `input.files = e.dataTransfer.files`
            // FileList is a read-only property in Firefox and throws a
            // TypeError: Setting a property that has only a getter.
            // The assignment was also unnecessary — the callback already
            // receives the file directly from e.dataTransfer.files[0].
            // FIX: Simply call the callback with the file; remove the
            // read-only assignment entirely.
            callback(file);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) callback(input.files[0]);
    });
}

// ============================================================
// Main Upload Handler
// ============================================================
function handleMainUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('preview-image').src = e.target.result;
        document.getElementById('upload-zone').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');
        document.getElementById('results-container').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// ============================================================
// Buttons
// ============================================================
function initButtons() {
    document.getElementById('btn-analyze')?.addEventListener('click', analyzeImage);
    document.getElementById('btn-change-image')?.addEventListener('click', () => {
        document.getElementById('preview-container').classList.add('hidden');
        document.getElementById('upload-zone').classList.remove('hidden');
        document.getElementById('results-container').classList.add('hidden');
        document.getElementById('image-input').value = '';
        currentFile = null;
    });
    document.getElementById('btn-compare')?.addEventListener('click', compareImages);
    document.getElementById('btn-change-filter')?.addEventListener('click', resetFilterTab);
}

// ============================================================
// Analyze Image
// ============================================================
async function analyzeImage() {
    if (!currentFile) return;

    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('loading-container').classList.remove('hidden');
    document.getElementById('results-container').classList.add('hidden');

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        analysisData = data;
        renderResults(data);
    } catch (err) {
        alert('Analysis failed: ' + err.message);
        document.getElementById('preview-container').classList.remove('hidden');
    } finally {
        document.getElementById('loading-container').classList.add('hidden');
    }
}

// ============================================================
// Render Results
// ============================================================
function renderResults(data) {
    const container = document.getElementById('results-container');
    container.classList.remove('hidden');

    document.getElementById('results-time').textContent = `Analyzed at ${data.analyzed_at}`;

    renderStats(data.stats, data.faces.count, data.contour_count);

    document.getElementById('face-result-img').src = `data:image/jpeg;base64,${data.face_image}`;
    renderFaceDetails(data.faces);

    renderColors(data.colors);

    document.getElementById('edge-result-img').src = `data:image/jpeg;base64,${data.edge_image}`;

    renderHistogram(data.histogram);

    document.getElementById('contour-result-img').src = `data:image/jpeg;base64,${data.contour_image}`;
    document.getElementById('contour-info').textContent =
        `${data.contour_count} contours detected in the image`;

    // Reset to first tab
    document.querySelectorAll('.result-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.result-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('.result-tab').classList.add('active');
    document.querySelector('.result-panel').classList.add('active');

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderStats(stats, faceCount, contourCount) {
    const grid = document.getElementById('stats-grid');
    const items = [
        { label: 'Dimensions', value: stats.dimensions, color: 'cyan' },
        { label: 'Faces Found', value: faceCount, sub: faceCount === 0 ? 'No faces' : `${faceCount} face(s)`, color: 'purple' },
        { label: 'Brightness', value: stats.brightness, sub: stats.brightness_label, color: 'pink' },
        { label: 'Sharpness', value: stats.sharpness, sub: stats.sharpness_label, color: 'blue' },
        { label: 'Contrast', value: stats.contrast, sub: stats.contrast_label, color: 'cyan' },
        { label: 'Saturation', value: stats.saturation, sub: stats.saturation_label, color: 'purple' },
        { label: 'Temperature', value: stats.color_temperature, color: 'pink' },
        { label: 'Contours', value: contourCount, color: 'blue' },
    ];
    grid.innerHTML = items.map(item => `
        <div class="stat-card">
            <div class="stat-label">${item.label}</div>
            <div class="stat-value ${item.color}">${item.value}</div>
            ${item.sub ? `<div class="stat-sub">${item.sub}</div>` : ''}
        </div>
    `).join('');
}

function renderFaceDetails(faces) {
    const container = document.getElementById('face-details');
    if (faces.count === 0) {
        container.innerHTML = `
            <div class="no-faces">
                <p>No faces detected in this image.</p>
                <p style="font-size:0.8rem;margin-top:0.5rem;color:var(--text-muted)">
                    Try uploading a photo with visible human faces.
                </p>
            </div>`;
        return;
    }
    container.innerHTML = faces.details.map(f => `
        <div class="face-item">
            <div class="face-item-icon">👤</div>
            <div class="face-item-info">
                <h4>Face ${f.id}</h4>
                <p>Position: (${f.x}, ${f.y}) • Size: ${f.width}×${f.height}px • Eyes: ${f.eyes_detected}</p>
            </div>
            <span class="face-badge">${f.smile_detected ? '😊 Smile' : '😐 Neutral'}</span>
        </div>
    `).join('');
}

function renderColors(colors) {
    const palette = document.getElementById('color-palette');
    palette.innerHTML = colors.map(c => `
        <div class="color-swatch" title="${c.hex}">
            <div class="swatch-circle" style="background:${c.hex}"></div>
            <div class="swatch-hex">${c.hex}</div>
            <div class="swatch-pct">${c.percentage}%</div>
        </div>
    `).join('');

    const bars = document.getElementById('color-bars');
    bars.innerHTML = colors.map(c => `
        <div class="color-bar-item">
            <span class="color-bar-label">${c.percentage}%</span>
            <div class="color-bar-fill" style="background:${c.hex};width:${c.percentage}%;max-width:100%"></div>
        </div>
    `).join('');
}

function renderHistogram(histData) {
    const ctx = document.getElementById('histogram-chart').getContext('2d');
    if (histogramChart) histogramChart.destroy();

    const labels = Array.from({ length: histData.red.length }, (_, i) => i * 4);

    histogramChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Red',
                    data: histData.red,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255,107,107,0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Green',
                    data: histData.green,
                    borderColor: '#51cf66',
                    backgroundColor: 'rgba(81,207,102,0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Blue',
                    data: histData.blue,
                    borderColor: '#339af0',
                    backgroundColor: 'rgba(51,154,240,0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8892a8', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Pixel Intensity', color: '#556180' },
                    ticks: { color: '#556180', maxTicksLimit: 10 },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    title: { display: true, text: 'Frequency', color: '#556180' },
                    ticks: { color: '#556180' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
            },
        },
    });
}

// ============================================================
// Filter Tab
// ============================================================
function resetFilterTab() {
    filterFile = null;
    document.getElementById('filter-workspace').classList.add('hidden');
    document.getElementById('filter-upload-zone').classList.remove('hidden');
    document.getElementById('filter-image-input').value = '';
    document.getElementById('filter-result-title').textContent = 'Filtered';
    // Clear active state on all filter buttons
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
}

function handleFilterUpload(file) {
    if (!file.type.startsWith('image/')) return;
    filterFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('filter-original-img').src = e.target.result;
        document.getElementById('filter-result-img').src = e.target.result;
        document.getElementById('filter-upload-zone').classList.add('hidden');
        document.getElementById('filter-workspace').classList.remove('hidden');
        buildFilterButtons();
    };
    reader.readAsDataURL(file);
}

function buildFilterButtons() {
    const filters = {
        grayscale: '🔲 Grayscale', blur: '💧 Blur', sharpen: '🔪 Sharpen',
        edge_canny: '📐 Canny Edges', edge_sobel: '📏 Sobel Edges',
        sepia: '🟤 Sepia', emboss: '🪨 Emboss', sketch: '✏️ Sketch',
        cartoon: '🎨 Cartoon', hdr: '✨ HDR', invert: '🔄 Invert',
        threshold: '⬛ Threshold', warm: '🔥 Warm', cool: '❄️ Cool',
        vintage: '📷 Vintage', denoise: '🧹 Denoise',
    };
    const container = document.getElementById('filter-buttons');
    container.innerHTML = Object.entries(filters).map(([key, label]) =>
        `<button class="filter-btn" data-filter="${key}">${label}</button>`
    ).join('');

    container.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => applyFilter(btn.dataset.filter, btn));
    });
}

async function applyFilter(filterType, btnElement) {
    if (!filterFile) return;

    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btnElement.classList.add('active');

    const formData = new FormData();
    formData.append('image', filterFile);
    formData.append('filter', filterType);

    try {
        const res = await fetch('/filter', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('filter-result-img').src =
            `data:image/jpeg;base64,${data.filtered_image}`;
        document.getElementById('filter-result-title').textContent = data.filter_name;
    } catch (err) {
        alert('Filter failed: ' + err.message);
    }
}

// ============================================================
// Compare Tab
// ============================================================
// FIX #4: ORIGINAL BUG: initCompareUploads called initUploadZone with
// 'compare-box-1' / 'compare-box-2' — the outer wrapper divs, not the
// inner .upload-zone elements. This meant:
//   1. The click/drag listeners were on the full box (including the preview
//      image), so clicking the preview after upload re-opened the file picker.
//   2. The dragover visual feedback (classList 'dragover') was applied to
//      the wrapper, not the styled .upload-zone, so the hover effect broke.
// FIX: Use querySelector('.upload-zone') inside each box to bind events
// only to the inner drop zone, leaving the preview image unaffected.
function initCompareUploads() {
    setupCompareBox('compare-box-1', 'compare-input-1', 1);
    setupCompareBox('compare-box-2', 'compare-input-2', 2);
}

function setupCompareBox(boxId, inputId, num) {
    const box = document.getElementById(boxId);
    const zone = box.querySelector('.upload-zone');  // inner zone only
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleCompareFile(file, num);
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleCompareFile(input.files[0], num);
    });
}

function handleCompareFile(file, num) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    if (num === 1) compareFile1 = file;
    else compareFile2 = file;

    const reader = new FileReader();
    reader.onload = e => {
        const previewId = `compare-preview-${num}`;
        const boxId = `compare-box-${num}`;
        const preview = document.getElementById(previewId);
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        document.querySelector(`#${boxId} .upload-zone`).classList.add('hidden');
        checkCompareReady();
    };
    reader.readAsDataURL(file);
}

function checkCompareReady() {
    document.getElementById('btn-compare').disabled = !(compareFile1 && compareFile2);
}

async function compareImages() {
    if (!compareFile1 || !compareFile2) return;

    const formData = new FormData();
    formData.append('image1', compareFile1);
    formData.append('image2', compareFile2);

    const btn = document.getElementById('btn-compare');
    btn.disabled = true;
    btn.textContent = 'Comparing...';

    try {
        const res = await fetch('/compare', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderCompareResults(data);
    } catch (err) {
        alert('Comparison failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🔀</span> Compare Images';
    }
}

function renderCompareResults(data) {
    const container = document.getElementById('compare-results');
    container.classList.remove('hidden');

    const pct = data.similarity_percentage;
    const circle = document.getElementById('score-circle');
    const circumference = 339.3;
    const offset = circumference - (pct / 100) * circumference;

    // Reset to 0 first so the CSS transition plays from start each time
    circle.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            circle.style.strokeDashoffset = offset;
        });
    });

    document.getElementById('score-value').textContent = `${pct}%`;

    const metrics = document.getElementById('compare-metrics');
    const items = [
        { label: 'Correlation', value: data.correlation },
        { label: 'MSE', value: data.mse },
        { label: 'PSNR', value: data.psnr },
    ];
    metrics.innerHTML = items.map(m => `
        <div class="stat-card">
            <div class="stat-label">${m.label}</div>
            <div class="stat-value cyan">${m.value}</div>
        </div>
    `).join('');

    document.getElementById('diff-heatmap').src =
        `data:image/jpeg;base64,${data.diff_image}`;
    container.scrollIntoView({ behavior: 'smooth' });
}