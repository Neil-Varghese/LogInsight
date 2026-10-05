const fileList = document.getElementById('file-list');
const selectionText = document.getElementById('selection-text');
const statusText = document.getElementById('status-text');
const runButton = document.getElementById('run-btn');
const liveButton = document.getElementById('live-btn');
const stopButton = document.getElementById('stop-btn');
const result = document.getElementById('result');
const progressPanel = document.getElementById('progress-panel');
const progressLabel = document.getElementById('progress-label');
const progressBar = document.getElementById('progress-bar');

let selectedFile = null;
let liveJobId = null;
let summary = null;
let blocksBox = null;
const blockRows = new Map();

async function loadFiles() {
  try {
    const response = await fetch('/api/test-sets');
    if (!response.ok) {
      throw new Error('Unable to load test set list.');
    }

    const files = await response.json();
    fileList.innerHTML = '';

    files.forEach((file) => {
      const item = document.createElement('li');
      item.className = 'file-item';
      item.textContent = file.name;

      item.addEventListener('click', () => {
        selectedFile = file;
        selectionText.textContent = `Selected: ${file.name}`;
        statusText.textContent = 'Ready to run when you are.';
        result.hidden = true;
        progressPanel.hidden = true;

        document.querySelectorAll('.file-item').forEach((entry) => {
          entry.classList.remove('active');
        });
        item.classList.add('active');
      });

      fileList.appendChild(item);
    });
  } catch (error) {
    fileList.innerHTML = '<li class="file-item">No test sets found.</li>';
    selectionText.textContent = 'The test set list could not be loaded.';
    statusText.textContent = error.message;
  }
}

const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Unable to read run progress.');
    if (job.stage === 'error') throw new Error(job.error || 'Prediction failed.');
    if (job.stage === 'complete') return job.result;

    if (job.stage === 'preprocessing') {
      const progress = Number.isFinite(job.progress) ? job.progress : 0;
      progressLabel.textContent = `Preprocessing log file: ${progress.toFixed(1)}%`;
      progressBar.classList.remove('indeterminate');
      progressBar.style.width = `${progress}%`;
    } else {
      const progress = Number.isFinite(job.progress) ? job.progress : 0;
      progressLabel.textContent = `Classifying sequences: ${progress}%`;
      progressBar.classList.remove('indeterminate');
      progressBar.style.width = `${progress}%`;
    }
    await pause(500);
  }
}

// Empty the result panel: a summary line on top, flagged blocks below.
function resetResult() {
  result.replaceChildren();
  summary = document.createElement('div');
  blocksBox = document.createElement('div');
  result.append(summary, blocksBox);
  blockRows.clear();
  result.hidden = false;
}

function showCounts(counts, warning) {
  summary.innerHTML = `<strong>Final score</strong><span>Normal: ${counts.normal_logs.toLocaleString()}</span><span>Anomalous: ${counts.anomalous_logs.toLocaleString()}</span>`;
  if (warning) summary.append(` (last update skipped: ${warning})`);
}

// textContent only: explanation and log text come from an LLM and a log file, so never inject them as HTML.
function line(title, body) {
  const p = document.createElement('p');
  const b = document.createElement('strong');
  b.textContent = `${title}: `;
  p.append(b, body);
  return p;
}

async function explainBlock(jobId, blockId, button, text) {
  button.disabled = true;
  text.textContent = 'Asking the model…';
  const base = `/api/jobs/${jobId}/blocks/${encodeURIComponent(blockId)}`;
  const [reply, detail] = await Promise.all([fetch(`${base}/explain`), fetch(base)]);
  const data = await reply.json();
  const logs = detail.ok ? (await detail.json()).raw_logs : [];
  button.disabled = false;
  const e = data.explanation;
  if (!e) { text.textContent = data.error; return; }
  const checks = document.createElement('ul');
  e.suggested_checks.forEach((c) => { const li = document.createElement('li'); li.textContent = c; checks.appendChild(li); });
  const parts = [
    line('What happened', e.summary),
    line('Likely cause', `${e.likely_cause} (explanation confidence: ${e.confidence})`),
  ];
  if (e.incomplete_sequence) parts.push(line('Warning', 'this block looks incomplete, so treat the cause as a guess'));
  parts.push(line('Check next', ''), checks);
  if (logs.length) {
    const details = document.createElement('details');
    const title = document.createElement('summary');
    title.textContent = `Log lines (${logs.length})`;
    const pre = document.createElement('pre');
    pre.textContent = logs.join('\n');
    details.append(title, pre);
    parts.push(details);
  }
  text.replaceChildren(...parts);
}

// One row per block, created once and then only relabelled, so an open explanation survives live refreshes.
async function showTopBlocks(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/blocks?filter=anomalous&limit=10`);
  if (!response.ok) return;
  const { blocks } = await response.json();
  blocks.forEach((block) => {
    const labelText = `${block.block_id} (${(block.anomaly_score * 100).toFixed(1)}%, ${block.sequence_len} events) `;
    const existing = blockRows.get(block.block_id);
    if (existing) { existing.textContent = labelText; return; }
    const row = document.createElement('div');
    const label = document.createElement('span');
    label.textContent = labelText;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Explain';
    const text = document.createElement('div');
    button.addEventListener('click', () => explainBlock(jobId, block.block_id, button, text));
    row.append(label, button, text);
    blocksBox.appendChild(row);
    blockRows.set(block.block_id, label);
  });
}

function setBusy(live) {
  runButton.disabled = true;
  liveButton.disabled = true;
  stopButton.hidden = !live;
}

function setIdle() {
  runButton.disabled = false;
  liveButton.disabled = false;
  stopButton.hidden = true;
}

runButton.addEventListener('click', async () => {
  if (!selectedFile) {
    statusText.textContent = 'Please select a test set first.';
    return;
  }

  setBusy(false);
  result.hidden = true;
  progressPanel.hidden = false;
  progressLabel.textContent = 'Preprocessing log file: 0.0%';
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = '0%';
  statusText.textContent = `Running ${selectedFile.name}. This can take several minutes for large files.`;

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: selectedFile.path }),
    });
    const startData = await response.json();
    if (!response.ok) throw new Error(startData.error || 'Prediction failed.');
    const data = await waitForJob(startData.job_id);

    resetResult();
    showCounts(data);
    await showTopBlocks(startData.job_id);
    progressPanel.hidden = true;
    statusText.textContent = `Completed ${selectedFile.name}.`;
  } catch (error) {
    statusText.textContent = error.message;
    progressPanel.hidden = true;
  } finally {
    setIdle();
  }
});

// Live mode: the server re-scores the file as it grows; the page just polls for the latest.
liveButton.addEventListener('click', async () => {
  if (!selectedFile) {
    statusText.textContent = 'Please select a log file first.';
    return;
  }

  setBusy(true);
  progressPanel.hidden = true;
  try {
    const response = await fetch('/api/live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: selectedFile.path }),
    });
    const startData = await response.json();
    if (!response.ok) throw new Error(startData.error || 'Could not start watching.');
    const jobId = startData.job_id;
    liveJobId = jobId;
    resetResult();
    summary.textContent = 'Watching… waiting for the first results.';
    statusText.textContent = `Watching ${selectedFile.name} live. New blocks appear as the file grows.`;

    while (liveJobId === jobId) {
      const job = await (await fetch(`/api/jobs/${jobId}`)).json();
      if (job.stage === 'stopped') break;
      if (job.result) showCounts(job.result, job.warning);
      if (job.database) await showTopBlocks(jobId);
      await pause(3000);
    }
    statusText.textContent = `Stopped watching ${selectedFile.name}.`;
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    liveJobId = null;
    setIdle();
  }
});

stopButton.addEventListener('click', async () => {
  const jobId = liveJobId;
  liveJobId = null;
  if (jobId) await fetch(`/api/jobs/${jobId}/stop`, { method: 'POST' });
});

loadFiles();
