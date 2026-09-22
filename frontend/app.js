const fileList = document.getElementById('file-list');
const selectionText = document.getElementById('selection-text');
const statusText = document.getElementById('status-text');
const runButton = document.getElementById('run-btn');
const result = document.getElementById('result');
const progressPanel = document.getElementById('progress-panel');
const progressLabel = document.getElementById('progress-label');
const progressBar = document.getElementById('progress-bar');

let selectedFile = null;

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

runButton.addEventListener('click', async () => {
  if (!selectedFile) {
    statusText.textContent = 'Please select a test set first.';
    return;
  }

  runButton.disabled = true;
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

    result.innerHTML = `<strong>Final score</strong><span>Normal: ${data.normal_logs.toLocaleString()}</span><span>Anomalous: ${data.anomalous_logs.toLocaleString()}</span>`;
    result.hidden = false;
    progressPanel.hidden = true;
    statusText.textContent = `Completed ${selectedFile.name}.`;
  } catch (error) {
    statusText.textContent = error.message;
    progressPanel.hidden = true;
  } finally {
    runButton.disabled = false;
  }
});

loadFiles();
