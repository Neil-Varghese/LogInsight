const fileList = document.getElementById('file-list');
const selectionText = document.getElementById('selection-text');
const statusText = document.getElementById('status-text');
const runButton = document.getElementById('run-btn');

let selectedFile = null;

async function loadFiles() {
  try {
    const response = await fetch('../test%20sets/manifest.json');
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

runButton.addEventListener('click', () => {
  if (!selectedFile) {
    statusText.textContent = 'Please select a test set first.';
    return;
  }

  statusText.textContent = `Run clicked for ${selectedFile.name}. Processing is not wired up yet.`;
});

loadFiles();
