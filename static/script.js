// Fashion Detection Demo - Client Side JavaScript

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const originalImage = document.getElementById('originalImage');
const detectedImage = document.getElementById('detectedImage');
const jsonOutput = document.getElementById('jsonOutput');

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadArea.addEventListener('click', (e) => {
    if (e.target !== uploadBtn) {
        fileInput.click();
    }
});

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file upload and detection
async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }

    // Reset UI
    hideError();
    resultsSection.style.display = 'none';
    loading.style.display = 'block';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        // Send request
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Detection failed');
        }

        const result = await response.json();

        // Display results
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred during detection');
    } finally {
        loading.style.display = 'none';
    }
}

// Display detection results
function displayResults(result) {
    // Set images
    originalImage.src = result.images.original;
    detectedImage.src = result.images.annotated;

    // Format and display JSON
    const formattedDetections = result.detections.map(det => ({
        label: det.label,
        confidence: det.confidence,
        box: det.box,
        attributes: det.attributes
    }));

    jsonOutput.textContent = JSON.stringify(formattedDetections, null, 2);

    // Show results section
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = `❌ ${message}`;
    errorMessage.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Health check on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();

        if (!health.model_loaded) {
            showError('Model not loaded. Please check server logs.');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
