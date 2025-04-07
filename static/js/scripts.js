// Initialize global variables
let socket;
let activeJobId = null;
let jobStartTime = null;
let lastKnownStatus = null;

// Initialize application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Initialize Socket.IO connection
    initSocketConnection();

    // Set up event listeners
    setupEventListeners();

    // Initial data load
    fetchLogs();

    // Start the job timer if there's an active job
    startJobTimer();
});

// Initialize Socket.IO connection
function initSocketConnection() {
    console.log("Attempting to connect to Socket.IO...");
    socket = io();

    socket.on('connect', function () {
        console.log("Socket.IO connected successfully");
    });

    socket.on('connect_error', function (error) {
        console.error("Socket.IO connection error:", error);
    });

    socket.on('disconnect', function () {
        console.warn("Socket.IO disconnected");
    });

    // Listen for status updates pushed by the server
    socket.on('status_update', function (data) {
        console.log("Received status update:", data);
        handleStatusUpdate(data);
    });
}

// Set up all event listeners
function setupEventListeners() {
    // File upload form submission
    document.getElementById('upload-form').addEventListener('submit', handleFormSubmit);

    // PMID File input change (to display filename)
    document.getElementById('file').addEventListener('change', function (e) {
        const fileName = e.target.files[0] ? e.target.files[0].name : 'No file selected';
        document.querySelector('.file-name').textContent = fileName;
    });

    // Job file input change (to display filename)
    document.getElementById('job-file').addEventListener('change', function (e) {
        const fileName = e.target.files[0] ? e.target.files[0].name : 'No file selected';
        document.querySelector('.job-file-name').textContent = fileName;
    });

    // Load job visualization button
    document.getElementById('load-job-btn').addEventListener('click', function () {
        loadSelectedJobFile();
    });

    // Clear cache button
    document.getElementById('clear-cache-btn').addEventListener('click', handleClearCache);

    // Toggle logs visibility
    document.getElementById('toggle-logs-btn').addEventListener('click', function () {
        const logs = document.getElementById('logs');
        const computedStyle = window.getComputedStyle(logs);
        if (computedStyle.display === 'none') {
            logs.style.display = 'block';
            this.textContent = 'Hide Logs';
        } else {
            logs.style.display = 'none';
            this.textContent = 'Show Logs';
        }
    });
}

// Handle form submission
function handleFormSubmit(e) {
    e.preventDefault();
    const formData = new FormData(this);

    // Validate job name
    const jobName = document.getElementById('job_name').value.trim();
    if (!jobName) {
        showError('Job name is required');
        return;
    }

    console.log("Form submission started");
    console.log("File selected:", document.getElementById('file').files[0]?.name || "No file");
    console.log("Job name:", jobName);

    // Clear previous status
    resetInterface();
    updateTimelineStatus('validation', 'active');

    // Submit form data
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            console.log("Server response status:", response.status);
            if (!response.ok) {
                return response.json().then(data => {
                    console.error("Error response:", data);
                    throw new Error(data.error || 'Unknown error');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("Success response:", data);

            // Set active job ID
            activeJobId = data.job_id;

            // Update job info display
            document.getElementById('current-job-id').textContent = `Job ID: ${activeJobId.substring(0, 8)}... (${jobName})`;

            // Reset job timer
            jobStartTime = new Date();
            startJobTimer();

            // Begin polling job status
            pollJobStatus(activeJobId);

            // Update UI
            updateTimelineStatus('validation', 'completed');
            updateTimelineStatus('fetching_data', 'active');

            // Update PMID count stat
            document.querySelector('#pmid-count .stat-value').textContent = data.pmid_count || '0';
        })
        .catch(error => {
            console.error("Fetch error:", error);
            showError(`Upload failed: ${error.message}`);
            updateTimelineStatus('validation', 'error');
        });
}

// Reset interface to initial state
function resetInterface() {
    // Reset timeline
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.classList.remove('active', 'completed', 'error');
    });

    // Reset stats
    document.querySelector('#pmid-count .stat-value').textContent = '0';
    document.querySelector('#dataset-count .stat-value').textContent = '0';
    document.querySelector('#cluster-count .stat-value').textContent = '0';

    // Clear visualization area
    document.getElementById('viz-status').textContent = '';
    document.getElementById('viz-status').className = 'status-message';
    document.getElementById('cluster-summary').textContent = '';
    document.getElementById('plotly-div').innerHTML = '';

    // Reset workflow visualization
    document.querySelector('.stage-visualization').innerHTML = '<div class="waiting-message">Waiting for job to start...</div>';
    document.querySelector('.stage-description').textContent = '';
}

// Show error message
function showError(message) {
    const statusDiv = document.createElement('div');
    statusDiv.className = 'status-message error';
    statusDiv.textContent = message;

    const vizStatus = document.getElementById('viz-status');
    vizStatus.innerHTML = '';
    vizStatus.appendChild(statusDiv);
}

// Handle clear cache button click
function handleClearCache() {
    if (confirm('Are you sure you want to clear the cache? This will remove all processed data.')) {
        fetch('/clear_cache', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                alert(`Cache cleared: ${data.deleted_files} files removed`);
            })
            .catch(error => {
                showError(`Error clearing cache: ${error.message}`);
            });
    }
}

// Update timeline status for a specific stage
function updateTimelineStatus(stage, status) {
    const timelineItems = document.querySelectorAll('.timeline-item');
    let reachedStage = false;

    timelineItems.forEach(item => {
        const itemStage = item.getAttribute('data-stage');

        if (itemStage === stage) {
            reachedStage = true;
            item.classList.remove('active', 'completed', 'error');
            item.classList.add(status);
        } else if (!reachedStage) {
            item.classList.remove('active', 'error');
            item.classList.add('completed');
        } else {
            item.classList.remove('active', 'completed', 'error');
        }
    });

    // Update workflow visualization based on stage
    updateWorkflowVisualization(stage, status);
}

// Update workflow visualization based on stage
function updateWorkflowVisualization(stage, status) {
    const stageViz = document.querySelector('.stage-visualization');
    const stageDesc = document.querySelector('.stage-description');

    switch (stage) {
        case 'validation':
            stageViz.innerHTML = `
                <div class="stage-icon validation-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
                <div class="stage-label">Validating PMIDs</div>
            `;
            stageDesc.textContent = 'Checking PMID format and preparing for processing...';
            break;

        case 'fetching_data':
            stageViz.innerHTML = `
                <div class="stage-icon fetching-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                    </svg>
                </div>
                <div class="stage-label">Retrieving GEO Datasets</div>
            `;
            stageDesc.textContent = 'Connecting to NCBI databases to retrieve GEO datasets linked to provided PMIDs...';
            break;

        case 'analyzing_text':
            stageViz.innerHTML = `
                <div class="stage-icon analyzing-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="8" y1="12" x2="16" y2="12"></line>
                        <line x1="8" y1="16" x2="16" y2="16"></line>
                        <line x1="8" y1="8" x2="16" y2="8"></line>
                    </svg>
                </div>
                <div class="stage-label">Analyzing Dataset Text</div>
            `;
            stageDesc.textContent = 'Processing dataset descriptions with TF-IDF vectorization to prepare for clustering...';
            break;

        case 'clustering':
            stageViz.innerHTML = `
                <div class="stage-icon clustering-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="18" cy="18" r="3"></circle>
                        <circle cx="6" cy="6" r="3"></circle>
                        <circle cx="6" cy="18" r="3"></circle>
                        <line x1="6" y1="9" x2="6" y2="15"></line>
                        <line x1="9" y1="6" x2="15" y2="6"></line>
                        <line x1="9" y1="18" x2="15" y2="18"></line>
                    </svg>
                </div>
                <div class="stage-label">Clustering Similar Datasets</div>
            `;
            stageDesc.textContent = 'Applying hierarchical clustering to group similar datasets based on text similarity...';
            break;

        case 'completed':
            stageViz.innerHTML = `
                <div class="stage-icon completed-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
                <div class="stage-label">Analysis Complete</div>
            `;
            stageDesc.textContent = 'The analysis is now complete. You can download the results for future reference.';
            break;

        case 'failed':
            stageViz.innerHTML = `
                <div class="stage-icon error-icon">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </div>
                <div class="stage-label">Analysis Failed</div>
            `;
            stageDesc.textContent = 'The analysis encountered an error. Please check the logs for details.';
            break;
    }
}

// Handle status updates from Socket.IO
// Handle status updates from Socket.IO
function handleStatusUpdate(data) {
    console.log("Status update received:", data);

    // Set active job ID if not already set
    if (!activeJobId && data.job_id) {
        activeJobId = data.job_id;

        // Update job info display with job ID and name if available
        const jobName = data.job_name || document.getElementById('job_name').value || '';
        if (jobName) {
            document.getElementById('current-job-id').textContent = `Job ID: ${activeJobId.substring(0, 8)}... (${jobName})`;
        } else {
            document.getElementById('current-job-id').textContent = `Job ID: ${activeJobId.substring(0, 8)}...`;
        }

        // Initialize job timer if not already started
        if (!jobStartTime) {
            jobStartTime = new Date();
            startJobTimer();
        }
    }

    // Handle different status updates
    if (data.status === 'validating') {
        updateTimelineStatus('validation', 'active');

        // Fetch job details to update PMID count
        if (data.job_id) {
            fetch(`/job_status/${data.job_id}`)
                .then(response => response.json())
                .then(jobData => {
                    if (jobData.params && jobData.params.pmids) {
                        document.querySelector('#pmid-count .stat-value').textContent =
                            jobData.params.pmids.length || '0';
                    }
                })
                .catch(error => console.error('Error fetching job details:', error));
        }
    } else if (data.status === 'fetching_data') {
        updateTimelineStatus('validation', 'completed');
        updateTimelineStatus('fetching_data', 'active');
    } else if (data.status === 'analyzing_text') {
        updateTimelineStatus('validation', 'completed');
        updateTimelineStatus('fetching_data', 'completed');
        updateTimelineStatus('analyzing_text', 'active');

        // Update dataset count if available
        if (data.job_id) {
            fetch(`/job_status/${data.job_id}`)
                .then(response => response.json())
                .then(jobData => {
                    if (jobData.result && jobData.result.analysis_data) {
                        document.querySelector('#dataset-count .stat-value').textContent =
                            jobData.result.analysis_data.dataset_count || '0';
                    }
                })
                .catch(error => console.error('Error fetching job details:', error));
        }
    } else if (data.status === 'clustering') {
        updateTimelineStatus('validation', 'completed');
        updateTimelineStatus('fetching_data', 'completed');
        updateTimelineStatus('analyzing_text', 'completed');
        updateTimelineStatus('clustering', 'active');
    } // Update the 'completed' section in handleStatusUpdate function
    else if (data.status === 'completed') {
        // Update all timeline stages to completed
        updateTimelineStatus('validation', 'completed');
        updateTimelineStatus('fetching_data', 'completed');
        updateTimelineStatus('analyzing_text', 'completed');
        updateTimelineStatus('clustering', 'completed');
        updateTimelineStatus('completed', 'completed');

        // Display success message (simple version)
        document.getElementById('viz-status').innerHTML =
            '<p class="status-message success">Job completed successfully!</p>';

        // Get job name for saving visualization
        const jobName = data.job_name || document.getElementById('job_name').value || 'Unnamed Job';

        // Save visualization - important to keep this functionality
        saveVisualization(data.job_id, jobName);

        // Update job name in display
        if (jobName) {
            document.getElementById('current-job-id').textContent =
                `Job: ${jobName} (${data.job_id.substring(0, 8)}...)`;
        }

        // Stop the job timer if running
        if (jobStartTime) {
            const now = new Date();
            const elapsed = now - jobStartTime;
            const seconds = Math.floor(elapsed / 1000) % 60;
            const minutes = Math.floor(elapsed / 60000) % 60;
            const hours = Math.floor(elapsed / 3600000);
            const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            document.getElementById('job-time').textContent = `Total: ${timeString}`;
        }
    } else if (data.status === 'failed') {
        // Update timeline to show error
        updateTimelineStatus('failed', 'error');

        // Display error message
        const errorMessage = data.error || 'Unknown error occurred';
        document.getElementById('viz-status').innerHTML = `<p class="status-message error">Job failed: ${errorMessage}</p>`;

        // Add option to download error log
        const downloadErrButton = document.createElement('button');
        downloadErrButton.className = 'secondary-btn';
        downloadErrButton.textContent = 'Download Error Log';
        downloadErrButton.style.marginTop = '10px';
        downloadErrButton.onclick = function () {
            const errorData = {
                job_id: data.job_id,
                timestamp: new Date().toISOString(),
                error: errorMessage,
                status: data.status
            };

            const blob = new Blob([JSON.stringify(errorData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `error-log-${data.job_id.substring(0, 8)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };

        document.getElementById('viz-status').appendChild(downloadErrButton);
    }

    // Store last known status
    lastKnownStatus = data.status;
}

function downloadJobFile(jobId) {
    fetch(`/job_status/${jobId}`)
        .then(response => response.json())
        .then(jobData => {
            // Create a blob with the job data
            const blob = new Blob([JSON.stringify(jobData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);

            // Create a download link and click it
            const a = document.createElement('a');
            a.href = url;
            a.download = `job-${jobId.substring(0, 8)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error("Error downloading job file:", error);
            alert("Error downloading job file: " + error.message);
        });
}

// Save visualization to a named file
// Modified saveVisualization function
function saveVisualization(jobId, jobName) {
    console.log(`Saving visualization for job ${jobId} with name "${jobName}"`);

    fetch(`/save_visualization?job_id=${jobId}&name=${encodeURIComponent(jobName)}`)
        .then(response => {
            console.log("Save response status:", response.status);
            return response.json();
        })
        .then(saveResult => {
            console.log("Save result:", saveResult);

            if (saveResult.status === 'success') {
                // Clear any existing success messages to avoid duplication
                const vizStatus = document.getElementById('viz-status');
                if (vizStatus.innerHTML.includes('Visualization saved')) {
                    // Don't add another message if one already exists
                } else {
                    vizStatus.innerHTML +=
                        `<p class="status-message success">Visualization saved successfully</p>`;
                }

                // Always load the visualization
                loadCompletedVisualization(jobId);
            } else {
                document.getElementById('viz-status').innerHTML +=
                    `<p class="status-message error">Error saving visualization: ${saveResult.message}</p>`;
            }
        })
        .catch(error => {
            console.error('Error saving visualization:', error);
            document.getElementById('viz-status').innerHTML +=
                `<p class="status-message error">Error saving visualization: ${error.message}</p>`;
        });
}

function loadCompletedVisualization(jobId) {
    fetch(`/visualization?job_id=${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                Plotly.newPlot('plotly-div', data.figure.data, data.figure.layout);

                document.querySelector('#cluster-count .stat-value').textContent = data.summary.clusters || '0';
                document.querySelector('#dataset-count .stat-value').textContent = data.summary.datasets || '0';

                document.getElementById('cluster-summary').innerHTML =
                    `<strong>Summary:</strong> ${data.summary.clusters} clusters containing ${data.summary.datasets} datasets`;

                setupPlotlyClickEvents('plotly-div');
            }
        })
        .catch(error => {
            console.error("Error loading visualization:", error);
        });
}

// Poll job status periodically until completed
function pollJobStatus(jobId) {
    const interval = setInterval(function () {
        fetch(`/job_status/${jobId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error fetching job status');
                }
                return response.json();
            })
            .then(data => {
                if (data.params && data.params.pmids) {
                    document.querySelector('#pmid-count .stat-value').textContent =
                        data.params.pmids.length || '0';
                }
                if (data.result && typeof data.result === 'object') {
                    if (data.result.analysis_data) {
                        if (data.result.analysis_data.dataset_count) {
                            document.querySelector('#dataset-count .stat-value').textContent =
                                data.result.analysis_data.dataset_count || '0';
                        }
                        if (data.result.visualization_data && data.result.visualization_data.cluster_info) {
                            document.querySelector('#cluster-count .stat-value').textContent =
                                Object.keys(data.result.visualization_data.cluster_info).length || '0';
                        }
                    }
                }

                if (data.status !== lastKnownStatus) {
                    lastKnownStatus = data.status;
                }

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(interval);
                }
            })
    }, 3000);
}

// Start job timer to show elapsed time
function startJobTimer() {
    if (!jobStartTime || !activeJobId) return;

    const timerInterval = setInterval(function () {
        if (!jobStartTime) {
            clearInterval(timerInterval);
            return;
        }

        const now = new Date();
        const elapsed = now - jobStartTime;
        const seconds = Math.floor(elapsed / 1000) % 60;
        const minutes = Math.floor(elapsed / 60000) % 60;
        const hours = Math.floor(elapsed / 3600000);

        const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        document.getElementById('job-time').textContent = `Elapsed: ${timeString}`;
    }, 1000);
}

// Load selected job file
function loadSelectedJobFile() {
    const fileInput = document.getElementById('job-file');
    if (!fileInput.files || fileInput.files.length === 0) {
        document.getElementById('viz-status').innerHTML =
            '<p class="status-message warning">Please select a job file to load.</p>';
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    document.getElementById('viz-status').innerHTML =
        '<p class="status-message">Loading job file...</p>';

    reader.onload = function (e) {
        try {
            const jobData = JSON.parse(e.target.result);
            let vizData = null;

            // Check for different possible data structures
            if (jobData.visualization_data) {
                // This is a saved visualization file from data/processed/
                vizData = jobData.visualization_data;
                console.log("Found visualization_data directly in the file");
            } else if (jobData.result && jobData.result.visualization_data) {
                // This is from the jobs directory, visualization data is nested in result
                vizData = jobData.result.visualization_data;
                console.log("Found visualization_data in result object");
            }

            if (vizData) {
                renderVisualization(vizData);

                // Update stats
                document.querySelector('#dataset-count .stat-value').textContent =
                    vizData.datasets ? vizData.datasets.length : '0';
                document.querySelector('#cluster-count .stat-value').textContent =
                    (vizData.cluster_info) ? Object.keys(vizData.cluster_info).length : '0';

                // Update job name if available
                if (jobData.name) {
                    document.getElementById('current-job-id').textContent =
                        `Job: ${jobData.name}`;
                } else if (jobData.id) {
                    document.getElementById('current-job-id').textContent =
                        `Job ID: ${jobData.id.substring(0, 8)}...`;
                }

                document.getElementById('viz-status').innerHTML =
                    '<p class="status-message success">Job file loaded successfully!</p>';

                document.getElementById('cluster-summary').innerHTML =
                    `<strong>Summary:</strong> ${vizData.cluster_info ? Object.keys(vizData.cluster_info).length : '0'} clusters containing ${vizData.datasets ? vizData.datasets.length : '0'} datasets`;

            } else {
                throw new Error("No visualization data found in the file. The job may not be complete or the file format is incorrect.");
            }

        } catch (error) {
            console.error("Error processing job file:", error);
            document.getElementById('viz-status').innerHTML =
                `<p class="status-message error">Error loading job file: ${error.message}</p>`;
            document.getElementById('plotly-div').innerHTML = '';
            document.getElementById('cluster-summary').textContent = '';
        }
    };

    reader.onerror = function () {
        document.getElementById('viz-status').innerHTML =
            '<p class="status-message error">Error reading the file.</p>';
    };

    reader.readAsText(file);
}

// Render visualization from data
function renderVisualization(vizData) {
    // Store visualization data globally for access in click events
    window.visualizationData = vizData;

    // Group dataset points by cluster (same as before)
    const clusters = {};
    for (const d of vizData.datasets) {
        const cluster = d.cluster;
        if (!clusters[cluster]) {
            clusters[cluster] = {x: [], y: [], text: [], ids: [], customdata: []};
        }
        clusters[cluster].x.push(d.x);
        clusters[cluster].y.push(d.y);
        clusters[cluster].ids.push(d.id);
        clusters[cluster].customdata.push(d.id);  // Add customdata
        const pmidsText = (d.pmids || []).join(', ');
        const hoverText = `ID: ${d.id}<br>PMIDs: ${pmidsText}`;
        clusters[cluster].text.push(hoverText);
    }

    // Store clusters data globally for access in click events
    window.clusters = clusters;

    const traces = [];
    for (const [cluster, data] of Object.entries(clusters)) {
        traces.push({
            x: data.x,
            y: data.y,
            mode: 'markers',
            name: `Cluster ${cluster}`,
            text: data.text,
            hoverinfo: 'text',
            customdata: data.customdata,  // Include customdata in trace
            marker: {
                size: 10,
                line: {
                    width: 1,
                    color: 'rgba(0,0,0,0.5)'
                }
            }
        });
    }

    const layout = {
        title: "GEO Dataset Clusters",
        xaxis: {title: "Principal Component 1"},
        yaxis: {title: "Principal Component 2"},
        hovermode: 'closest',
        legend: {title: "Clusters"},
        height: 600,
        clickmode: 'event+select'
    };

    // Clear previous plot
    document.getElementById('plotly-div').innerHTML = '';

    // Create the new plot
    Plotly.newPlot('plotly-div', traces, layout, {responsive: true});

    console.log("Visualization rendered, attaching click events");

    // Ensure events are attached after the plot is created with a small delay
    setTimeout(() => {
        setupPlotlyClickEvents('plotly-div');
    }, 100);
}

// Periodically fetch logs from the server
function fetchLogs() {
    setInterval(function () {
        fetch('/logs')
            .then(response => response.json())
            .then(data => {
                if (data.logs) {
                    const logLines = data.logs.split('\n');
                    const last15Lines = logLines.slice(-15).join('\n');
                    document.getElementById('logs').textContent = last15Lines;
                }
            })
            .catch(error => {
                console.error('Error fetching logs:', error);
            });
    }, 5000);
}

function setupPlotlyClickEvents(divId) {
    console.log("Setting up click events for", divId);
    const plotlyDiv = document.getElementById(divId);

    if (!plotlyDiv) {
        console.error("Plotly div not found:", divId);
        return;
    }

    // Clear existing click events to avoid duplicates
    if (plotlyDiv._prevOnClick) {
        plotlyDiv.removeEventListener('plotly_click', plotlyDiv._prevOnClick);
    }

    // Create and save new handler
    const clickHandler = function (data) {
        console.log("Plotly click detected!", data);

        if (!data || !data.points || data.points.length === 0) {
            console.warn("Click event had no points data");
            return;
        }

        const point = data.points[0];
        console.log("Point data:", point);

        // Extract dataset ID from point data
        let datasetId = null;

        // Try different methods to extract the dataset ID
        if (point.customdata !== undefined) {
            datasetId = point.customdata;
            console.log("Found dataset ID in customdata:", datasetId);
        } else if (point.text && typeof point.text === 'string') {
            const match = point.text.match(/ID: ([^<\s]+)/);
            if (match && match[1]) {
                datasetId = match[1];
                console.log("Extracted dataset ID from text:", datasetId);
            }
        }

        if (!datasetId) {
            console.error("Could not extract dataset ID from point", point);
            document.getElementById('dataset-content').innerHTML =
                "<p>Could not identify the selected dataset. Please try again.</p>";
            return;
        }

        // Show loading indicator
        document.getElementById('dataset-content').innerHTML =
            '<p class="loading">Loading dataset details...</p>';

        // Fetch and display dataset details
        fetchDatasetDetails(datasetId);
    };

    plotlyDiv._prevOnClick = clickHandler;
    plotlyDiv.on('plotly_click', clickHandler);

    console.log("Click events setup complete for", divId);
}

function fetchDatasetDetails(datasetId) {
    console.log("Fetching details for dataset:", datasetId);

    fetch(`/dataset_details?id=${datasetId}`)
        .then(response => {
            console.log("Response status:", response.status);
            return response.json().then(data => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}: ${data.message || response.statusText}`);
                }
                return data;
            });
        })
        .then(data => {
            console.log("Dataset details response:", data);
            if (data.status === 'success') {
                displayDatasetDetails(data.dataset);
            } else {
                document.getElementById('dataset-content').innerHTML =
                    `<p>Could not load details for dataset ${datasetId}: ${data.message || 'Unknown error'}</p>`;
            }
        })
        .catch(error => {
            console.error("Error fetching dataset details:", error);
            document.getElementById('dataset-content').innerHTML = `
                <div class="error-message">
                    <p>Error loading dataset details: ${error.message}</p>
                    <p>Dataset ID: ${datasetId}</p>
                </div>
            `;
        });
}

function displayDatasetDetails(dataset) {
    let detailsHtml = `
        <div class="dataset-header">
            <h4>${dataset.title || 'No title available'}</h4>
            <div class="dataset-id">ID: ${dataset.id}</div>
        </div>
        <div class="dataset-metadata">
            <div class="metadata-item">
                <strong>Organism:</strong> ${dataset.organism || 'Not specified'}
            </div>
            <div class="metadata-item">
                <strong>Experiment Type:</strong> ${dataset.experiment_type || 'Not specified'}
            </div>
            <div class="metadata-item">
                <strong>PMIDs:</strong> ${dataset.pmids?.join(', ') || 'None'}
            </div>
        </div>
    `;

    if (dataset.summary) {
        detailsHtml += `
            <div class="dataset-section">
                <h5>Summary</h5>
                <p>${dataset.summary}</p>
            </div>
        `;
    }

    if (dataset.overall_design) {
        detailsHtml += `
            <div class="dataset-section">
                <h5>Overall Design</h5>
                <p>${dataset.overall_design}</p>
            </div>
        `;
    }

    document.getElementById('dataset-content').innerHTML = detailsHtml;
}