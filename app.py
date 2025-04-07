from flask import Flask, render_template, request, jsonify, make_response
from flask_socketio import SocketIO
import os
import logging
import time
import json
import glob
import pickle
from werkzeug.utils import secure_filename
import plotly.graph_objects as go

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'txt'}
LOG_FILE = os.path.join(os.getcwd(), "app.log")
JOBS_DIR = os.path.join(os.getcwd(), "jobs")
CACHE_DIR = os.path.join(os.getcwd(), "cache")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['JSON_SORT_KEYS'] = False  # Preserve order in JSON responses

# Set up SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60)

# Import utility functions - with fallbacks if modules are missing
try:
    from src.main import process_pmids_async
    from src.utils import init_socketio, get_job_status, JobStatus, clear_cache, get_project_root
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")


    # Fallback implementations for critical functions
    class JobStatus:
        PENDING = "pending"
        UPLOADING = "uploading"
        PROCESSING = "processing"
        VALIDATING = "validating"
        FETCHING_DATA = "fetching_data"
        ANALYZING_TEXT = "analyzing_text"
        CLUSTERING = "clustering"
        SAVING_RESULTS = "saving_results"
        COMPLETED = "completed"
        FAILED = "failed"


    def get_project_root():
        return os.getcwd()


    def init_socketio(sio):
        logger.info("Initialized SocketIO with fallback implementation")


    def get_job_status(job_id):
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if not os.path.exists(job_file):
            return None
        try:
            with open(job_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading job {job_id}: {e}")
            return None


    def clear_cache(confirm=True):
        """Fallback implementation to clear cache files"""
        import glob
        deleted = 0
        for cache_file in glob.glob(os.path.join(CACHE_DIR, "*.pkl")):
            try:
                os.remove(cache_file)
                deleted += 1
            except:
                pass
        return {"status": "success", "deleted_files": deleted}


    def process_pmids_async(pmids):
        """Fallback implementation that creates a failed job"""
        import uuid
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "pmid_processing",
            "status": "failed",
            "created_at": time.time(),
            "updated_at": time.time(),
            "params": {"pmids": pmids},
            "result": None,
            "error": "Backend processing modules not available"
        }
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        return job_id

# Initialize SocketIO
init_socketio(socketio)


# Cache Manager implementation
class CacheManager:
    def __init__(self):
        self.cache_dir = CACHE_DIR

    def get_intermediate_data(self, job_id, stage):
        """Get intermediate data for a specific job stage"""
        cache_key = f"{job_id}_{stage}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache for {cache_key}: {e}")
                return None
        return None

    def save_intermediate_data(self, job_id, stage, data):
        """Save intermediate data for a specific job stage"""
        cache_key = f"{job_id}_{stage}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.warning(f"Error saving cache for {cache_key}: {e}")
            return False


# Create a singleton cache manager
_cache_manager = CacheManager()


def get_cache_manager():
    return _cache_manager


# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_plotly_figure(viz_data):
    """Create a Plotly figure from visualization data"""
    # Group dataset points by cluster
    clusters = {}
    for d in viz_data.get("datasets", []):
        cluster = d.get("cluster")
        if cluster not in clusters:
            clusters[cluster] = {"x": [], "y": [], "text": [], "ids": [], "customdata": []}
        clusters[cluster]["x"].append(d.get("x"))
        clusters[cluster]["y"].append(d.get("y"))
        clusters[cluster]["ids"].append(d.get("id"))
        # Store dataset ID in customdata for easier access
        clusters[cluster]["customdata"].append(d.get("id"))
        pmids_text = ", ".join(d.get('pmids', []))
        hover_text = f"ID: {d.get('id')}<br>PMIDs: {pmids_text}"
        clusters[cluster]["text"].append(hover_text)

    traces = []
    for cluster, data in clusters.items():
        trace = go.Scatter(
            x=data["x"],
            y=data["y"],
            mode='markers',
            name=f"Cluster {cluster}",
            text=data["text"],
            hoverinfo='text',
            customdata=data["customdata"],  # Add customdata with dataset IDs
            marker=dict(
                size=10,
                line=dict(width=1, color='rgba(0,0,0,0.5)')  # Add border to make points more distinct
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title="GEO Dataset Clusters",
        xaxis=dict(title="Principal Component 1"),
        yaxis=dict(title="Principal Component 2"),
        hovermode='closest',
        legend=dict(title="Clusters"),
        height=600,
        clickmode='event+select'  # Enable both click events and selection
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig.to_plotly_json()


# Dataset details cache
dataset_details_cache = {}


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Upload endpoint called")

    # Validate request
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename}")

    # Get job name - make it required
    job_name = request.form.get('job_name', '').strip()
    if not job_name:
        logger.error("No job name provided")
        return jsonify({'error': 'Job name is required'}), 400

    # Sanitize job name for file system
    job_name = secure_filename(job_name)
    logger.info(f"Job name provided: {job_name}")

    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not file or not allowed_file(file.filename):
        logger.error(f"File type not allowed: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved to {file_path}")

        # Read PMIDs from file
        with open(file_path, 'r') as f:
            pmids = [line.strip() for line in f if line.strip()]

        if not pmids:
            return jsonify({'error': 'No valid PMIDs found in file'}), 400

        logger.info(f"Found {len(pmids)} PMIDs in file")

        # Start asynchronous processing
        job_id = process_pmids_async(pmids, n_clusters=None, dim_reduction='tsne')
        logger.info(f"Created job with ID: {job_id}")

        # Update job name in job file
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                job_data = json.load(f)

            job_data['name'] = job_name

            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)

        # Emit socket event for immediate feedback
        socketio.emit('status_update', {
            'job_id': job_id,
            'status': 'validating',
            'job_name': job_name,
            'timestamp': time.time()
        })

        return jsonify({
            'job_id': job_id,
            'job_name': job_name,
            'pmid_count': len(pmids),
            'message': 'Processing started'
        }), 200

    except Exception as e:
        logger.exception(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/save_visualization', methods=['GET'])
def save_visualization():
    """Save visualization data with user-provided name"""
    job_id = request.args.get('job_id')
    job_name = request.args.get('name', 'Unnamed Job')

    if not job_id:
        return jsonify({'error': 'No job ID provided'}), 400

    job_status = get_job_status(job_id)
    if not job_status or job_status.get('status') != 'completed':
        return jsonify({
            'status': 'error',
            'message': 'Job not found or not completed'
        }), 404

    # Determine visualization file path
    project_root = get_project_root()
    viz_file = os.path.join(project_root, "data", "processed", job_id, "cluster_visualization.json")

    if not os.path.exists(viz_file):
        return jsonify({
            'status': 'error',
            'message': 'Visualization data not found'
        }), 404

    try:
        # Read visualization data
        with open(viz_file, 'r') as f:
            viz_data = json.load(f)

        # Create user-friendly filename
        safe_name = secure_filename(job_name)
        output_file = os.path.join(PROCESSED_DIR, f"{safe_name}.json")

        # Prepare output data with job metadata
        output_data = {
            'job_id': job_id,
            'job_name': job_name,
            'created_at': job_status.get('created_at', time.time()),
            'visualization_data': viz_data
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        return jsonify({
            'status': 'success',
            'file_path': output_file,
            'message': f'Visualization saved as {os.path.basename(output_file)}'
        }), 200

    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saving visualization: {str(e)}'
        }), 500


@app.route('/job_status/<job_id>')
def job_status(job_id):
    # Get job status with caching
    status = get_job_status(job_id)
    if status is None:
        return jsonify({'error': 'Job not found'}), 404

    # Add cache headers for short-term caching
    response = make_response(jsonify(status))
    response.headers['Cache-Control'] = 'private, max-age=2'  # 2 seconds
    return response


@app.route('/logs')
def get_logs():
    try:
        # Get the last 100 lines of logs for better performance
        num_lines = request.args.get('lines', 100, type=int)

        with open(LOG_FILE, 'r') as f:
            logs = f.readlines()
            logs = logs[-num_lines:]  # Get last N lines

        return jsonify({'logs': ''.join(logs)}), 200
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache_route():
    """Endpoint to clear cache and processed data."""
    try:
        result = clear_cache(confirm=False)

        # Also clear the dataset details cache
        dataset_details_cache.clear()

        logger.info(f"Cache cleared: {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/visualization')
def visualization():
    """Generate and return Plotly figure for clustering results."""
    # Get job_id from query parameters
    job_id = request.args.get('job_id')

    # ETag support for caching
    etag = request.headers.get('If-None-Match', '')

    # Determine visualization file path
    project_root = get_project_root()

    if job_id:
        # Check if job exists and is completed
        job_status = get_job_status(job_id)
        if not job_status:
            return jsonify({'status': 'error', 'message': 'Job not found'}), 404

        if job_status.get('status') != 'completed':
            return jsonify({
                'status': 'pending',
                'message': f'Job is still in progress: {job_status.get("status")}'
            }), 200

        # Use job-specific visualization file
        visualization_file = os.path.join(project_root, "data", "processed", job_id, "cluster_visualization.json")
    else:
        # Default location for the most recent analysis
        visualization_file = os.path.join(project_root, "data", "processed", "cluster_visualization.json")

    if not os.path.exists(visualization_file):
        return jsonify({
            'status': 'nodata',
            'message': 'No visualization data available. Please submit a PMID list for analysis first.'
        }), 200

    # Generate ETag based on file modification time
    file_mtime = os.path.getmtime(visualization_file)
    new_etag = f'"{job_id or "main"}-{file_mtime}"'

    # If ETag matches, return 304 Not Modified
    if etag and etag == new_etag:
        return "", 304

    try:
        with open(visualization_file, 'r') as f:
            viz_data = json.load(f)

        # Create the plot
        fig_json = create_plotly_figure(viz_data)

        # Gather additional statistics
        total_pmids = 0
        pmid_set = set()
        for cluster_info in viz_data.get("cluster_info", {}).values():
            pmid_set.update(cluster_info.get("pmids", []))
        total_pmids = len(pmid_set)

        # Count clusters and datasets
        datasets = viz_data.get("datasets", [])
        clusters = {}
        for dataset in datasets:
            cluster = dataset.get("cluster")
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(dataset.get("id"))

        result = {
            "status": "success",
            "figure": fig_json,
            "summary": {
                "clusters": len(clusters),
                "datasets": len(datasets),
                "pmids": total_pmids
            }
        }

        # Set response with cache headers
        response = make_response(jsonify(result))
        response.headers['Cache-Control'] = 'private, max-age=60'  # 1 minute cache
        response.headers['ETag'] = new_etag
        return response

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error generating visualization: {str(e)}'
        }), 500


@app.route('/get_jobs')
def get_jobs():
    """Get a list of completed jobs available for visualization."""
    try:
        # Get all job files
        job_files = glob.glob(os.path.join(JOBS_DIR, "*.json"))

        available_jobs = []
        for job_file in job_files:
            job_id = os.path.basename(job_file).split('.')[0]
            job_status = get_job_status(job_id)

            if job_status and job_status.get('status') == 'completed':
                # Check if visualization data exists
                viz_file = os.path.join(PROCESSED_DIR, job_id, "cluster_visualization.json")
                if os.path.exists(viz_file):
                    # Add job creation time for sorting and job name
                    job_info = {
                        'job_id': job_id,
                        'created_at': job_status.get('created_at', 0),
                        'name': job_status.get('name', ''),
                        'dataset_count': 0  # Default value
                    }

                    # Try to get dataset count
                    try:
                        with open(viz_file, 'r') as f:
                            viz_data = json.load(f)
                            job_info['dataset_count'] = len(viz_data.get('datasets', []))
                    except Exception as e:
                        logger.warning(f"Error reading visualization data for job {job_id}: {e}")

                    available_jobs.append(job_info)

        # Sort by creation time (most recent first)
        available_jobs.sort(key=lambda x: x['created_at'], reverse=True)

        # Set cache headers
        response = make_response(jsonify({'jobs': available_jobs}))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/dataset_details')
def dataset_details():
    dataset_id = request.args.get('id')
    if not dataset_id:
        logger.error("Dataset ID not provided in request")
        return jsonify({'status': 'error', 'message': 'Dataset ID is required'}), 400

    logger.info(f"Looking up details for dataset ID: {dataset_id}")

    # Check cache first
    cache_key = f"dataset_{dataset_id}"
    if cache_key in dataset_details_cache:
        logger.info(f"Returning cached details for dataset {dataset_id}")
        return jsonify(dataset_details_cache[cache_key])

    # First, look in named job files
    named_files = glob.glob(os.path.join(PROCESSED_DIR, "*.json"))
    for file_path in named_files:
        # Check if this is a saved job file that points to a specific job
        try:
            with open(file_path, 'r') as f:
                job_data = json.load(f)
                if 'job_id' in job_data:
                    job_id = job_data['job_id']
                    # Look for detailed info in the job's geo_dataset_info.json
                    dataset_info = get_dataset_from_geo_info(job_id, dataset_id)
                    if dataset_info:
                        logger.info(f"Found dataset {dataset_id} in job {job_id}")
                        response = {'status': 'success', 'dataset': dataset_info}
                        dataset_details_cache[cache_key] = response
                        return jsonify(response)
        except Exception as e:
            logger.error(f"Error checking named job file {file_path}: {e}")

    # Next, check all job directories
    job_dirs = [d for d in glob.glob(os.path.join(PROCESSED_DIR, "*")) if os.path.isdir(d)]
    for job_dir in job_dirs:
        job_id = os.path.basename(job_dir)
        dataset_info = get_dataset_from_geo_info(job_id, dataset_id)
        if dataset_info:
            logger.info(f"Found dataset {dataset_id} in job directory {job_id}")
            response = {'status': 'success', 'dataset': dataset_info}
            dataset_details_cache[cache_key] = response
            return jsonify(response)

    # If not found anywhere, return error
    logger.error(f"Dataset {dataset_id} not found in any geo_dataset_info.json files")
    return jsonify({'status': 'error', 'message': f'Dataset with ID {dataset_id} not found'}), 404


def get_dataset_from_geo_info(job_id, dataset_id):
    """Get dataset info from job's geo_dataset_info.json file"""
    info_file = os.path.join(PROCESSED_DIR, job_id, "geo_dataset_info.json")

    if not os.path.exists(info_file):
        logger.warning(f"geo_dataset_info.json not found for job {job_id}")
        return None

    try:
        with open(info_file, 'r') as f:
            info_data = json.load(f)

        if "datasets" in info_data and dataset_id in info_data["datasets"]:
            dataset = info_data["datasets"][dataset_id]

            # Add the dataset ID to the result
            dataset_info = {
                'id': dataset_id,
                'title': dataset.get('title', 'No title available'),
                'experiment_type': dataset.get('experiment_type', 'Not specified'),
                'summary': dataset.get('summary', 'No summary available'),
                'organism': dataset.get('organism', 'Not specified'),
                'overall_design': dataset.get('overall_design', 'Not specified')
            }

            # Try to get associated PMIDs
            dataset_info['pmids'] = dataset.get('associated_pmids', [])

            # Try to get cluster information from visualization file
            viz_file = os.path.join(PROCESSED_DIR, job_id, "cluster_visualization.json")
            if os.path.exists(viz_file):
                try:
                    with open(viz_file, 'r') as f:
                        viz_data = json.load(f)

                    for d in viz_data.get("datasets", []):
                        if d.get("id") == dataset_id:
                            dataset_info['cluster'] = d.get('cluster', 0)
                            if not dataset_info.get('pmids') and 'pmids' in d:
                                dataset_info['pmids'] = d.get('pmids', [])
                            break
                except Exception as e:
                    logger.error(f"Error reading visualization file for job {job_id}: {e}")

            return dataset_info
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {info_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error reading geo_dataset_info.json for job {job_id}: {str(e)}")

    return None


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_status_update')
def handle_status_update_request(data):
    """Handle client request for status update"""
    job_id = data.get('job_id')
    if job_id:
        status = get_job_status(job_id)
        if status:
            socketio.emit('status_update', {
                'job_id': job_id,
                'status': status.get('status'),
                'error': status.get('error'),
                'timestamp': time.time()
            }, room=request.sid)
        else:
            socketio.emit('status_update', {
                'job_id': job_id,
                'status': 'not_found',
                'timestamp': time.time()
            }, room=request.sid)


# Run the application
if __name__ == '__main__':
    logger.info("Starting GEO Dataset Clustering Application")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)
