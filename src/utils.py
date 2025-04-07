"""
Utility functions for the GEO dataset clustering application
"""

from typing import List, Dict, Any, Optional
import uuid
from enum import Enum
import shutil
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

socketio = None  # Will be initialized in app.py


def init_socketio(sio):
    """Initialize the SocketIO instance."""
    global socketio
    socketio = sio


# Global executor for background tasks
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_jobs = {}


def run_async_job(job_id: str, func, *args, **kwargs):
    """Run a function asynchronously and track its status"""

    def _wrapped_func(job_id, func, *args, **kwargs):
        try:
            # Update job status to processing
            update_job_status(job_id, JobStatus.PROCESSING)
            # Run the actual function
            result = func(*args, **kwargs)
            # Update job status to completed with result
            update_job_status(job_id, JobStatus.COMPLETED, result=result)
            return result
        except Exception as e:
            # Update job status to failed with error
            error_msg = str(e)
            logging.error(f"Job {job_id} failed: {error_msg}")
            update_job_status(job_id, JobStatus.FAILED, error=error_msg)
            raise

    # Submit the wrapped function to the executor
    future = _executor.submit(_wrapped_func, job_id, func, *args, **kwargs)
    _jobs[job_id] = future
    return job_id


def cancel_job(job_id: str) -> bool:
    """Cancel a running job if possible"""
    if job_id in _jobs:
        future = _jobs[job_id]
        if not future.done():
            cancelled = future.cancel()
            if cancelled:
                update_job_status(job_id, JobStatus.FAILED, error="Job cancelled")
            return cancelled
    return False


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the result of a completed job"""
    status = get_job_status(job_id)
    if not status:
        return None
    if status.get("status") != JobStatus.COMPLETED.value:
        return None
    return status.get("result")


def get_visualization_data(job_id: str) -> Optional[Dict[str, Any]]:
    """Extract visualization data from job results"""
    result = get_job_result(job_id)
    if not result:
        return None
    return result.get("visualization_data")


class JobStatus(Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"  # New state added for processing
    VALIDATING = "validating"
    FETCHING_DATA = "fetching_data"
    ANALYZING_TEXT = "analyzing_text"
    CLUSTERING = "clustering"
    SAVING_RESULTS = "saving_results"
    COMPLETED = "completed"
    FAILED = "failed"


def get_jobs_dir():
    jobs_dir = os.path.join(os.getcwd(), "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    return jobs_dir


def create_job(job_type: str, params: Dict[str, Any] = None, job_name: str = None) -> str:
    """Create a new job and return its ID"""
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "type": job_type,
        "name": job_name or "",
        "status": JobStatus.PENDING.value,
        "created_at": time.time(),
        "updated_at": time.time(),
        "params": params or {},
        "result": None,
        "error": None
    }
    job_file = os.path.join(get_jobs_dir(), f"{job_id}.json")
    with open(job_file, "w") as f:
        json.dump(job_data, f, indent=2)
    return job_id


def update_job_status(job_id: str, status: JobStatus, result=None, error: str = None) -> bool:
    """Update job status with safe file handling to prevent corruption."""
    job_file = os.path.join(get_jobs_dir(), f"{job_id}.json")

    if not os.path.exists(job_file):
        return False

    try:
        # Read existing job data
        job_data = None
        try:
            with open(job_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)
        except json.JSONDecodeError:
            # Create a minimal structure if parsing fails
            job_data = {
                "id": job_id,
                "status": "pending",
                "created_at": time.time(),
                "updated_at": time.time(),
                "type": "pmid_processing",
                "params": {},
                "result": None,
                "error": None
            }
            logging.warning(f"Created new job data structure for {job_id} due to parsing error")

        # Update job data
        job_data["status"] = status.value
        job_data["updated_at"] = time.time()

        if result is not None:
            # For complex results, create a simplified version
            if isinstance(result, dict):
                # First try to serialize as-is
                try:
                    json.dumps(result)
                    job_data["result"] = result
                except (TypeError, ValueError):
                    # If serialization fails, create a simplified copy
                    simplified_result = {}
                    for k, v in result.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            simplified_result[k] = v
                        elif isinstance(v, dict):
                            simplified_result[k] = {"type": "dict", "summary": f"Dictionary with {len(v)} items"}
                        elif isinstance(v, list):
                            simplified_result[k] = {"type": "list", "summary": f"List with {len(v)} items"}
                        else:
                            simplified_result[k] = str(v)
                    job_data["result"] = simplified_result
            elif isinstance(result, str) and len(result) > 1000:
                # For large string results, truncate
                job_data["result"] = result[:1000] + "... [truncated]"
            else:
                # For other result types, use string representation
                job_data["result"] = str(result)

        if error is not None:
            job_data["error"] = error

        # Write to a temporary file first, then move it atomically
        temp_file = f"{job_file}.temp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2)

        # Verify the temp file can be parsed
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                json.load(f)
            # Replace the original file atomically
            os.replace(temp_file, job_file)
        except Exception as e:
            logging.error(f"Validation of temp job file failed: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

        logging.info(f"Job {job_id} updated to status {status.value}")

        # Emit status update via SocketIO
        if socketio:
            try:
                socketio.emit('status_update', {'job_id': job_id, 'status': status.value, 'error': error})
            except Exception as e:
                logging.error(f"Error emitting SocketIO update: {e}")

        return True

    except Exception as e:
        logging.error(f"Error updating job {job_id}: {e}")
        return False


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get current status of a job with resilient error handling."""
    job_file = os.path.join(get_jobs_dir(), f"{job_id}.json")

    if not os.path.exists(job_file):
        return None

    try:
        with open(job_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing job file for {job_id}: {e}")

        # Check if the corresponding visualization file exists
        viz_file = os.path.join(get_project_root(), "data", "processed", job_id, "cluster_visualization.json")
        if os.path.exists(viz_file):
            # Return a minimal valid status
            return {
                "id": job_id,
                "status": "completed",
                "created_at": time.time() - 3600,
                "updated_at": time.time(),
                "params": {},
                "result": {
                    "output_file": viz_file
                },
                "error": "Status file was corrupted but visualization exists."
            }
        return None
    except Exception as e:
        logging.error(f"Error reading job {job_id}: {e}")
        return None

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    return project_root


def save_to_cache(key: str, data: Any, cache_dir: Optional[str] = None) -> None:
    """
    Cache data to avoid redundant API calls
    """
    import pickle
    if cache_dir is None:
        project_root = get_project_root()
        cache_dir = os.path.join(project_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{key}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def load_from_cache(key: str, cache_dir: Optional[str] = None) -> Any:
    """
    Load data from cache if available
    """
    import pickle
    if cache_dir is None:
        project_root = get_project_root()
        cache_dir = os.path.join(project_root, "cache")
    cache_file = os.path.join(cache_dir, f"{key}.pkl")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Error loading cache for {key}: {e}")
        return None


def clear_cache(confirm=True):
    """Completely clear the cache, processed data, and jobs directories.

    Args:
        confirm (bool): Whether to prompt for confirmation before deletion.

    Returns:
        dict: Status and number of directories cleared.
    """
    directories_to_clear = [
        os.path.join(get_project_root(), "cache"),
        os.path.join(get_project_root(), "data", "processed"),
        os.path.join(get_project_root(), "jobs")
    ]

    # Prompt for confirmation if enabled
    if confirm:
        print(
            f"Are you sure you want to delete all contents in the following directories?\n{', '.join(directories_to_clear)}\nThis action cannot be undone. (y/N)")
        response = input().strip().lower()
        if response != 'y':
            logger.info("Cache clearing cancelled by user")
            return {"status": "cancelled", "message": "Operation cancelled by user"}

    deleted_dirs = 0
    for directory in directories_to_clear:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)  # Recursively delete everything
                os.makedirs(directory, exist_ok=True)  # Recreate empty directory
                deleted_dirs += 1
                logger.info(f"Cleared directory: {directory}")
            except Exception as e:
                logger.error(f"Error deleting directory {directory}: {e}")
        else:
            logger.warning(f"Directory {directory} does not exist")

    logger.info(f"Successfully cleared {deleted_dirs} directories")
    return {"status": "success", "deleted_directories": deleted_dirs}


import os
import pickle
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union


class CacheManager:
    """Enhanced cache management system with support for intermediate data access"""

    def __init__(self, base_cache_dir: Optional[str] = None):
        """
        Initialize the cache manager

        Args:
            base_cache_dir: Base directory for cache storage (uses default if None)
        """
        if base_cache_dir is None:
            project_root = get_project_root()
            self.base_cache_dir = os.path.join(project_root, "cache")
        else:
            self.base_cache_dir = base_cache_dir

        os.makedirs(self.base_cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.base_cache_dir, "cache_metadata.json")
        self.logger = logging.getLogger(__name__)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk or initialize if not present"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cache metadata: {e}")
                self.metadata = {"entries": {}, "last_updated": time.time()}
        else:
            self.metadata = {"entries": {}, "last_updated": time.time()}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            self.logger.warning(f"Error saving cache metadata: {e}")

    def _get_normalized_key(self, key: str) -> str:
        """Normalize cache key for consistency"""
        return key.replace('/', '_').replace('\\', '_')

    def _get_cache_path(self, namespace: str, key: str) -> str:
        """Get the file path for a cache entry"""
        norm_key = self._get_normalized_key(key)
        namespace_dir = os.path.join(self.base_cache_dir, namespace)
        os.makedirs(namespace_dir, exist_ok=True)
        return os.path.join(namespace_dir, f"{norm_key}.pkl")

    def save(self, namespace: str, key: str, data: Any,
             metadata: Optional[Dict[str, Any]] = None,
             expiry: Optional[int] = None) -> bool:
        """
        Save data to cache with namespace organization

        Args:
            namespace: Logical category for data (e.g., 'pmids', 'datasets', 'clusters')
            key: Unique identifier within namespace
            data: Data to cache
            metadata: Additional information about cached data
            expiry: Time in seconds until cache entry expires (None = no expiry)

        Returns:
            bool: Success or failure
        """
        try:
            cache_path = self._get_cache_path(namespace, key)

            # Save the actual data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

            # Record in metadata
            norm_key = self._get_normalized_key(key)
            meta_key = f"{namespace}/{norm_key}"

            entry_metadata = metadata or {}
            entry_metadata.update({
                "created": time.time(),
                "accessed": time.time(),
                "path": cache_path,
                "namespace": namespace
            })

            if expiry is not None:
                entry_metadata["expires"] = time.time() + expiry

            self.metadata["entries"][meta_key] = entry_metadata
            self.metadata["last_updated"] = time.time()
            self._save_metadata()

            return True
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
            return False

    def load(self, namespace: str, key: str, default: Any = None) -> Any:
        """
        Load data from cache

        Args:
            namespace: Logical category for data
            key: Unique identifier within namespace
            default: Value to return if cache miss

        Returns:
            Cached data or default value
        """
        try:
            norm_key = self._get_normalized_key(key)
            meta_key = f"{namespace}/{norm_key}"

            # Check if entry exists and is not expired
            if meta_key in self.metadata["entries"]:
                entry = self.metadata["entries"][meta_key]

                # Check expiry
                if "expires" in entry and time.time() > entry["expires"]:
                    # Expired entry
                    self.invalidate(namespace, key)
                    return default

                cache_path = entry["path"]

                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)

                    # Update access time
                    self.metadata["entries"][meta_key]["accessed"] = time.time()
                    self._save_metadata()

                    return data

            return default
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {e}")
            return default

    def invalidate(self, namespace: str, key: str) -> bool:
        """
        Remove a specific cache entry

        Args:
            namespace: Logical category for data
            key: Unique identifier within namespace

        Returns:
            bool: Success or failure
        """
        try:
            norm_key = self._get_normalized_key(key)
            meta_key = f"{namespace}/{norm_key}"

            if meta_key in self.metadata["entries"]:
                entry = self.metadata["entries"][meta_key]
                cache_path = entry["path"]

                # Remove file if it exists
                if os.path.exists(cache_path):
                    os.remove(cache_path)

                # Remove from metadata
                del self.metadata["entries"][meta_key]
                self.metadata["last_updated"] = time.time()
                self._save_metadata()

            return True
        except Exception as e:
            self.logger.error(f"Error invalidating cache entry: {e}")
            return False

    def list_namespace(self, namespace: str) -> List[str]:
        """
        List all keys in a namespace

        Args:
            namespace: Logical category to list

        Returns:
            List of keys
        """
        try:
            keys = []
            prefix = f"{namespace}/"

            for meta_key in self.metadata["entries"]:
                if meta_key.startswith(prefix):
                    # Extract just the key portion
                    keys.append(meta_key[len(prefix):])

            return keys
        except Exception as e:
            self.logger.error(f"Error listing namespace: {e}")
            return []

    def get_job_cache_dir(self, job_id: str) -> str:
        """
        Get job-specific cache directory

        Args:
            job_id: Unique job identifier

        Returns:
            Path to job-specific cache directory
        """
        job_cache_dir = os.path.join(self.base_cache_dir, job_id)
        os.makedirs(job_cache_dir, exist_ok=True)
        return job_cache_dir

    def get_intermediate_data(self, job_id: str, stage: str) -> Dict[str, Any]:
        """
        Retrieve intermediate processing results for a job

        Args:
            job_id: Job identifier
            stage: Processing stage ('fetching', 'analysis', 'clustering')

        Returns:
            Dictionary of available intermediate data
        """
        try:
            namespace = f"job_{job_id}"
            key = f"intermediate_{stage}"

            data = self.load(namespace, key, {})
            if data:
                self.logger.info(f"Retrieved intermediate {stage} data for job {job_id}")
            else:
                self.logger.debug(f"No intermediate {stage} data found for job {job_id}")

            return data
        except Exception as e:
            self.logger.error(f"Error retrieving intermediate data: {e}")
            return {}

    def save_intermediate_data(self, job_id: str, stage: str, data: Dict[str, Any]) -> bool:
        """
        Save intermediate processing results for a job

        Args:
            job_id: Job identifier
            stage: Processing stage ('fetching', 'analysis', 'clustering')
            data: Intermediate data to save

        Returns:
            bool: Success or failure
        """
        try:
            namespace = f"job_{job_id}"
            key = f"intermediate_{stage}"

            result = self.save(namespace, key, data, {
                "job_id": job_id,
                "stage": stage,
                "timestamp": time.time()
            })

            if result:
                self.logger.info(f"Saved intermediate {stage} data for job {job_id}")
            else:
                self.logger.warning(f"Failed to save intermediate {stage} data for job {job_id}")

            return result
        except Exception as e:
            self.logger.error(f"Error saving intermediate data: {e}")
            return False

    def clear_job_cache(self, job_id: str) -> bool:
        """
        Clear all cache entries for a specific job

        Args:
            job_id: Job identifier

        Returns:
            bool: Success or failure
        """
        try:
            namespace = f"job_{job_id}"
            keys_to_remove = []

            # Find all entries for this job
            prefix = f"{namespace}/"
            for meta_key in list(self.metadata["entries"].keys()):
                if meta_key.startswith(prefix):
                    entry = self.metadata["entries"][meta_key]
                    cache_path = entry["path"]

                    # Remove file if it exists
                    if os.path.exists(cache_path):
                        os.remove(cache_path)

                    # Mark for removal from metadata
                    keys_to_remove.append(meta_key)

            # Remove from metadata
            for key in keys_to_remove:
                del self.metadata["entries"][key]

            self.metadata["last_updated"] = time.time()
            self._save_metadata()

            self.logger.info(f"Cleared cache for job {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing job cache: {e}")
            return False


# Create a global instance of the cache manager
_cache_manager = None


def get_cache_manager(base_cache_dir: Optional[str] = None) -> CacheManager:
    """
    Get or initialize the global cache manager

    Args:
        base_cache_dir: Base directory for cache storage

    Returns:
        CacheManager instance
    """
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(base_cache_dir)

    return _cache_manager


# Compatibility wrappers for existing cache functions
def save_to_cache(key: str, data: Any, cache_dir: Optional[str] = None) -> None:
    """
    Backward-compatible wrapper for cache saving

    Args:
        key: Cache key
        data: Data to cache
        cache_dir: Cache directory
    """
    # Parse key to determine namespace (use simple default if not clear)
    if '_' in key:
        parts = key.split('_', 1)
        namespace = parts[0]
        subkey = parts[1]
    else:
        namespace = "default"
        subkey = key

    cache_manager = get_cache_manager(cache_dir)
    cache_manager.save(namespace, subkey, data)


def load_from_cache(key: str, cache_dir: Optional[str] = None) -> Any:
    """
    Backward-compatible wrapper for cache loading

    Args:
        key: Cache key
        cache_dir: Cache directory

    Returns:
        Cached data or None
    """
    # Parse key to determine namespace
    if '_' in key:
        parts = key.split('_', 1)
        namespace = parts[0]
        subkey = parts[1]
    else:
        namespace = "default"
        subkey = key

    cache_manager = get_cache_manager(cache_dir)
    return cache_manager.load(namespace, subkey)


def repair_job_file(job_id: str) -> bool:
    """
    Repair a corrupted job file by reading it carefully and rewriting it.

    Args:
        job_id: The ID of the job to repair

    Returns:
        bool: True if repair was successful, False otherwise
    """
    job_file = os.path.join(get_jobs_dir(), f"{job_id}.json")
    backup_file = os.path.join(get_jobs_dir(), f"{job_id}_backup.json")

    if not os.path.exists(job_file):
        logging.error(f"Job file for {job_id} not found")
        return False

    try:
        # First try to make a backup of the current file
        shutil.copy2(job_file, backup_file)
        logging.info(f"Created backup of job file at {backup_file}")

        # Try to read the job file carefully
        job_data = None
        try:
            # First attempt normal JSON parsing
            with open(job_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON error in job file: {e}")
            # If that fails, try reading the file and fixing it
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Try to determine where the valid JSON ends
                # Based on the error position, we'll truncate the file
                valid_part = content[:1900]  # Safely before error at char 1938

                # Try to parse this truncated content
                try:
                    job_data = json.loads(valid_part + '}')  # Add closing brace
                except:
                    # If that fails, use a more conservative approach
                    try:
                        # Find the last occurrence of a closing curly brace
                        last_brace = valid_part.rfind('}')
                        if last_brace > 0:
                            job_data = json.loads(valid_part[:last_brace + 1])
                    except:
                        pass
            except Exception as e:
                logging.error(f"Error reading job file: {e}")

        # If we still couldn't recover the data, recreate a minimal valid structure
        if job_data is None:
            job_data = {
                "id": job_id,
                "status": "completed",  # Assume it was completed based on logs
                "created_at": time.time(),
                "updated_at": time.time(),
                "type": "pmid_processing",
                "params": {},
                "result": {
                    "analysis_data": {
                        "dataset_count": 0,
                        "pmid_count": 0
                    },
                    "visualization_data": None,
                    "output_file": os.path.join(get_project_root(), "data", "processed", job_id,
                                                "cluster_visualization.json")
                },
                "error": None
            }

            # Check if we can get visualization data from the output directory
            viz_file = os.path.join(get_project_root(), "data", "processed", job_id, "cluster_visualization.json")
            if os.path.exists(viz_file):
                try:
                    with open(viz_file, "r", encoding="utf-8") as f:
                        viz_data = json.load(f)

                    # Update counts based on visualization data
                    job_data["result"]["visualization_data"] = viz_data
                    job_data["result"]["analysis_data"]["dataset_count"] = len(viz_data.get("datasets", []))
                    if viz_data.get("cluster_info"):
                        job_data["result"]["cluster_count"] = len(viz_data.get("cluster_info", {}))
                except Exception as e:
                    logging.error(f"Error reading visualization data: {e}")

        # Write the repaired job data back to the file
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2)

        logging.info(f"Successfully repaired job file for {job_id}")
        return True

    except Exception as e:
        logging.error(f"Failed to repair job file: {e}")
        # Try to restore from backup if repair failed
        if os.path.exists(backup_file):
            try:
                shutil.copy2(backup_file, job_file)
                logging.info(f"Restored job file from backup")
            except Exception as restore_error:
                logging.error(f"Failed to restore from backup: {restore_error}")
        return False