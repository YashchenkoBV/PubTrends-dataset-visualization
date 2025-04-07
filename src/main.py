"""
Main module for integrating GEO dataset processing components
and providing asynchronous processing capabilities with real-time updates.
"""

import os
import tempfile
import logging
import re
from typing import List, Dict, Any, Optional
import io
from contextlib import redirect_stdout

from src.data_processing import DataProcessor
from src.text_analysis import TextAnalyzer
from src.clustering import DatasetClusterer
from src.utils import create_job, run_async_job, get_project_root, update_job_status, JobStatus

logger = logging.getLogger(__name__)


def validate_pmids(pmids: List[str]) -> List[str]:
    """
    Validate PMIDs and return only valid ones.

    Args:
        pmids: List of PMIDs to validate.

    Returns:
        List of valid PMIDs.
    """
    valid_pmids = []
    for pmid in pmids:
        # Remove any whitespace and validate format
        cleaned_pmid = pmid.strip()
        if re.match(r'^\d+$', cleaned_pmid):
            valid_pmids.append(cleaned_pmid)
        else:
            logger.warning(f"Invalid PMID format: '{pmid}' - skipping")

    return valid_pmids


def process_pmids_async(pmids: List[str], n_clusters: Optional[int] = None,
                        dim_reduction: str = 'tsne', job_name: str = None) -> str:
    """
    Process a list of PMIDs asynchronously.

    Args:
        pmids: List of PMIDs to process.
        n_clusters: Number of clusters (if None, determined automatically).
        dim_reduction: Dimensionality reduction method.
        job_name: Optional name for the job

    Returns:
        Job ID for tracking progress.
    """
    # Validate PMIDs first
    valid_pmids = validate_pmids(pmids)

    if not valid_pmids:
        raise ValueError("No valid PMIDs provided")

    # Create a job
    job_params = {
        "pmids": valid_pmids,
        "n_clusters": n_clusters,
        "dim_reduction": dim_reduction
    }

    if job_name:
        job_params["job_name"] = job_name

    job_id = create_job("pmid_processing", job_params)

    # Run the job asynchronously
    run_async_job(job_id, _process_job, job_id, valid_pmids, n_clusters, dim_reduction)

    return job_id


def _process_job(job_id: str, pmids: List[str], n_clusters: Optional[int],
                 dim_reduction: str) -> Dict[str, Any]:
    """Process PMIDs with safer result handling."""
    # Create a StringIO buffer to capture output
    output_capture = io.StringIO()

    # Get cache manager
    from src.utils import get_cache_manager
    cache_manager = get_cache_manager()

    try:
        with redirect_stdout(output_capture):
            print(f"Job {job_id}: Starting processing of {len(pmids)} PMIDs")
            update_job_status(job_id, JobStatus.VALIDATING)

            # Create job-specific directories
            project_root = get_project_root()
            job_output_dir = os.path.join(project_root, "data", "processed", job_id)
            job_cache_dir = os.path.join(project_root, "cache", job_id)

            os.makedirs(job_output_dir, exist_ok=True)
            os.makedirs(job_cache_dir, exist_ok=True)

            # Create temporary file for PMIDs
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                for pmid in pmids:
                    f.write(f"{pmid}\n")
                temp_file = f.name
            print(f"Job {job_id}: Created temporary file at {temp_file}")

            update_job_status(job_id, JobStatus.FETCHING_DATA)
            # Process PMIDs and retrieve dataset information
            from src.data_processing import DataProcessor
            processor = DataProcessor(output_dir=job_output_dir, cache_dir=job_cache_dir)
            analysis_data, errors = processor.process_pmids(temp_file)

            # Save intermediate results for 'fetching' stage
            cache_manager.save_intermediate_data(job_id, "fetching", {
                "pmid_geo_map": analysis_data.get("pmid_geo_map", {}),
                "datasets": analysis_data.get("datasets", {}),
                "errors": errors
            })

            print(f"Job {job_id}: Retrieved information for {len(analysis_data.get('datasets', {}))} datasets")

            update_job_status(job_id, JobStatus.ANALYZING_TEXT)
            # Perform text analysis
            from src.text_analysis import TextAnalyzer
            analyzer = TextAnalyzer(output_dir=job_output_dir, cache_dir=job_cache_dir)
            dataset_info_path = os.path.join(job_output_dir, "geo_dataset_info.json")
            text_results = analyzer.analyze_dataset_text(dataset_info_path=dataset_info_path)

            # Save intermediate results for 'analysis' stage
            cache_manager.save_intermediate_data(job_id, "analysis", {
                "dataset_ids": text_results.get("dataset_ids", []),
                "parameters": text_results.get("parameters", {})
            })

            print(f"Job {job_id}: Completed text analysis")

            update_job_status(job_id, JobStatus.CLUSTERING)
            # Perform clustering
            from src.clustering import DatasetClusterer
            clusterer = DatasetClusterer(input_dir=job_output_dir, output_dir=job_output_dir, cache_dir=job_cache_dir)
            clustering_results = clusterer.cluster_datasets(
                n_clusters=n_clusters,
                dim_reduction_method=dim_reduction
            )

            # Save intermediate results for 'clustering' stage (final results)
            cache_manager.save_intermediate_data(job_id, "clustering", {
                "output_file": clustering_results.get("output_file", "")
            })

            print(f"Job {job_id}: Completed clustering")

            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Job {job_id}: Failed to delete temporary file: {e}")

            # Create a simplified results structure without large objects
            results = {
                "analysis_data": {
                    "dataset_count": len(analysis_data.get('datasets', {})),
                    "pmid_count": len(analysis_data.get('pmid_geo_map', {}))
                },
                "errors": errors[:10] if errors else [],
                "output_file": clustering_results.get("output_file", "")
            }
            print(f"Job {job_id}: Processing completed successfully")

        # Update job status with the simplified result
        # DO NOT include the full captured output or visualization data
        update_job_status(job_id, JobStatus.COMPLETED, result=results)
        return results

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Job {job_id} failed: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error=error_msg)
        raise