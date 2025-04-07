"""
Optimized clustering module for GEO dataset clustering based on TF-IDF vectors
with enhanced deterministic methods and error handling
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from scipy.spatial.distance import squareform, pdist

# Import utilities for file paths and caching
from src.utils import get_project_root, save_to_cache, load_from_cache, setup_logging


class ClusteringError(Exception):
    """Exception raised for errors during clustering."""
    pass


class WebServiceError(Exception):
    """Exception raised for errors when called from web service."""
    pass


class DatasetClusterer:
    """Handles clustering of GEO datasets based on TF-IDF distance matrices"""

    def __init__(
            self,
            input_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            job_id: Optional[str] = None
    ):
        """
        Initialize the dataset clusterer

        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save output files
            cache_dir: Directory for caching results
            job_id: Optional job ID for web service requests
        """
        self.logger = logging.getLogger(__name__)

        # Set default directories if not provided
        project_root = get_project_root()

        # If job_id is provided, use job-specific directories
        if job_id:
            if input_dir is None:
                self.input_dir = os.path.join(project_root, "data", "processed", job_id)
            else:
                self.input_dir = input_dir

            if output_dir is None:
                self.output_dir = os.path.join(project_root, "data", "processed", job_id)
            else:
                self.output_dir = output_dir

            if cache_dir is None:
                self.cache_dir = os.path.join(project_root, "cache", job_id)
            else:
                self.cache_dir = cache_dir
        else:
            if input_dir is None:
                self.input_dir = os.path.join(project_root, "data", "processed")
            else:
                self.input_dir = input_dir

            if output_dir is None:
                self.output_dir = os.path.join(project_root, "data", "processed")
            else:
                self.output_dir = output_dir

            if cache_dir is None:
                self.cache_dir = os.path.join(project_root, "cache")
            else:
                self.cache_dir = cache_dir

        self.job_id = job_id

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize data containers
        self.distance_matrix = None
        self.dataset_ids = None
        self.dataset_pmids = None
        self.linkage_matrix = None
        self.cluster_labels = None
        self.coords = None

    def load_analysis_results(
            self,
            distance_matrix_path: Optional[str] = None,
            dataset_info_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load previously calculated distance matrix and dataset information

        Args:
            distance_matrix_path: Path to distance matrix NPY file
            dataset_info_path: Path to dataset information CSV file

        Returns:
            Dictionary containing analysis results

        Raises:
            FileNotFoundError: If required files cannot be found
            ClusteringError: If files cannot be loaded or processed
        """
        # Set default paths if not provided
        if distance_matrix_path is None:
            distance_matrix_path = os.path.join(self.input_dir, "distance_matrix.npy")

        if dataset_info_path is None:
            dataset_info_path = os.path.join(self.input_dir, "dataset_info.csv")

        self.logger.info(f"Loading analysis results from {self.input_dir}")

        try:
            # Check if files exist
            if not os.path.exists(distance_matrix_path):
                error_msg = f"Distance matrix file not found: {distance_matrix_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            if not os.path.exists(dataset_info_path):
                error_msg = f"Dataset info file not found: {dataset_info_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Load distance matrix
            self.distance_matrix = np.load(distance_matrix_path)

            # Load dataset information
            dataset_info = pd.read_csv(dataset_info_path)

            # Extract dataset IDs and associated PMIDs
            self.dataset_ids = dataset_info["dataset_id"].tolist()
            self.dataset_pmids = {}

            for _, row in dataset_info.iterrows():
                pmids = row["associated_pmids"].split(',') if not pd.isna(row["associated_pmids"]) else []
                self.dataset_pmids[row["dataset_id"]] = pmids

            self.logger.info(f"Loaded distance matrix with shape {self.distance_matrix.shape}")
            self.logger.info(f"Loaded information for {len(self.dataset_ids)} datasets")

            results = {
                "distance_matrix": self.distance_matrix,
                "dataset_ids": self.dataset_ids,
                "dataset_pmids": self.dataset_pmids
            }

            return results

        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except Exception as e:
            error_msg = f"Error loading analysis results: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def load_data_for_web(
            self,
            distance_matrix: np.ndarray,
            dataset_ids: List[str],
            dataset_pmids: Dict[str, List[str]]
    ) -> None:
        """
        Load data directly for web interface use instead of from files

        Args:
            distance_matrix: Pre-computed distance matrix
            dataset_ids: List of dataset IDs
            dataset_pmids: Dictionary mapping dataset IDs to PMIDs

        Raises:
            WebServiceError: If data is invalid
        """
        try:
            if distance_matrix.shape[0] != len(dataset_ids):
                raise ValueError(
                    f"Distance matrix shape {distance_matrix.shape} doesn't match dataset count {len(dataset_ids)}")

            self.distance_matrix = distance_matrix
            self.dataset_ids = dataset_ids
            self.dataset_pmids = dataset_pmids

            self.logger.info(f"Loaded data for web service with {len(dataset_ids)} datasets")

        except Exception as e:
            error_msg = f"Error loading data for web service: {str(e)}"
            self.logger.error(error_msg)
            raise WebServiceError(error_msg)

    def _prepare_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Prepare distance matrix for clustering by ensuring it has proper properties

        Args:
            distance_matrix: Raw distance matrix

        Returns:
            Prepared distance matrix
        """
        # Create a copy of the matrix
        distance_matrix_copy = distance_matrix.copy()

        # Ensure no negative values
        if np.any(distance_matrix_copy < 0):
            self.logger.warning("Found negative values in distance matrix. Converting to absolute values.")
            distance_matrix_copy = np.abs(distance_matrix_copy)

        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix_copy, 0)

        # Ensure matrix is symmetric
        if not np.allclose(distance_matrix_copy, distance_matrix_copy.T, rtol=1e-5, atol=1e-8):
            self.logger.warning("Distance matrix is not symmetric. Symmetrizing.")
            distance_matrix_copy = (distance_matrix_copy + distance_matrix_copy.T) / 2

        return distance_matrix_copy

    def determine_optimal_clusters(
            self,
            distance_matrix: Optional[np.ndarray] = None,
            max_clusters: int = 10
    ) -> int:
        """
        Determine optimal number of clusters using silhouette score
        for more deterministic results

        Args:
            distance_matrix: Pairwise distance matrix (uses self.distance_matrix if None)
            max_clusters: Maximum number of clusters to consider

        Returns:
            Optimal cluster count

        Raises:
            ClusteringError: If cluster determination fails
        """
        # Use instance variable if not provided
        if distance_matrix is None:
            if self.distance_matrix is None:
                error_msg = "No distance matrix available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            distance_matrix = self.distance_matrix

        if len(distance_matrix) <= 1:
            return 1

        self.logger.info("Determining optimal number of clusters using silhouette method")

        try:
            # Create cache key for this operation
            matrix_hash = hash(str(distance_matrix.shape) + str(np.sum(distance_matrix)))
            cache_key = f"optimal_clusters_{matrix_hash}_{max_clusters}"
            cached_result = load_from_cache(cache_key, self.cache_dir)

            if cached_result is not None:
                self.logger.info("Using cached optimal cluster count")
                return cached_result["optimal_clusters"]

            # For very small datasets
            if len(distance_matrix) <= 3:
                optimal_k = min(len(distance_matrix), 2)

                # Cache result
                save_to_cache(cache_key, {
                    "optimal_clusters": optimal_k
                }, self.cache_dir)

                return optimal_k

            # Prepare the distance matrix
            distance_matrix_copy = self._prepare_distance_matrix(distance_matrix)

            # Only evaluate from 2 clusters to max_clusters (or dataset size)
            range_n_clusters = range(2, min(max_clusters + 1, len(distance_matrix)))

            silhouette_scores = []
            for n_clusters in range_n_clusters:
                # Use Agglomerative Clustering with deterministic parameters
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )

                # Get cluster labels
                cluster_labels = clusterer.fit_predict(distance_matrix_copy)

                # Calculate silhouette score if we have multiple clusters
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(
                        distance_matrix_copy,
                        cluster_labels,
                        metric='precomputed'
                    )
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(-1)  # Invalid score

            # Find the best silhouette score
            if silhouette_scores:
                optimal_k = range_n_clusters[np.argmax(silhouette_scores)]
            else:
                # Fallback to simple rule if silhouette fails
                optimal_k = max(2, min(5, len(distance_matrix) // 4))

            # Handle special case - if we have very few datasets
            if optimal_k > len(distance_matrix) // 3:
                self.logger.info(
                    f"Reducing clusters from {optimal_k} to {len(distance_matrix) // 3} to avoid too many clusters")
                optimal_k = max(2, min(optimal_k, len(distance_matrix) // 3))

            # Cache result
            save_to_cache(cache_key, {
                "optimal_clusters": optimal_k
            }, self.cache_dir)

            self.logger.info(f"Determined optimal number of clusters: {optimal_k}")
            return optimal_k

        except Exception as e:
            error_msg = f"Error determining optimal clusters: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def perform_hierarchical_clustering(
            self,
            distance_matrix: Optional[np.ndarray] = None,
            n_clusters: Optional[int] = None,
            linkage_method: str = 'average'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hierarchical clustering on the distance matrix

        Args:
            distance_matrix: Pairwise distance matrix (uses self.distance_matrix if None)
            n_clusters: Number of clusters (if None, determined automatically)
            linkage_method: Linkage method for hierarchical clustering

        Returns:
            Tuple containing (cluster_labels, linkage_matrix)

        Raises:
            ClusteringError: If clustering fails
        """
        # Use instance variable if not provided
        if distance_matrix is None:
            if self.distance_matrix is None:
                error_msg = "No distance matrix available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            distance_matrix = self.distance_matrix

        self.logger.info(f"Performing hierarchical clustering using {linkage_method} linkage")

        try:
            # Create cache key for this clustering operation
            matrix_hash = hash(str(distance_matrix.shape) + str(np.sum(distance_matrix)))
            cache_key = f"hierarchical_clustering_{matrix_hash}_{n_clusters}_{linkage_method}"
            cached_result = load_from_cache(cache_key, self.cache_dir)

            if cached_result is not None:
                self.logger.info("Using cached hierarchical clustering results")
                cluster_labels, linkage_matrix = cached_result["cluster_labels"], cached_result["linkage_matrix"]
            else:
                # Prepare distance matrix
                distance_matrix_copy = self._prepare_distance_matrix(distance_matrix)

                # Determine number of clusters if not specified
                if n_clusters is None:
                    n_clusters = self.determine_optimal_clusters(distance_matrix_copy)

                # For datasets with only one element, assign single cluster
                if len(distance_matrix_copy) <= 1:
                    self.logger.warning("Only one dataset provided, assigning to a single cluster")
                    cluster_labels = np.array([1])
                    linkage_matrix = np.zeros((0, 4))  # Empty linkage matrix
                else:
                    # Convert to condensed form for linkage
                    condensed_distances = squareform(distance_matrix_copy)

                    # Compute linkage matrix for deterministic clustering
                    linkage_matrix = linkage(condensed_distances, method=linkage_method)

                    # Use fcluster for consistent label assignment
                    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

                # Cache results
                save_to_cache(cache_key, {
                    "cluster_labels": cluster_labels,
                    "linkage_matrix": linkage_matrix
                }, self.cache_dir)

            # Store results as instance variables
            self.cluster_labels = cluster_labels
            self.linkage_matrix = linkage_matrix

            self.logger.info(f"Created {len(set(cluster_labels))} clusters")
            return cluster_labels, linkage_matrix

        except Exception as e:
            error_msg = f"Error performing hierarchical clustering: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def reduce_dimensions(
            self,
            distance_matrix: Optional[np.ndarray] = None,
            method: str = 'pca',
            n_components: int = 2,
            random_state: int = 42
    ) -> np.ndarray:
        """
        Reduce dimensions of the distance matrix for visualization

        Args:
            distance_matrix: Pairwise distance matrix (uses self.distance_matrix if None)
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_components: Number of components for the reduced space
            random_state: Random seed for reproducibility

        Returns:
            Reduced-dimension coordinates

        Raises:
            ClusteringError: If dimension reduction fails
            ValueError: If unknown method is specified
        """
        # Use instance variable if not provided
        if distance_matrix is None:
            if self.distance_matrix is None:
                error_msg = "No distance matrix available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            distance_matrix = self.distance_matrix

        method = method.lower()
        self.logger.info(f"Reducing dimensions using {method} to {n_components} components")

        try:
            # For single element, return simple coordinates
            if len(distance_matrix) <= 1:
                self.logger.warning("Only one dataset provided, using default coordinates")
                return np.zeros((len(distance_matrix), n_components))

            # Create cache key for this reduction operation
            matrix_hash = hash(str(distance_matrix.shape) + str(np.sum(distance_matrix)))
            cache_key = f"dim_reduction_{matrix_hash}_{method}_{n_components}_{random_state}"
            cached_coords = load_from_cache(cache_key, self.cache_dir)

            if cached_coords is not None:
                self.logger.info(f"Using cached {method} reduction results")
                reduced_coords = cached_coords
            else:
                # Prepare distance matrix
                distance_matrix_copy = self._prepare_distance_matrix(distance_matrix)

                if method == 'tsne':
                    # Apply t-SNE for visualization
                    # When using precomputed distances, we must use random initialization
                    tsne = TSNE(
                        n_components=n_components,
                        metric='precomputed',
                        random_state=random_state,
                        init='random'  # Use random initialization with precomputed distances
                    )
                    reduced_coords = tsne.fit_transform(distance_matrix_copy)

                elif method == 'pca':
                    # For PCA, we need to convert distances to a feature matrix
                    # Use MDS to get a Euclidean embedding from distances
                    from sklearn.manifold import MDS
                    mds = MDS(
                        n_components=min(50, distance_matrix_copy.shape[0] - 1),
                        dissimilarity='precomputed',
                        random_state=random_state
                    )
                    mds_coords = mds.fit_transform(distance_matrix_copy)

                    # Apply PCA to the MDS coordinates
                    pca = PCA(n_components=n_components, random_state=random_state)
                    reduced_coords = pca.fit_transform(mds_coords)

                else:
                    raise ValueError(f"Unknown dimensionality reduction method: {method}")

                # Cache results
                save_to_cache(cache_key, reduced_coords, self.cache_dir)

            # Store results as instance variable
            self.coords = reduced_coords

            self.logger.info(f"Reduced dimensions to shape {reduced_coords.shape}")
            return reduced_coords

        except ValueError:
            # Re-raise ValueError for unknown methods
            raise
        except Exception as e:
            error_msg = f"Error reducing dimensions: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def create_cluster_visualization_data(
            self,
            distance_matrix: Optional[np.ndarray] = None,
            dataset_ids: Optional[List[str]] = None,
            dataset_pmids: Optional[Dict[str, List[str]]] = None,
            n_clusters: Optional[int] = None,
            dim_reduction_method: str = 'pca'
    ) -> Dict[str, Any]:
        """
        Create data for cluster visualization

        Args:
            distance_matrix: Pairwise distance matrix (uses self.distance_matrix if None)
            dataset_ids: List of dataset IDs (uses self.dataset_ids if None)
            dataset_pmids: Dictionary mapping dataset IDs to associated PMIDs (uses self.dataset_pmids if None)
            n_clusters: Number of clusters (if None, determined automatically)
            dim_reduction_method: Dimensionality reduction method

        Returns:
            Dictionary containing visualization data

        Raises:
            ClusteringError: If visualization data creation fails
        """
        # Use instance variables if not provided
        if distance_matrix is None:
            if self.distance_matrix is None:
                error_msg = "No distance matrix available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            distance_matrix = self.distance_matrix

        if dataset_ids is None:
            if self.dataset_ids is None:
                error_msg = "No dataset IDs available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            dataset_ids = self.dataset_ids

        if dataset_pmids is None:
            if self.dataset_pmids is None:
                error_msg = "No dataset PMIDs available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            dataset_pmids = self.dataset_pmids

        self.logger.info("Creating cluster visualization data")

        try:
            # Create cache key for this visualization operation
            matrix_hash = hash(str(distance_matrix.shape) + str(np.sum(distance_matrix)))
            ids_hash = hash(tuple(dataset_ids))
            cache_key = f"viz_data_{matrix_hash}_{ids_hash}_{n_clusters}_{dim_reduction_method}"
            cached_viz_data = load_from_cache(cache_key, self.cache_dir)

            if cached_viz_data is not None:
                self.logger.info("Using cached visualization data")
                return cached_viz_data

            # Perform hierarchical clustering
            cluster_labels, linkage_matrix = self.perform_hierarchical_clustering(
                distance_matrix=distance_matrix,
                n_clusters=n_clusters
            )

            # Reduce dimensions
            coords = self.reduce_dimensions(
                distance_matrix=distance_matrix,
                method=dim_reduction_method
            )

            # Prepare visualization data
            viz_data = {
                "datasets": [],
                "linkage_matrix": linkage_matrix.tolist(),
                "cluster_info": {}
            }

            # Create data for each dataset
            for i, dataset_id in enumerate(dataset_ids):
                cluster = int(cluster_labels[i])

                # Add dataset to visualization data
                viz_data["datasets"].append({
                    "id": dataset_id,
                    "cluster": cluster,
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "pmids": dataset_pmids.get(dataset_id, [])
                })

                # Update cluster information
                if cluster not in viz_data["cluster_info"]:
                    viz_data["cluster_info"][cluster] = {
                        "count": 0,
                        "datasets": [],
                        "pmids": set()
                    }

                viz_data["cluster_info"][cluster]["count"] += 1
                viz_data["cluster_info"][cluster]["datasets"].append(dataset_id)

                # Add PMIDs to cluster
                for pmid in dataset_pmids.get(dataset_id, []):
                    viz_data["cluster_info"][cluster]["pmids"].add(pmid)

            # Convert sets to lists for JSON serialization
            for cluster in viz_data["cluster_info"]:
                viz_data["cluster_info"][cluster]["pmids"] = list(viz_data["cluster_info"][cluster]["pmids"])

            # Cache visualization data
            save_to_cache(cache_key, viz_data, self.cache_dir)

            self.logger.info(
                f"Created visualization data for {len(viz_data['datasets'])} datasets in {len(viz_data['cluster_info'])} clusters")
            return viz_data

        except Exception as e:
            error_msg = f"Error creating visualization data: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def save_visualization_data(
            self,
            viz_data: Dict[str, Any],
            output_dir: Optional[str] = None,
            filename: str = "cluster_visualization.json"
    ) -> str:
        """
        Save visualization data to JSON file

        Args:
            viz_data: Visualization data dictionary
            output_dir: Directory to save output files (uses self.output_dir if None)
            filename: Name of the output file

        Returns:
            Path to the saved JSON file

        Raises:
            ClusteringError: If saving fails
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save visualization data to JSON
            output_file = os.path.join(output_dir, filename)
            with open(output_file, 'w') as f:
                json.dump(viz_data, f, indent=2)

            self.logger.info(f"Saved visualization data to {output_file}")
            return output_file

        except Exception as e:
            error_msg = f"Error saving visualization data: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def plot_dendrogram(
            self,
            linkage_matrix: Optional[np.ndarray] = None,
            dataset_ids: Optional[List[str]] = None,
            max_leaf_nodes: int = 50,
            output_file: Optional[str] = None
    ) -> None:
        """
        Plot dendrogram of hierarchical clustering

        Args:
            linkage_matrix: Linkage matrix from hierarchical clustering (uses self.linkage_matrix if None)
            dataset_ids: List of dataset IDs (uses self.dataset_ids if None)
            max_leaf_nodes: Maximum number of leaf nodes to display
            output_file: Path to save the plot

        Raises:
            ClusteringError: If plotting fails
        """
        # Use instance variables if not provided
        if linkage_matrix is None:
            if self.linkage_matrix is None:
                error_msg = "No linkage matrix available. Perform clustering first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            linkage_matrix = self.linkage_matrix

        if dataset_ids is None:
            if self.dataset_ids is None:
                error_msg = "No dataset IDs available. Load analysis results first."
                self.logger.error(error_msg)
                raise ClusteringError(error_msg)
            dataset_ids = self.dataset_ids

        if output_file is None:
            output_file = os.path.join(self.output_dir, "dendrogram.png")

        self.logger.info("Creating dendrogram plot")

        try:
            # For single element, skip dendrogram
            if len(dataset_ids) <= 1 or linkage_matrix.shape[0] == 0:
                self.logger.warning("Not enough data for dendrogram, skipping plot")
                return

            plt.figure(figsize=(10, 7))

            # If we have too many datasets, truncate the dendrogram
            if len(dataset_ids) > max_leaf_nodes:
                self.logger.info(f"Truncating dendrogram to {max_leaf_nodes} leaf nodes")
                truncate_mode = 'lastp'
            else:
                truncate_mode = None

            # Create dendrogram
            dendrogram(
                linkage_matrix,
                truncate_mode=truncate_mode,
                p=max_leaf_nodes,
                leaf_font_size=10,
                labels=dataset_ids if len(dataset_ids) <= max_leaf_nodes else None
            )

            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Dataset')
            plt.ylabel('Distance')
            plt.tight_layout()

            # Save the plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved dendrogram to {output_file}")

        except Exception as e:
            error_msg = f"Error plotting dendrogram: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def cluster_datasets(
            self,
            n_clusters: Optional[int] = None,
            dim_reduction_method: str = 'pca',
            save_dendrogram: bool = True
    ) -> Dict[str, Any]:
        """
        Perform clustering on GEO datasets based on TF-IDF distances

        Args:
            n_clusters: Number of clusters (if None, determined automatically)
            dim_reduction_method: Dimensionality reduction method
            save_dendrogram: Whether to save dendrogram plot

        Returns:
            Dictionary containing clustering results

        Raises:
            ClusteringError: If clustering fails
        """
        self.logger.info("Clustering GEO datasets")

        try:
            # Check if clustering results are cached
            cache_key = f"clustering_{n_clusters}_{dim_reduction_method}"
            clustering_results = load_from_cache(cache_key, self.cache_dir)

            if clustering_results is not None:
                self.logger.info(f"Loaded clustering results from cache")
                return clustering_results

            # Load analysis results if not already loaded
            if self.distance_matrix is None or self.dataset_ids is None or self.dataset_pmids is None:
                self.load_analysis_results()

            # Create visualization data
            viz_data = self.create_cluster_visualization_data(
                n_clusters=n_clusters,
                dim_reduction_method=dim_reduction_method
            )

            # Save visualization data
            viz_file = self.save_visualization_data(viz_data)

            # Create and save dendrogram if requested
            if save_dendrogram:
                self.plot_dendrogram()

            # Prepare results
            clustering_results = {
                "visualization_data": viz_data,
                "output_file": viz_file
            }

            # Save results to cache
            save_to_cache(cache_key, clustering_results, self.cache_dir)

            return clustering_results

        except Exception as e:
            error_msg = f"Error clustering datasets: {str(e)}"
            self.logger.error(error_msg)
            raise ClusteringError(error_msg)

    def cluster_datasets_web(
            self,
            distance_matrix: np.ndarray,
            dataset_ids: List[str],
            dataset_pmids: Dict[str, List[str]],
            n_clusters: Optional[int] = None,
            dim_reduction_method: str = 'pca'
    ) -> Dict[str, Any]:
        """
        Web service compatible method to cluster datasets with data provided directly

        Args:
            distance_matrix: Pre-computed distance matrix
            dataset_ids: List of dataset IDs
            dataset_pmids: Dictionary mapping dataset IDs to PMIDs
            n_clusters: Number of clusters (if None, determined automatically)
            dim_reduction_method: Dimensionality reduction method

        Returns:
            Dictionary containing visualization data suitable for web display

        Raises:
            WebServiceError: If web clustering fails
        """
        try:
            # Load data directly instead of from files
            self.load_data_for_web(distance_matrix, dataset_ids, dataset_pmids)

            # Create visualization data
            viz_data = self.create_cluster_visualization_data(
                n_clusters=n_clusters,
                dim_reduction_method=dim_reduction_method
            )

            return viz_data

        except Exception as e:
            error_msg = f"Error clustering datasets for web service: {str(e)}"
            self.logger.error(error_msg)
            raise WebServiceError(error_msg)


# Command-line interface
if __name__ == "__main__":
    import argparse

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perform clustering on GEO datasets")
    parser.add_argument("--input", help="Directory containing input files")
    parser.add_argument("--output", help="Directory to save output files")
    parser.add_argument("--cache", help="Cache directory")
    parser.add_argument("--clusters", type=int, help="Number of clusters (if not specified, determined automatically)")
    parser.add_argument("--method", default="pca", choices=["tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--job-id", help="Job ID for web service requests")
    args = parser.parse_args()

    try:
        # Create dataset clusterer
        clusterer = DatasetClusterer(
            input_dir=args.input,
            output_dir=args.output,
            cache_dir=args.cache,
            job_id=args.job_id
        )

        # Perform clustering
        results = clusterer.cluster_datasets(
            n_clusters=args.clusters,
            dim_reduction_method=args.method
        )

        # Print summary of results
        viz_data = results["visualization_data"]
        print(f"\nClustering complete:")
        print(f"  - Created {len(viz_data['cluster_info'])} clusters for {len(viz_data['datasets'])} datasets")

        # Print cluster sizes
        print("\nCluster sizes:")
        for cluster, info in sorted(viz_data["cluster_info"].items()):
            print(f"  Cluster {cluster}: {info['count']} datasets, {len(info['pmids'])} PMIDs")

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except ClusteringError as e:
        logger.error(f"Clustering error: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except WebServiceError as e:
        logger.error(f"Web service error: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: An unexpected error occurred. See log for details.")
        exit(1)