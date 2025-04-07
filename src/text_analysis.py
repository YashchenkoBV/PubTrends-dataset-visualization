"""
Text analysis module for GEO dataset clustering using TF-IDF vectorization
with enhanced error handling and organization
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, pdist
import logging

# Import utilities for file paths and caching
from src.utils import get_project_root, save_to_cache, load_from_cache, setup_logging


class TextAnalysisError(Exception):
    """Exception raised for errors during text analysis."""
    pass


class TextAnalyzer:
    """Handles text analysis and vectorization for GEO dataset descriptions"""

    def __init__(
            self,
            output_dir: Optional[str] = None,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize the text analyzer

        Args:
            output_dir: Directory to save output files
            cache_dir: Directory for caching results
        """
        self.logger = logging.getLogger(__name__)

        # Set default directories if not provided
        project_root = get_project_root()

        if output_dir is None:
            self.output_dir = os.path.join(project_root, "data", "processed")
        else:
            self.output_dir = output_dir

        if cache_dir is None:
            self.cache_dir = os.path.join(project_root, "cache")
        else:
            self.cache_dir = cache_dir

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_dataset_info(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load GEO dataset information from JSON file

        Args:
            file_path: Path to the JSON file containing dataset information

        Returns:
            Dictionary containing dataset information

        Raises:
            FileNotFoundError: If the file doesn't exist
            TextAnalysisError: If the file cannot be parsed
        """
        if file_path is None:
            # Use default path
            file_path = os.path.join(self.output_dir, "geo_dataset_info.json")

        self.logger.info(f"Loading dataset information from {file_path}")

        if not os.path.exists(file_path):
            error_msg = f"Dataset information file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(file_path, 'r') as f:
                dataset_info = json.load(f)

            dataset_count = len(dataset_info.get("datasets", {}))
            self.logger.info(f"Loaded information for {dataset_count} datasets")

            return dataset_info

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing dataset information file: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading dataset information: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)

    def prepare_text_data(
            self,
            dataset_info: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract and combine text fields from dataset information

        Args:
            dataset_info: Dictionary containing dataset information

        Returns:
            Tuple containing (dataset_ids, combined_texts)
        """
        self.logger.info("Preparing text data by combining all fields")

        dataset_ids = []
        combined_texts = []

        try:
            for geo_id, data in dataset_info["datasets"].items():
                # Combine all text fields with spaces in between
                combined_text = " ".join([
                    data.get("title", ""),
                    data.get("experiment_type", ""),
                    data.get("summary", ""),
                    data.get("organism", ""),
                    data.get("overall_design", "")
                ])

                dataset_ids.append(geo_id)
                combined_texts.append(combined_text)

            self.logger.info(f"Prepared text data for {len(dataset_ids)} datasets")
            return dataset_ids, combined_texts

        except KeyError as e:
            error_msg = f"Missing key in dataset information: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)
        except Exception as e:
            error_msg = f"Error preparing text data: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)

    def vectorize_text(
            self,
            texts: List[str],
            max_features: int = 5000,
            min_df: int = 2
    ) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
        """
        Apply TF-IDF vectorization to the text data

        Args:
            texts: List of text documents to vectorize
            max_features: Maximum number of features (terms) to keep
            min_df: Minimum document frequency for terms

        Returns:
            Tuple containing (tfidf_matrix, feature_names, vectorizer)

        Raises:
            TextAnalysisError: If vectorization fails
        """
        self.logger.info(f"Vectorizing text with max_features={max_features}, min_df={min_df}")

        try:
            # Create and configure TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                stop_words='english',
                lowercase=True,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
                norm='l2'  # L2 normalization
            )

            # Fit and transform the text data
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Get feature names (terms)
            feature_names = vectorizer.get_feature_names_out()

            self.logger.info(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
            self.logger.info(f"Number of features (terms): {len(feature_names)}")

            return tfidf_matrix, feature_names, vectorizer

        except ValueError as e:
            error_msg = f"Vectorization error: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during vectorization: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)

    def calculate_distance_matrix(
            self,
            tfidf_matrix: np.ndarray,
            metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Calculate distance matrix between TF-IDF vectors

        Args:
            tfidf_matrix: Matrix of TF-IDF vectors
            metric: Distance metric to use (default: cosine)

        Returns:
            Distance matrix

        Raises:
            TextAnalysisError: If distance calculation fails
        """
        self.logger.info(f"Calculating pairwise distances using {metric} metric")

        try:
            # Calculate pairwise distances
            # For cosine_similarity, we need to convert to distances (1 - similarity)
            if metric == 'cosine':
                similarities = cosine_similarity(tfidf_matrix)
                distances = 1 - similarities
            else:
                # For other metrics, use pdist/squareform
                distances = squareform(pdist(tfidf_matrix.toarray(), metric=metric))

            self.logger.info(f"Created distance matrix with shape {distances.shape}")
            return distances

        except ValueError as e:
            error_msg = f"Distance calculation error: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error calculating distances: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)

    def analyze_dataset_text(
            self,
            dataset_info_path: Optional[str] = None,
            max_features: int = 5000,
            min_df: int = 2,
            distance_metric: str = 'cosine'
    ) -> Dict[str, Any]:
        """
        Perform text analysis on GEO dataset descriptions

        Args:
            dataset_info_path: Path to dataset information JSON file
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency for terms
            distance_metric: Distance metric to use

        Returns:
            Dictionary containing analysis results

        Raises:
            FileNotFoundError: If the dataset information file doesn't exist
            TextAnalysisError: If analysis fails
        """
        self.logger.info("Starting text analysis of GEO dataset descriptions")

        # Check if analysis results are cached
        cache_key = f"text_analysis_{max_features}_{min_df}_{distance_metric}"
        analysis_results = load_from_cache(cache_key, self.cache_dir)

        if analysis_results is not None:
            self.logger.info(f"Loaded text analysis results from cache")
            return analysis_results

        try:
            # Load dataset information
            dataset_info = self.load_dataset_info(dataset_info_path)

            # Prepare text data
            dataset_ids, combined_texts = self.prepare_text_data(dataset_info)

            # Vectorize text
            tfidf_matrix, feature_names, vectorizer = self.vectorize_text(
                combined_texts,
                max_features=max_features,
                min_df=min_df
            )

            # Calculate distance matrix
            distance_matrix = self.calculate_distance_matrix(
                tfidf_matrix,
                metric=distance_metric
            )

            # Extract PMIDs for each dataset
            dataset_pmids = {}
            for i, geo_id in enumerate(dataset_ids):
                dataset_pmids[geo_id] = dataset_info["datasets"][geo_id].get("associated_pmids", [])

            # Create analysis results dictionary
            analysis_results = {
                "dataset_ids": dataset_ids,
                "tfidf_matrix": tfidf_matrix,
                "feature_names": feature_names.tolist(),
                "distance_matrix": distance_matrix,
                "dataset_pmids": dataset_pmids,
                "vectorizer": vectorizer,
                "parameters": {
                    "max_features": max_features,
                    "min_df": min_df,
                    "distance_metric": distance_metric
                }
            }

            # Save results to cache
            save_to_cache(cache_key, analysis_results, self.cache_dir)

            # Save outputs to files
            self._save_analysis_outputs(
                distance_matrix=distance_matrix,
                dataset_ids=dataset_ids,
                dataset_pmids=dataset_pmids,
                tfidf_matrix=tfidf_matrix,
                feature_names=feature_names
            )

            self.logger.info(f"Completed text analysis of {len(dataset_ids)} datasets")
            return analysis_results

        except FileNotFoundError:
            # Re-raise as is
            raise
        except TextAnalysisError:
            # Re-raise as is
            raise
        except Exception as e:
            error_msg = f"Unexpected error during text analysis: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)

    def _save_analysis_outputs(
            self,
            distance_matrix: np.ndarray,
            dataset_ids: List[str],
            dataset_pmids: Dict[str, List[str]],
            tfidf_matrix: np.ndarray,
            feature_names: np.ndarray
    ) -> None:
        """
        Save analysis outputs to files

        Args:
            distance_matrix: Pairwise distance matrix
            dataset_ids: List of dataset IDs
            dataset_pmids: Dictionary mapping dataset IDs to associated PMIDs
            tfidf_matrix: TF-IDF matrix
            feature_names: List of feature names (terms)
        """
        try:
            # Save distance matrix
            np.save(os.path.join(self.output_dir, "distance_matrix.npy"), distance_matrix)

            # Save dataset IDs with their associated PMIDs
            dataset_info_df = pd.DataFrame({
                "dataset_id": dataset_ids,
                "associated_pmids": [','.join(dataset_pmids[geo_id]) for geo_id in dataset_ids]
            })
            dataset_info_df.to_csv(os.path.join(self.output_dir, "dataset_info.csv"), index=False)

            # Save TF-IDF matrix
            from scipy.sparse import save_npz
            save_npz(os.path.join(self.output_dir, "tfidf_matrix.npz"), tfidf_matrix)

            # Save feature names
            with open(os.path.join(self.output_dir, "feature_names.json"), 'w') as f:
                json.dump(feature_names.tolist(), f)

            self.logger.info(f"Saved analysis outputs to {self.output_dir}")

        except Exception as e:
            error_msg = f"Error saving analysis outputs: {str(e)}"
            self.logger.error(error_msg)
            # Continue without raising - this is a non-critical operation

    def get_top_terms_per_dataset(
            self,
            tfidf_matrix: np.ndarray,
            feature_names: List[str],
            dataset_ids: List[str],
            n_terms: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top terms for each dataset based on TF-IDF weights

        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: List of feature names (terms)
            dataset_ids: List of dataset IDs
            n_terms: Number of top terms to extract

        Returns:
            Dictionary mapping dataset IDs to their top terms with weights
        """
        self.logger.info(f"Extracting top {n_terms} terms for each dataset")

        try:
            top_terms = {}

            # Convert sparse matrix to array for easier indexing if it's not already
            if hasattr(tfidf_matrix, 'toarray'):
                tfidf_array = tfidf_matrix.toarray()
            else:
                tfidf_array = tfidf_matrix

            for i, dataset_id in enumerate(dataset_ids):
                # Get TF-IDF weights for this dataset
                tfidf_weights = tfidf_array[i]

                # Get indices of top terms
                top_indices = np.argsort(tfidf_weights)[-n_terms:][::-1]

                # Extract terms and weights
                terms_weights = [(feature_names[idx], tfidf_weights[idx]) for idx in top_indices]

                top_terms[dataset_id] = terms_weights

            return top_terms

        except Exception as e:
            error_msg = f"Error extracting top terms: {str(e)}"
            self.logger.error(error_msg)
            raise TextAnalysisError(error_msg)


# Command-line interface
if __name__ == "__main__":
    import argparse

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perform text analysis on GEO datasets")
    parser.add_argument("--input", help="Path to dataset information JSON file")
    parser.add_argument("--output", help="Directory to save output files")
    parser.add_argument("--cache", help="Cache directory")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum number of features for TF-IDF")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for terms")
    parser.add_argument("--metric", default="cosine", help="Distance metric to use")
    args = parser.parse_args()

    try:
        # Create text analyzer
        analyzer = TextAnalyzer(
            output_dir=args.output,
            cache_dir=args.cache
        )

        # Perform text analysis
        results = analyzer.analyze_dataset_text(
            dataset_info_path=args.input,
            max_features=args.max_features,
            min_df=args.min_df,
            distance_metric=args.metric
        )

        # Print some statistics
        print(f"\nText analysis complete:")
        print(f"  - TF-IDF matrix shape: {results['tfidf_matrix'].shape}")
        print(f"  - Number of datasets: {len(results['dataset_ids'])}")
        print(f"  - Number of features: {len(results['feature_names'])}")

        # Extract and print top terms for a sample dataset
        top_terms = analyzer.get_top_terms_per_dataset(
            results['tfidf_matrix'],
            results['feature_names'],
            results['dataset_ids'],
            n_terms=10
        )

        # Print top terms for the first 3 datasets (or fewer if less than 3)
        sample_size = min(3, len(results['dataset_ids']))
        for i, dataset_id in enumerate(results['dataset_ids'][:sample_size]):
            print(f"\nTop terms for dataset {dataset_id}:")
            for term, weight in top_terms[dataset_id]:
                print(f"  {term}: {weight:.4f}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except TextAnalysisError as e:
        logger.error(f"Text analysis error: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: An unexpected error occurred. See log for details.")
        exit(1)