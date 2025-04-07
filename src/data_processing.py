"""
Optimized data processing module for GEO dataset information retrieval
with enhanced error handling and validation
"""
import time
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union, Tuple, Any
from bs4 import BeautifulSoup
import json
import os
import logging

# Import utilities for file paths and caching
from src.utils import setup_logging, save_to_cache, load_from_cache, get_project_root


class ValidationError(Exception):
    """Exception raised for validation errors in the input data."""
    pass


class APIError(Exception):
    """Exception raised for errors in API calls."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_text: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)


def validate_pmid(pmid: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single PMID.

    Args:
        pmid: A potential PMID string to validate

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing (is_valid, error_message)
    """
    # Check if PMID is empty
    if not pmid or not pmid.strip():
        return False, "PMID cannot be empty"

    # Check if PMID contains only digits
    if not pmid.strip().isdigit():
        return False, f"PMID must contain only digits, got '{pmid}'"

    # Check if PMID has reasonable length (typically 1-8 digits)
    if len(pmid.strip()) > 12:
        return False, f"PMID is too long, expected 1-12 digits, got {len(pmid)} digits"

    return True, None


def validate_pmids(pmids: List[str]) -> List[Dict[str, Union[str, bool]]]:
    """
    Validate a list of PMIDs and return validation results.

    Args:
        pmids: List of PMIDs to validate

    Returns:
        List of dictionaries containing validation results for each PMID

    Raises:
        ValidationError: If all PMIDs are invalid
    """
    if not pmids:
        raise ValidationError("No PMIDs provided for validation")

    results = []
    for pmid in pmids:
        is_valid, error_message = validate_pmid(pmid)
        results.append({
            "pmid": pmid,
            "valid": is_valid,
            "error": error_message
        })

    # Check if all PMIDs are invalid
    if all(not result["valid"] for result in results):
        raise ValidationError("All provided PMIDs are invalid")

    return results


def filter_valid_pmids(pmids: List[str]) -> Tuple[List[str], List[Dict[str, Union[str, bool]]]]:
    """
    Filter a list of PMIDs to keep only valid ones and return validation results.

    Args:
        pmids: List of PMIDs to validate and filter

    Returns:
        Tuple containing (valid_pmids, validation_results)

    Raises:
        ValidationError: If all PMIDs are invalid
    """
    validation_results = validate_pmids(pmids)
    valid_pmids = [result["pmid"] for result in validation_results if result["valid"]]

    if not valid_pmids:
        raise ValidationError("No valid PMIDs found in the input list")

    return valid_pmids, validation_results


def validate_geo_id(geo_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a GEO dataset ID.

    Args:
        geo_id: A potential GEO ID to validate

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing (is_valid, error_message)
    """
    if not geo_id or not geo_id.strip():
        return False, "GEO ID cannot be empty"

    # GEO IDs can be numeric or have a specific format (GSE followed by numbers)
    if geo_id.isdigit():
        return True, None

    # Check format for GSE IDs
    pattern = r'^GSE\d+$'
    if re.match(pattern, geo_id):
        return True, None

    return False, f"Invalid GEO ID format: '{geo_id}'. Expected numeric or GSE format (e.g., GSE123)"


class EUtilsClient:
    """Client for interacting with NCBI E-utilities API with enhanced error handling"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    GEO_WEB_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

    # Status codes that might benefit from retries
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

    def __init__(
            self,
            email: Optional[str] = None,
            api_key: Optional[str] = None,
            delay: float = 0.34,
            max_retries: int = 3,
            retry_delay: float = 1.0
    ):
        """
        Initialize E-utilities client with enhanced error handling

        Args:
            email: Email for NCBI API (recommended)
            api_key: API key for NCBI API (optional)
            delay: Time delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries (will be increased exponentially)
        """
        self.email = email
        self.api_key = api_key
        self.delay = delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0
        self.logger = logging.getLogger(__name__)

    def _respect_rate_limit(self):
        """Ensure we don't exceed NCBI's rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.delay:
            sleep_time = self.delay - time_since_last_request
            self.logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _build_auth_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Add authentication parameters if available"""
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _make_request(
            self,
            endpoint: str,
            params: Dict[str, str],
            retry_count: int = 0
    ) -> requests.Response:
        """
        Make an HTTP request to the E-utilities API with retry logic

        Args:
            endpoint: API endpoint
            params: Request parameters
            retry_count: Current retry attempt

        Returns:
            requests.Response: API response

        Raises:
            APIError: If the request fails after all retry attempts
        """
        self._respect_rate_limit()
        params = self._build_auth_params(params)

        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)

            # Check if request was successful
            if response.status_code == 200:
                return response

            # Decide whether to retry
            if (
                    retry_count < self.max_retries
                    and response.status_code in self.RETRY_STATUS_CODES
            ):
                # Calculate exponential backoff delay
                sleep_time = self.retry_delay * (2 ** retry_count)
                self.logger.warning(
                    f"Request failed with status {response.status_code}. "
                    f"Retrying in {sleep_time:.2f} seconds (attempt {retry_count + 1}/{self.max_retries})"
                )
                time.sleep(sleep_time)
                return self._make_request(endpoint, params, retry_count + 1)

            # If we get here, retries are exhausted or status code doesn't warrant retry
            raise APIError(
                f"API request failed with status code {response.status_code}",
                status_code=response.status_code,
                response_text=response.text
            )

        except requests.RequestException as e:
            # Handle network errors and timeouts
            if retry_count < self.max_retries:
                sleep_time = self.retry_delay * (2 ** retry_count)
                self.logger.warning(
                    f"Request failed with error: {str(e)}. "
                    f"Retrying in {sleep_time:.2f} seconds (attempt {retry_count + 1}/{self.max_retries})"
                )
                time.sleep(sleep_time)
                return self._make_request(endpoint, params, retry_count + 1)

            raise APIError(f"API request failed after {self.max_retries} attempts: {str(e)}")

    def fetch_geo_ids(self, pmid: str) -> List[str]:
        """
        Retrieve GEO dataset IDs linked to a PubMed ID

        Args:
            pmid: PubMed ID

        Returns:
            List of GEO dataset IDs

        Raises:
            APIError: If the API request fails
            ValueError: If the response cannot be parsed
        """
        self.logger.info(f"Fetching GEO IDs for PMID: {pmid}")

        params = {
            "dbfrom": "pubmed",
            "db": "gds",
            "linkname": "pubmed_gds",
            "id": pmid,
            "retmode": "xml"
        }

        try:
            response = self._make_request("elink.fcgi", params)

            # Parse XML response
            try:
                root = ET.fromstring(response.text)
                geo_ids = []

                for linkset in root.findall('.//LinkSet'):
                    for linksetdb in linkset.findall('.//LinkSetDb'):
                        if (
                                linksetdb.find('DbTo') is not None and
                                linksetdb.find('DbTo').text == 'gds' and
                                linksetdb.find('LinkName') is not None and
                                linksetdb.find('LinkName').text == 'pubmed_gds'
                        ):
                            for link in linksetdb.findall('.//Link'):
                                if link.find('Id') is not None:
                                    geo_id = link.find('Id').text
                                    is_valid, _ = validate_geo_id(geo_id)
                                    if is_valid:
                                        geo_ids.append(geo_id)
                                    else:
                                        self.logger.warning(f"Found invalid GEO ID: {geo_id} for PMID {pmid}")

                if not geo_ids:
                    self.logger.info(f"No GEO datasets found for PMID {pmid}")
                else:
                    self.logger.info(f"Found {len(geo_ids)} GEO datasets for PMID {pmid}")

                return geo_ids

            except ET.ParseError as e:
                error_msg = f"Error parsing XML response for PMID {pmid}: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        except APIError as e:
            self.logger.error(f"API error while fetching GEO IDs for PMID {pmid}: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error fetching GEO IDs for PMID {pmid}: {str(e)}"
            self.logger.error(error_msg)
            raise APIError(error_msg)

    def _extract_field_content(self, text: str, field_name: str) -> str:
        """
        Extract content for a specific field from raw text
        """
        # Common field names to use as potential end markers
        end_markers = [
            "Status", "Title", "Organism", "Experiment type",
            "Summary", "Overall design", "Contributor", "Citation",
            "Submission date", "Last update date", "Contact", "Web link"
        ]

        pattern = rf"{re.escape(field_name)}[:\s]*([^\n]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        field_idx = text.find(field_name)
        if field_idx >= 0:
            markers = [m for m in end_markers if m != field_name]

            start_idx = field_idx + len(field_name)

            end_idx = len(text)
            for marker in markers:
                marker_idx = text.find(marker, start_idx)
                if marker_idx > 0 and marker_idx < end_idx:
                    end_idx = marker_idx

            # Extract and clean the content
            content = text[start_idx:end_idx].strip()
            # Remove leading colon or whitespace if present
            return re.sub(r'^[:;\s]+', '', content)

        return ""

    def _extract_from_html(self, html: str, field_name: str) -> str:
        """Extract field content directly from HTML structure"""
        # Try to find a table row with the field name
        field_row_match = re.search(
            rf'<tr[^>]*>.*?{re.escape(field_name)}.*?<td[^>]*>(.*?)</td>',
            html,
            re.DOTALL | re.IGNORECASE
        )

        if field_row_match:
            field_html = field_row_match.group(1)
            field_soup = BeautifulSoup(field_html, 'html.parser')
            return field_soup.get_text(strip=True)

        # Alternative approach: find TD after field name
        field_idx = html.find(field_name)
        if field_idx >= 0:
            td_start = html.find("<td", field_idx)
            td_end = html.find("</td>", td_start)
            if td_start > 0 and td_end > td_start:
                field_html = html[td_start:td_end + 5]
                field_soup = BeautifulSoup(field_html, 'html.parser')
                return field_soup.get_text(strip=True)

        return ""

    def _scrape_geo_page(self, geo_accession: str) -> Dict[str, str]:
        """
        Scrape relevant fields from a GEO webpage
        """
        # Normalize GEO accession ID
        if geo_accession.isdigit():
            geo_accession = f"GSE{geo_accession}"

        url = f"{self.GEO_WEB_URL}?acc={geo_accession}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                self.logger.warning(f"Error: HTTP status {response.status_code}")
                return {}

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text from all elements
            all_text = ""
            for element in soup.find_all(['p', 'div', 'td', 'th', 'li', 'h1', 'h2', 'h3', 'h4', 'h5']):
                element_text = element.get_text(strip=True)
                if element_text:
                    all_text += element_text + "\n"

            # Extract the overall design field
            fields = {
                "overall_design": self._extract_field_content(all_text, "Overall design")
            }

            # If text extraction failed, try HTML structure-based extraction
            if not fields["overall_design"]:
                fields["overall_design"] = self._extract_from_html(response.text, "Overall design")

            return fields

        except Exception as e:
            self.logger.warning(f"Error scraping GEO page: {str(e)}")
            return {}

    def _trim_overall_design(self, text: str) -> str:
        """
        Trim the overall design field when specific boundary words appear
        """
        boundary_words = ["Contributor", "Citation", "Web link"]

        trimmed_text = text
        for word in boundary_words:
            index = text.find(word)
            if index > 0 and (len(trimmed_text) > index):
                trimmed_text = text[:index].strip()

        return trimmed_text

    def fetch_geo_details(self, geo_id: str) -> Dict[str, str]:
        """
        Retrieve detailed information about a GEO dataset with enhanced error handling

        Args:
            geo_id: GEO dataset ID

        Returns:
            Dictionary containing dataset information

        Raises:
            APIError: If the API request fails
            ValueError: If the response cannot be parsed
        """
        self.logger.info(f"Fetching details for GEO ID: {geo_id}")

        # Initialize empty result structure
        dataset_info = {
            "title": "",
            "experiment_type": "",
            "summary": "",
            "organism": "",
            "overall_design": ""
        }

        params = {
            "db": "gds",
            "id": geo_id,
            "retmode": "xml"
        }

        try:
            response = self._make_request("esummary.fcgi", params)

            try:
                root = ET.fromstring(response.text)

                doc_sum = root.find('.//DocSum')
                if doc_sum is None:
                    self.logger.warning(f"No dataset information found for GEO ID {geo_id}")
                    return dataset_info

                # Get accession ID for potential fallback
                accession = ""
                for item in doc_sum.findall('.//Item'):
                    if item.get('Name') == 'Accession':
                        accession = item.text
                        break

                field_mapping = {
                    'title': 'title',
                    'summary': 'summary',
                    'gdsType': 'experiment_type',
                    'taxon': 'organism',
                    'design': 'overall_design'
                }

                for item in doc_sum.findall('.//Item'):
                    item_name = item.get('Name')
                    if item_name in field_mapping and item.text:
                        dataset_info[field_mapping[item_name]] = item.text

                # Check if overall_design field is empty and try fallback
                if not dataset_info["overall_design"] and accession:
                    try:
                        scraped_fields = self._scrape_geo_page(accession)
                        if scraped_fields.get("overall_design"):
                            dataset_info["overall_design"] = scraped_fields["overall_design"]
                    except Exception as scrape_error:
                        self.logger.warning(f"Fallback scraping failed for {geo_id}: {str(scrape_error)}")

                # Trim the overall design field if it exists
                if dataset_info["overall_design"]:
                    dataset_info["overall_design"] = self._trim_overall_design(dataset_info["overall_design"])

                return dataset_info

            except ET.ParseError as e:
                error_msg = f"Error parsing XML for GEO ID {geo_id}: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        except APIError as e:
            self.logger.error(f"API error while fetching details for GEO ID {geo_id}: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error fetching details for GEO ID {geo_id}: {str(e)}"
            self.logger.error(error_msg)
            raise APIError(error_msg)


class DataProcessor:
    """Handles processing of PubMed IDs and GEO datasets with robust error handling"""

    def __init__(
            self,
            email: Optional[str] = None,
            api_key: Optional[str] = None,
            output_dir: Optional[str] = None,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize the data processor

        Args:
            email: Email for NCBI API (recommended)
            api_key: API key for NCBI API (optional)
            output_dir: Directory to store output files
            cache_dir: Directory for caching results
        """
        self.client = EUtilsClient(email=email, api_key=api_key)
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

    def _get_pmid_cache_key(self, pmid: str) -> str:
        """Generate a cache key for a single PMID"""
        return f"pmid_{pmid}"

    def _get_geo_cache_key(self, geo_id: str) -> str:
        """Generate a cache key for a single GEO dataset"""
        return f"geo_{geo_id}"

    def _load_individual_cache(self, key: str) -> Optional[Any]:
        """Load a single item from cache"""
        return load_from_cache(key, self.cache_dir)

    def _save_individual_cache(self, key: str, data: Any) -> None:
        """Save a single item to cache"""
        save_to_cache(key, data, self.cache_dir)

    def load_pmids(self, file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Read and validate PMIDs from a text file

        Args:
            file_path: Path to file containing PMIDs

        Returns:
            Tuple containing (valid_pmids, validation_results)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValidationError: If all PMIDs are invalid
        """
        self.logger.info(f"Loading PMIDs from {file_path}")

        if not os.path.exists(file_path):
            error_msg = f"PMID file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(file_path, 'r') as f:
                raw_pmids = [line.strip() for line in f if line.strip()]

            self.logger.info(f"Loaded {len(raw_pmids)} PMIDs from file")

            # Validate PMIDs
            valid_pmids, validation_results = filter_valid_pmids(raw_pmids)

            # Log validation results
            invalid_count = len(raw_pmids) - len(valid_pmids)
            if invalid_count > 0:
                self.logger.warning(
                    f"Found {invalid_count} invalid PMIDs out of {len(raw_pmids)}"
                )
                for result in validation_results:
                    if not result["valid"]:
                        self.logger.warning(
                            f"Invalid PMID '{result['pmid']}': {result['error']}"
                        )

            return valid_pmids, validation_results

        except ValidationError as e:
            self.logger.error(f"PMID validation failed: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Error loading PMIDs from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def fetch_geo_ids_for_pmids(
            self,
            pmids: List[str]
    ) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
        """
        Fetch GEO IDs for a list of PubMed IDs with error tracking and dynamic caching
        """
        if not pmids:
            raise ValidationError("No valid PMIDs provided")

        self.logger.info(f"Fetching GEO IDs for {len(pmids)} PMIDs")

        # Prepare cache key for the entire result
        pmids_hash = hash(tuple(sorted(pmids)))
        complete_cache_key = f"pmid_geo_map_{pmids_hash}"

        # Check if complete result is cached
        cached_data = load_from_cache(complete_cache_key, self.cache_dir)
        if cached_data:
            self.logger.info(f"Loaded GEO IDs for {len(cached_data['pmid_geo_map'])} PMIDs from cache")
            return cached_data["pmid_geo_map"], cached_data["error_records"]

        # Initialize results
        pmid_geo_map = {}
        error_records = []

        # Process each PMID, checking individual cache first
        for pmid in pmids:
            # Check if this PMID's results are already cached
            pmid_cache_key = self._get_pmid_cache_key(pmid)
            cached_pmid_result = self._load_individual_cache(pmid_cache_key)

            if cached_pmid_result:
                # Use cached result
                if "error" in cached_pmid_result:
                    error_records.append(cached_pmid_result["error"])
                else:
                    pmid_geo_map[pmid] = cached_pmid_result["geo_ids"]
                self.logger.info(f"PMID {pmid}: Loaded GEO IDs from cache")
                continue

            # If not cached, fetch from API
            try:
                geo_ids = self.client.fetch_geo_ids(pmid)
                pmid_geo_map[pmid] = geo_ids
                self.logger.info(f"PMID {pmid}: Found {len(geo_ids)} GEO datasets")

                # Cache this individual PMID result
                self._save_individual_cache(pmid_cache_key, {
                    "geo_ids": geo_ids
                })

            except APIError as e:
                error_record = {
                    "pmid": pmid,
                    "error_type": "api_error",
                    "error_message": str(e),
                    "status_code": getattr(e, "status_code", None)
                }
                self.logger.error(f"API error for PMID {pmid}: {str(e)}")
                error_records.append(error_record)

                # Cache the error too
                self._save_individual_cache(pmid_cache_key, {
                    "error": error_record
                })

            except Exception as e:
                error_record = {
                    "pmid": pmid,
                    "error_type": "unexpected_error",
                    "error_message": str(e)
                }
                self.logger.error(f"Unexpected error for PMID {pmid}: {str(e)}")
                error_records.append(error_record)

                # Cache the error
                self._save_individual_cache(pmid_cache_key, {
                    "error": error_record
                })

        # Cache complete results
        cache_data = {
            "pmid_geo_map": pmid_geo_map,
            "error_records": error_records
        }
        save_to_cache(complete_cache_key, cache_data, self.cache_dir)

        self.logger.info(
            f"Fetched GEO IDs for {len(pmid_geo_map)} PMIDs with {len(error_records)} errors"
        )
        return pmid_geo_map, error_records

    def fetch_dataset_details(
            self,
            pmid_geo_map: Dict[str, List[str]],
            max_workers: int = 5,
            batch_size: int = 10
    ) -> Tuple[Dict[str, Dict], List[Dict[str, Any]]]:
        """
        Fetch detailed information for GEO datasets with parallel processing and enhanced caching

        Args:
            pmid_geo_map: Dictionary mapping PMIDs to lists of GEO dataset IDs
            max_workers: Maximum number of concurrent workers for API requests
            batch_size: Number of GEO IDs to process in each batch

        Returns:
            Tuple containing (dataset_collection, error_records)
        """
        import concurrent.futures
        from time import time

        self.logger.info(f"Fetching dataset details using {max_workers} workers")

        # Create a map of GEO IDs to their associated PMIDs
        geo_pmid_map = {}
        for pmid, geo_ids in pmid_geo_map.items():
            for geo_id in geo_ids:
                if geo_id not in geo_pmid_map:
                    geo_pmid_map[geo_id] = []
                geo_pmid_map[geo_id].append(pmid)

        # Prepare cache key for the entire result
        geo_ids_hash = hash(tuple(sorted(geo_pmid_map.keys())))
        complete_cache_key = f"dataset_collection_{geo_ids_hash}"

        # Check if complete result is cached
        cached_data = load_from_cache(complete_cache_key, self.cache_dir)
        if cached_data:
            self.logger.info(
                f"Loaded details for {len(cached_data['dataset_collection'])} datasets from cache"
            )
            return cached_data["dataset_collection"], cached_data["error_records"]

        # Define the worker function that processes a batch of GEO IDs
        def process_geo_batch(geo_batch):
            batch_collection = {}
            batch_errors = []

            for geo_id, pmids in geo_batch:
                # Check if this GEO ID's details are already cached
                geo_cache_key = self._get_geo_cache_key(geo_id)
                cached_geo_result = self._load_individual_cache(geo_cache_key)

                if cached_geo_result:
                    # Use cached result
                    if "error" in cached_geo_result:
                        batch_errors.append(cached_geo_result["error"])
                    else:
                        # Update PMIDs to ensure they're current
                        dataset_info = cached_geo_result["dataset_info"]
                        dataset_info['associated_pmids'] = pmids
                        batch_collection[geo_id] = dataset_info
                    self.logger.debug(f"Loaded details for GEO ID {geo_id} from cache")
                    continue

                # If not cached, fetch from API
                try:
                    dataset_info = self.client.fetch_geo_details(geo_id)

                    if dataset_info:
                        dataset_info['associated_pmids'] = pmids
                        batch_collection[geo_id] = dataset_info
                        self.logger.debug(f"Retrieved details for GEO ID {geo_id}")

                        # Cache this individual GEO result
                        self._save_individual_cache(geo_cache_key, {
                            "dataset_info": dataset_info
                        })
                    else:
                        error_record = {
                            "geo_id": geo_id,
                            "error_type": "no_details",
                            "error_message": "No dataset details returned",
                            "associated_pmids": pmids
                        }
                        self.logger.warning(f"No details found for GEO ID {geo_id}")
                        batch_errors.append(error_record)

                        # Cache the error
                        self._save_individual_cache(geo_cache_key, {
                            "error": error_record
                        })

                except APIError as e:
                    error_record = {
                        "geo_id": geo_id,
                        "error_type": "api_error",
                        "error_message": str(e),
                        "status_code": getattr(e, "status_code", None),
                        "associated_pmids": pmids
                    }
                    self.logger.error(f"API error for GEO ID {geo_id}: {str(e)}")
                    batch_errors.append(error_record)

                    # Cache the error
                    self._save_individual_cache(geo_cache_key, {
                        "error": error_record
                    })

                except Exception as e:
                    error_record = {
                        "geo_id": geo_id,
                        "error_type": "unexpected_error",
                        "error_message": str(e),
                        "associated_pmids": pmids
                    }
                    self.logger.error(f"Unexpected error for GEO ID {geo_id}: {str(e)}")
                    batch_errors.append(error_record)

                    # Cache the error
                    self._save_individual_cache(geo_cache_key, {
                        "error": error_record
                    })

            return batch_collection, batch_errors

        # Prepare batches of GEO IDs for parallel processing
        geo_items = list(geo_pmid_map.items())
        geo_batches = [geo_items[i:i + batch_size] for i in range(0, len(geo_items), batch_size)]

        self.logger.info(f"Processing {len(geo_pmid_map)} GEO IDs in {len(geo_batches)} batches")

        # Initialize results
        dataset_collection = {}
        error_records = []

        # Process batches in parallel with controlled concurrency
        start_time = time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to the executor
            future_to_batch = {executor.submit(process_geo_batch, batch): i
                               for i, batch in enumerate(geo_batches)}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_collection, batch_errors = future.result()
                    dataset_collection.update(batch_collection)
                    error_records.extend(batch_errors)

                    # Log progress
                    completed = batch_idx + 1
                    self.logger.info(f"Completed batch {completed}/{len(geo_batches)} "
                                     f"({completed * 100 / len(geo_batches):.1f}%)")

                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed with error: {str(e)}")

        elapsed_time = time() - start_time
        self.logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")

        # Cache complete results
        cache_data = {
            "dataset_collection": dataset_collection,
            "error_records": error_records
        }
        save_to_cache(complete_cache_key, cache_data, self.cache_dir)

        self.logger.info(
            f"Fetched details for {len(dataset_collection)} datasets with {len(error_records)} errors"
        )
        return dataset_collection, error_records

    def prepare_dataset_for_text_analysis(
            self,
            dataset_collection: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """
        Combine text fields from dataset information for TF-IDF analysis

        Args:
            dataset_collection: Dictionary containing dataset information

        Returns:
            Dictionary with dataset IDs as keys and combined text as values
        """
        self.logger.info("Preparing dataset text for analysis")

        text_data = {}

        for geo_id, data in dataset_collection.items():
            # Combine all text fields with spaces in between
            combined_text = " ".join([
                data.get("title", ""),
                data.get("experiment_type", ""),
                data.get("summary", ""),
                data.get("organism", ""),
                data.get("overall_design", "")
            ])

            # Store with GEO ID as key
            text_data[geo_id] = {
                "text": combined_text,
                "pmids": data.get("associated_pmids", [])
            }

        self.logger.info(f"Prepared text data for {len(text_data)} datasets")
        return text_data

    def process_pmids(
            self,
            pmids_file: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process PMIDs file and retrieve all related GEO dataset information

        Args:
            pmids_file: Path to file containing PMIDs

        Returns:
            Tuple containing (analysis_data, error_summary)

        Raises:
            FileNotFoundError: If the PMIDs file doesn't exist
            ValidationError: If all PMIDs are invalid
        """
        self.logger.info(f"Processing PMIDs from {pmids_file}")

        try:
            # Load and validate PMIDs
            valid_pmids, validation_results = self.load_pmids(pmids_file)

            # Fetch GEO IDs for valid PMIDs
            pmid_geo_map, geo_fetch_errors = self.fetch_geo_ids_for_pmids(valid_pmids)

            # Fetch dataset details
            dataset_collection, details_fetch_errors = self.fetch_dataset_details(pmid_geo_map)

            # Prepare result structure
            analysis_data = {
                "pmid_geo_map": pmid_geo_map,
                "datasets": dataset_collection
            }

            # Prepare error summary
            error_summary = []

            # Add validation errors
            for result in validation_results:
                if not result["valid"]:
                    error_summary.append({
                        "stage": "validation",
                        "pmid": result["pmid"],
                        "error_type": "validation_error",
                        "error_message": result["error"]
                    })

            # Add GEO fetch errors
            for error in geo_fetch_errors:
                error_summary.append({
                    "stage": "geo_id_fetch",
                    **error
                })

            # Add details fetch errors
            for error in details_fetch_errors:
                error_summary.append({
                    "stage": "dataset_details_fetch",
                    **error
                })

            # Save results to file
            self._save_results(analysis_data, error_summary)

            return analysis_data, error_summary

        except ValidationError as e:
            self.logger.error(f"PMID validation failed: {str(e)}")
            raise
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in processing: {str(e)}")
            raise RuntimeError(f"Error processing PMIDs: {str(e)}")

    def _save_results(
            self,
            analysis_data: Dict[str, Any],
            error_summary: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Save processing results to output files

        Args:
            analysis_data: Dictionary containing analysis data
            error_summary: List of error records

        Returns:
            Tuple containing (analysis_file_path, error_file_path)
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save analysis data
        analysis_file = os.path.join(self.output_dir, "geo_dataset_info.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        # Save error summary
        error_file = os.path.join(self.output_dir, "error_summary.json")
        with open(error_file, 'w') as f:
            json.dump(error_summary, f, indent=2)

        self.logger.info(f"Saved dataset information to {analysis_file}")
        self.logger.info(f"Saved error summary to {error_file}")

        return analysis_file, error_file


# Command-line interface
if __name__ == "__main__":
    import argparse

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PMIDs and fetch linked GEO datasets")
    parser.add_argument("--pmids", help="Path to file containing PMIDs")
    parser.add_argument("--output", help="Output directory for processed data")
    parser.add_argument("--cache", help="Cache directory")
    parser.add_argument("--email", help="Email for NCBI API")
    parser.add_argument("--api-key", help="API key for NCBI API")
    args = parser.parse_args()

    # Determine PMIDs file path with fallbacks
    pmids_file = args.pmids
    if not pmids_file:
        # Try several possible locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "PMIDs_list.txt"),
            os.path.join(os.path.dirname(script_dir), "PMIDs_list.txt"),
            os.path.join(os.path.dirname(script_dir), "data", "PMIDs_list.txt")
        ]

        for path in possible_paths:
            if os.path.exists(path):
                pmids_file = path
                break

        if not pmids_file:
            logger.error("PMID file not found in any of the default locations")
            parser.error("PMIDs file not found. Please specify with --pmids")

    try:
        # Create data processor
        processor = DataProcessor(
            email=args.email,
            api_key=args.api_key,
            output_dir=args.output,
            cache_dir=args.cache
        )

        # Process PMIDs
        analysis_data, error_summary = processor.process_pmids(pmids_file)

        # Print summary
        dataset_count = len(analysis_data.get("datasets", {}))
        pmid_count = len(analysis_data.get("pmid_geo_map", {}))
        error_count = len(error_summary)

        print(f"\nProcessing complete:")
        print(f"  - Processed {pmid_count} PMIDs")
        print(f"  - Retrieved {dataset_count} GEO datasets")
        print(f"  - Encountered {error_count} errors")

        if error_count > 0:
            print(f"\nErrors have been saved to error_summary.json")

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
