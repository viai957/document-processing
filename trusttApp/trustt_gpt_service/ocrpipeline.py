import tempfile
import json 
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import sys
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import fitz
import cv2
import time
import base64
from PIL import Image
from pathlib import Path
import concurrent.futures
import re
import psutil
import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime

# Try to setup logger if available, otherwise create a simple logger
try:
    from trusttApp.common.logger import setup_logger
    TGPT_LOG_NAME = os.getenv("TGPT_LOG_NAME", "ocr_service.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logger('trustt_gpt_service', TGPT_LOG_NAME, level=LOG_LEVEL)
except ImportError:
    import logging
    logger = logging.getLogger('ocr_service')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

minicpm_api_key = os.getenv("MINICPM_API_KEY")
minicpm_api_base = os.getenv("MINICPM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

@dataclass
class OCRConfig:
    """Configuration for OCR processing with adaptive parameters"""
    # Resource management
    max_workers_multiplier: float = 1.0  # Reduced to minimize parallel processing issues
    min_workers: int = 1  # Minimize to avoid resource conflicts
    max_workers: int = 4  # Significantly reduced to avoid conflicts
    memory_safety_margin: float = 0.3  # Keep 30% memory free
    
    # OCR processing
    dpi: int = 400  # Higher DPI for better quality
    force_full_page_ocr: bool = True  # Force full page OCR even if text layer exists
    enable_table_detection: bool = True  # Enable table structure detection
    
    # Image preprocessing - simplified
    enable_preprocessing: bool = False  # Disable preprocessing, let RapidOCR handle it
    deskew_pages: bool = True  # Only keep deskewing as it can help with alignment
    
    # OCR engines and fallbacks
    primary_engine: str = "rapidocr"  # Primary OCR engine
    fallback_engines: List[str] = field(default_factory=lambda: ["tesseract"])  # Fallback OCR engines
    
    # Processing strategy
    chunk_size: int = 1  # Process one page at a time for maximum reliability
    enable_checkpointing: bool = True  # Enable checkpointing for recovery
    checkpoint_frequency: int = 1  # Save checkpoint after each page
    
    # Caching
    enable_caching: bool = True  # Enable result caching
    cache_dir: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), "ocr_cache"))
    cache_ttl_days: int = 7  # Cache time-to-live in days
    
    # OpenAI API settings
    max_tokens: int = 8000  # Maximum tokens for OpenAI API
    temperature: float = 0.1  # Lower temperature for more deterministic output
    top_p: float = 0.95  # Higher top_p for better quality
    
    def __post_init__(self):
        """Ensure cache directory exists"""
        if self.enable_caching and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_optimal_workers(self) -> int:
        """Dynamically determine optimal number of workers based on system resources"""
        # Calculate based on CPU cores available
        cpu_count = os.cpu_count() or 4
        suggested_workers = int(cpu_count * self.max_workers_multiplier)
        return max(self.min_workers, min(suggested_workers, self.max_workers))

class ResourceMonitor:
    """Monitors system resources and provides adaptive scaling recommendations"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.start_time = time.time()
        self.last_check = self.start_time
        self.check_interval = 5  # seconds
        
    def should_adjust_resources(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if resources should be adjusted based on current system state"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False, {}
            
        self.last_check = current_time
        
        # Check memory usage
        mem = psutil.virtual_memory()
        memory_usage = mem.percent / 100
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100
        
        # Determine if adjustment is needed
        needs_adjustment = False
        adjustment = {}
        
        # Memory pressure is high
        if memory_usage > (1 - self.config.memory_safety_margin):
            needs_adjustment = True
            adjustment["reduce_workers"] = True
            adjustment["reason"] = "High memory pressure"
            
        # CPU is underutilized but we have memory
        elif cpu_usage < 0.3 and memory_usage < 0.7:
            needs_adjustment = True
            adjustment["increase_workers"] = True
            adjustment["reason"] = "System underutilized"
            
        return needs_adjustment, adjustment
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        mem = psutil.virtual_memory()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": mem.percent,
            "memory_available_gb": mem.available / (1024 ** 3),
            "elapsed_time": time.time() - self.start_time,
        }

class CacheManager:
    """Manages caching of OCR results to avoid redundant processing"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.enable_caching = config.enable_caching
        self.cache_dir = config.cache_dir
        
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._clean_old_cache()
    
    def _clean_old_cache(self):
        """Clean cache entries older than the TTL"""
        if not self.enable_caching:
            return
            
        current_time = time.time()
        ttl_seconds = self.config.cache_ttl_days * 24 * 60 * 60
        
        for cache_file in os.listdir(self.cache_dir):
            cache_path = os.path.join(self.cache_dir, cache_file)
            if os.path.isfile(cache_path):
                file_mtime = os.path.getmtime(cache_path)
                if current_time - file_mtime > ttl_seconds:
                    try:
                        os.remove(cache_path)
                        logger.debug(f"Removed old cache file: {cache_file}")
                    except:
                        pass
    
    def get_cache_key(self, page_data: Any, config_hash: str) -> str:
        """Generate a unique cache key for a page based on its content and processing parameters"""
        # Handle different types of data for caching
        if isinstance(page_data, dict):
            # Convert dict to bytes using JSON serialization
            page_data_bytes = json.dumps(page_data, sort_keys=True).encode('utf-8')
        elif isinstance(page_data, str):
            # Convert string to bytes
            page_data_bytes = page_data.encode('utf-8')
        elif not isinstance(page_data, bytes):
            # For other non-bytes objects, convert to string representation
            try:
                # Try to get a deterministic hash of the object
                page_data_bytes = str(hash(page_data)).encode('utf-8')
            except TypeError:
                # If object is not hashable, use its string representation
                page_data_bytes = str(page_data).encode('utf-8')
        else:
            # Already bytes
            page_data_bytes = page_data
            
        content_hash = hashlib.md5(page_data_bytes).hexdigest()
        return f"{content_hash}_{config_hash}"
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached OCR results if available"""
        if not self.enable_caching:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                logger.info(f"Cache hit for key: {cache_key[:8]}...")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                
        return None
    
    def save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """Save OCR results to cache"""
        if not self.enable_caching:
            return False
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached result for key: {cache_key[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False

class ImagePreprocessor:
    """Simplified image preprocessing - focused only on deskewing when needed"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.enable_preprocessing = config.enable_preprocessing
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply minimal preprocessing - only deskew if enabled"""
        if not self.enable_preprocessing:
            return image
            
        # If deskewing is enabled, apply it
        if self.config.deskew_pages:
            return self._deskew(image)
        
        return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image by detecting and correcting skewed text lines"""
        try:
            # Convert to grayscale if needed for deskew algorithm
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Find all non-zero points
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
            
            # Correct the angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Rotate the image to deskew it if angle is significant
            if abs(angle) > 0.5:  # Only correct if skew is noticeable
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h), 
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            
        return image

class CheckpointManager:
    """Manages checkpoints for long-running OCR processes to enable recovery"""
    
    def __init__(self, config: OCRConfig, document_id: str):
        self.config = config
        self.enable_checkpointing = config.enable_checkpointing
        self.checkpoint_frequency = config.checkpoint_frequency
        self.document_id = document_id
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(tempfile.gettempdir(), "ocr_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Checkpoint file path
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{document_id}.json"
        )
        
        # Load existing checkpoint if any
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint if available"""
        if not self.enable_checkpointing:
            return {"processed_pages": [], "last_updated": None, "page_data": {}}
            
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Loaded checkpoint for document {self.document_id}")
                
                # Validate checkpoint data has content
                if "page_data" not in checkpoint_data or not checkpoint_data.get("page_data"):
                    logger.warning(f"Checkpoint found but contains no page data, will reprocess")
                    return {"processed_pages": [], "last_updated": None, "page_data": {}}
                    
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                
        return {"processed_pages": [], "last_updated": None, "page_data": {}}
    
    def should_process_page(self, page_num: int) -> bool:
        """Check if a page should be processed based on checkpoint data"""
        if not self.enable_checkpointing:
            return True
            
        # Check if page is marked as processed AND has actual content
        if page_num in self.checkpoint_data["processed_pages"]:
            # Also verify we have actual content for this page
            if "page_data" in self.checkpoint_data and str(page_num) in self.checkpoint_data["page_data"]:
                return False
                
        return True
    
    def mark_page_completed(self, page_num: int, result: Dict[str, Any]) -> bool:
        """Mark a page as completed and save checkpoint if needed"""
        if not self.enable_checkpointing:
            return False
            
        # Add to processed pages
        if page_num not in self.checkpoint_data["processed_pages"]:
            self.checkpoint_data["processed_pages"].append(page_num)
        
        # Store the actual page data
        if "page_data" not in self.checkpoint_data:
            self.checkpoint_data["page_data"] = {}
            
        # Store essential page data (avoid storing large objects)
        page_data = {
            "markdown": result.get("markdown", ""),
            "ocr_confidence": result.get("ocr_confidence", 0.0),
        }
        self.checkpoint_data["page_data"][str(page_num)] = page_data
        
        # Update timestamp
        self.checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        # Save checkpoint periodically
        if len(self.checkpoint_data["processed_pages"]) % self.checkpoint_frequency == 0:
            return self._save_checkpoint()
            
        return True
    
    def _save_checkpoint(self) -> bool:
        """Save checkpoint to disk"""
        if not self.enable_checkpointing:
            return False
            
        try:            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, ensure_ascii=False)
            logger.info(f"Saved checkpoint for document {self.document_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            return False
    
    def finalize(self) -> bool:
        """Save final checkpoint and cleanup"""
        if not self.enable_checkpointing:
            return False
            
        success = self._save_checkpoint()
        return success

class QualityAssurance:
    """Quality assurance measures for OCR output"""
    
    def __init__(self):
        self.confidence_threshold = 0.7  # Minimum confidence to accept OCR result
        self.expected_patterns = {
            "account_number": r'\d{10,20}',  # Account numbers are 10-20 digits
            "date": r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Date patterns
            "amount": r'[\$£€]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?',  # Money amounts
        }
    
    def assess_confidence(self, text: str) -> float:
        """Assess overall confidence in OCR text quality"""
        # Calculate basic confidence metrics
        if not text:
            return 0.0
            
        # Check for common OCR errors
        error_indicators = [
            ('l', '1'), ('O', '0'), ('S', '5'),  # Common character confusions
            '', '#', '@', '*',  # Unrecognized characters
        ]
        
        error_count = sum(text.count(ind[0]) + text.count(ind[1]) 
                          if isinstance(ind, tuple) else text.count(ind) 
                          for ind in error_indicators)
                          
        # Calculate ratio of error indicators to text length
        error_ratio = error_count / max(len(text), 1)
        
        # Initial confidence based on error ratio
        confidence = 1.0 - min(error_ratio * 5, 0.9)  # Scale error impact
        
        # Check for expected patterns
        pattern_matches = 0
        for pattern_name, pattern in self.expected_patterns.items():
            if re.search(pattern, text):
                pattern_matches += 1
                
        # Boost confidence based on expected pattern matches
        if pattern_matches > 0:
            confidence = min(confidence + 0.1 * pattern_matches, 1.0)
            
        return confidence
    
    def validate_extraction(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against expected patterns and structure"""
        validation_results = {
            "overall_quality": "unknown",
            "confidence_score": 0.0,
            "issues": [],
            "warnings": []
        }
        
        # Check for empty or minimal data
        if not extracted_data:
            validation_results["overall_quality"] = "poor"
            validation_results["issues"].append("No data extracted")
            return validation_results
            
        # Calculate overall confidence
        confidence_scores = []
        
        # Check account details if present
        if "account_details" in extracted_data:
            account_details = extracted_data["account_details"]
            
            # Check account number
            if "account_number" in account_details:
                account_number = account_details["account_number"]
                if not re.match(self.expected_patterns["account_number"], account_number):
                    validation_results["warnings"].append("Account number format is unusual")
                    confidence_scores.append(0.5)
                else:
                    confidence_scores.append(0.9)
            
        # Check transactions if present
        if "account_statement" in extracted_data and "transactions" in extracted_data["account_statement"]:
            transactions = extracted_data["account_statement"]["transactions"]
            
            # Check if we have a reasonable number of transactions
            if len(transactions) < 2:
                validation_results["warnings"].append("Very few transactions detected")
                confidence_scores.append(0.6)
            else:
                confidence_scores.append(0.8)
                
            # Check transaction dates
            for idx, txn in enumerate(transactions):
                if "txn_date" in txn and not re.match(self.expected_patterns["date"], txn["txn_date"]):
                    validation_results["warnings"].append(f"Transaction {idx+1} has unusual date format")
        
        # Calculate overall confidence score
        if confidence_scores:
            validation_results["confidence_score"] = sum(confidence_scores) / len(confidence_scores)
            
            # Set overall quality based on confidence
            if validation_results["confidence_score"] > 0.8:
                validation_results["overall_quality"] = "good"
            elif validation_results["confidence_score"] > 0.6:
                validation_results["overall_quality"] = "acceptable"
            else:
                validation_results["overall_quality"] = "poor"
        
        return validation_results

class Perform_OCR_v2:
    def __init__(self, custom_config: Dict[str, Any] = None):
        # Initialize configuration
        self.config = OCRConfig()
        if custom_config:
            # Update config with custom values
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize support systems
        self.resource_monitor = ResourceMonitor(self.config)
        self.cache_manager = CacheManager(self.config)
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.quality_assurance = QualityAssurance()
        
        # Initialize pipeline options for docling - ensure use_gpu is False
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = self.config.enable_table_detection
        self.pipeline_options.table_structure_options.do_cell_matching = True

        # Configure OCR options - don't use unsupported parameters
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=self.config.force_full_page_ocr
            # Removed use_gpu parameter as it's not supported
        )
        self.pipeline_options.ocr_options = ocr_options

        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )

        # JSON schema for the expected output
        self.json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "description": "The page number of the statement"
                    },
                    "account_details": {
                        "type": "object",
                        "description": "Account holder information and summary data",
                        "properties": {
                            "account_name": { "type": "string" },
                            "address": { "type": "string" },
                            "date": { "type": "string" },
                            "account_number": { "type": "string" },
                            "account_description": { "type": "string" },
                            "branch": { "type": "string" },
                            "drawing_power": { "type": "string" },
                            "interest_rate": { "type": "string" },
                            "mod_balance": { "type": "string" },
                            "cif_no": { "type": "string" },
                            "ckyc_number": { "type": "string" },
                            "ifs_code": { "type": "string" },
                            "micr_code": { "type": "string" },
                            "nomination_registered": { "type": "string" },
                            "opening_balance": { "type": "string", "description": "Balance as of the start of the statement period" },
                            "period": { "type": "string", "description": "Statement period" }
                        }
                    },
                    "account_statement": {
                        "type": "object",
                        "description": "Statement period and transactions",
                        "properties": {
                            "transactions": {
                                "type": "array",
                                "description": "List of all transactions in this page",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "txn_date": { "type": "string" },
                                        "value_date": { "type": "string" },
                                        "description": { "type": "string" },
                                        "ref_no_cheque_no": { "type": "string" },
                                        "debit": { "type": "string" },
                                        "credit": { "type": "string" },
                                        "balance": { "type": "string" }
                                    },
                                    "required": ["txn_date", "value_date", "description"]
                                }
                            }
                        },
                        "required": ["transactions"]
                    }
                },
                "required": ["page_number", "account_statement"]
            }
        }

        # System prompt for JSON structuring
        self.system_prompt = """
        You are an OCR tool designed to extract all the data from PDF with page numbers and structure it in JSON format. 
        Be consistent with the parameters and variables in case of repetition. The output should be in the following format:
        [
        {
            "page_number": "",
            "account_details": {
                "account_name": "",
                "address": "",
                "date": "",
                "account_number": "",
                "account_description": "",
                "branch": "",
                "drawing_power": "",
                "interest_rate": "",
                "mod_balance": "",
                "cif_no": "",
                "ckyc_number": "",
                "ifs_code": "",
                "micr_code": "",
                "nomination_registered": "",
                "opening_balance": "",
                "period": ""
            },
            "account_statement": {
                "transactions": [
                    {
                        "txn_date": "",
                        "value_date": "",
                        "description": "",
                        "ref_no_cheque_no": "",
                        "debit": "",
                        "credit": "",
                        "balance": ""
                    }
                ]
            }
        }
        ]
        """

    def extract_element_boxes(self, doc):
        """Extract text and image element boxes from the document"""
        element_boxes = {
            "text_boxes": [],
            "image_boxes": []
        }

        # Iterate through all the items in the doc
        for item, level in doc.iterate_items():
            # Extract text boxes - including all text types
            if hasattr(item, 'label') and hasattr(item, 'text'):
                for prov in item.prov:
                    # Get the bounding box coordinates
                    bbox_tuple = prov.bbox.as_tuple()
                    element_boxes["text_boxes"].append({
                        "text": item.text,
                        "label": item.label,
                        "page": prov.page_no,
                        "bbox": {
                            "left": bbox_tuple[0],
                            "top": bbox_tuple[1],
                            "right": bbox_tuple[2],
                            "bottom": bbox_tuple[3]
                        }
                    })

            # Extract image boxes
            elif hasattr(item, 'label') and item.label in ['picture', 'chart']:
                for prov in item.prov:
                    # Get the bounding box coordinates
                    bbox_tuple = prov.bbox.as_tuple()
                    element_boxes["image_boxes"].append({
                        "page": prov.page_no,
                        "bbox": {
                            "left": bbox_tuple[0],
                            "top": bbox_tuple[1],
                            "right": bbox_tuple[2],
                            "bottom": bbox_tuple[3]
                        }
                    })
        
        return element_boxes
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        # Keeping this method for compatibility
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        return denoised

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_pdf_page(self, page_num: int, pdf_document: fitz.Document, dpi: int = 300) -> str:
        # Keeping this method for compatibility
        scale = dpi / 72
        
        with tempfile.TemporaryDirectory() as temp_dir:
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
    
            preprocessed = self.preprocess_image(img_np)
            temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
            cv2.imwrite(temp_image_path, preprocessed)
        
            with open(temp_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            logger.info(f"Processed PDF page {page_num + 1}")
            return base64_image

    def pdf_to_base64_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        # Keeping this method for compatibility
        logger.info(f"Converting PDF to base64 images: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        max_workers = min(total_pages, os.cpu_count() or 1)
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_pdf_page, page_num, pdf_document, dpi): page_num
                for page_num in range(total_pages)
            }
            
            # Initialize results list to maintain page order
            base64_images = [None] * total_pages
            
            # Collect results as they complete
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    base64_images[page_num] = future.result()
                except Exception as e:
                    logger.error(f"Error processing PDF page {page_num + 1}: {str(e)}")
                    base64_images[page_num] = None
                    
        # Remove any failed pages
        base64_images = [img for img in base64_images if img is not None]
        
        logger.info(f"Completed processing {len(base64_images)} PDF pages")
        return base64_images

    def process_single_page(self, base64_image: str, system_prompt: str) -> Dict[str, Any]:
        """Process a single page of the document."""
        try:
            return self.extract_invoice_data(base64_image, system_prompt)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for image: {e}")
            return {}

    def extract_invoice_data(self, base64_image):
        """Extract invoice data from a base64-encoded image."""
        logger.info(f"Extracting invoice data from image ")
        
        # Use OpenAI to convert the markdown to the expected JSON format
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the data from the image and output JSON."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            top_p=float(os.getenv("TOP_P", "0.9")),
        )
        logger.info(json.loads(response.choices[0].message.content))
        return json.loads(response.choices[0].message.content)

    def process_document_with_docling(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document using docling library with sequential OCR processing.
        Prioritizes quality over speed by processing pages sequentially.
        """
        logger.info(f"Processing document with docling (quality-focused mode): {file_path}")
        start_time = time.time()
        
        try:
            # Generate document ID for checkpointing and caching
            document_hash = hashlib.md5(file_path.encode()).hexdigest()
            document_id = f"{os.path.basename(file_path)}_{document_hash[:8]}"
            
            # Initialize checkpoint manager
            checkpoint_manager = CheckpointManager(self.config, document_id)
            
            # Generate config hash for caching
            config_dict = {k: v for k, v in vars(self.config).items() 
                          if not k.startswith('_') and k != 'cache_dir'}
            config_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]
            
            # First, extract PDF pages
            logger.info(f"Extracting pages from {file_path}")
            pdf = fitz.open(file_path)
            num_pages = len(pdf)
            logger.info(f"Document has {num_pages} pages")
            
            # Create temporary directory to store intermediate results
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process pages SEQUENTIALLY to avoid CUDA/tensor errors
                page_results = []
                checkpoint_loaded_pages = 0
                
                for page_num in range(num_pages):
                    # Check if we already processed this page with valid data
                    if not checkpoint_manager.should_process_page(page_num):
                        try:
                            # Get page data from checkpoint
                            page_data = checkpoint_manager.checkpoint_data.get("page_data", {}).get(str(page_num), {})
                            
                            if page_data and page_data.get("markdown"):
                                logger.info(f"Using checkpoint data for page {page_num + 1}")
                                checkpoint_loaded_pages += 1
                                
                                # Create a result dict with the checkpoint data
                                result = {
                                    "page_num": page_num,
                                    "markdown": page_data.get("markdown", ""),
                                    "ocr_confidence": page_data.get("ocr_confidence", 0.0),
                                    "checkpoint_restored": True
                                }
                                page_results.append(result)
                                continue
                            else:
                                logger.warning(f"Checkpoint for page {page_num + 1} exists but contains no markdown data, will reprocess")
                        except Exception as e:
                            logger.warning(f"Error loading checkpoint data for page {page_num + 1}: {e}")
                    
                    try:
                        logger.info(f"Processing page {page_num + 1}/{num_pages}")
                        
                        # Extract page to temporary image
                        page = pdf.load_page(page_num)
                        temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                        
                        # Get page data for caching - use pixmap bytes for caching
                        pixmap = page.get_pixmap()
                        page_bytes = pixmap.samples
                        cache_key = self.cache_manager.get_cache_key(page_bytes, config_hash)
                        
                        # Check cache first
                        cached_result = self.cache_manager.get_from_cache(cache_key)
                        if cached_result:
                            # Update page number in cached result
                            if "markdown" in cached_result:
                                page_markdown = cached_result["markdown"]
                                # Ensure the page number is correct
                                page_pattern = re.compile(r'#{1,2}\s+Page\s+\d+', re.IGNORECASE)
                                if page_pattern.search(page_markdown):
                                    page_markdown = page_pattern.sub(f"# Page {page_num + 1}", page_markdown)
                                else:
                                    page_markdown = f"# Page {page_num + 1}\n\n{page_markdown}"
                                cached_result["markdown"] = page_markdown
                            
                            logger.info(f"Using cached result for page {page_num + 1}")
                            checkpoint_manager.mark_page_completed(page_num, cached_result)
                            page_results.append(cached_result)
                            continue
                        
                        # Render page with high DPI for quality
                        pix = page.get_pixmap(matrix=fitz.Matrix(self.config.dpi/72, self.config.dpi/72))
                        pix.save(temp_image_path)
                        
                        # Skip preprocessing - use original image directly
                        # Create a small PDF with just this page for docling to process
                        temp_pdf_path = os.path.join(temp_dir, f"page_{page_num}.pdf")
                        temp_pdf = fitz.open()
                        temp_pdf.insert_pdf(pdf, from_page=page_num, to_page=page_num)
                        temp_pdf.save(temp_pdf_path)
                        temp_pdf.close()
                        
                        # Initialize docling options with minimal preprocessing
                        pipeline_options = PdfPipelineOptions()
                        pipeline_options.do_ocr = True
                        pipeline_options.do_table_structure = self.config.enable_table_detection
                        pipeline_options.table_structure_options.do_cell_matching = True
                        
                        # Configure OCR options with minimal settings
                        ocr_options = RapidOcrOptions(
                            force_full_page_ocr=self.config.force_full_page_ocr
                        )
                        pipeline_options.ocr_options = ocr_options
                        
                        # Try multiple OCR variants for highest quality
                        ocr_success = False
                        ocr_variants = [
                            {"force_full_page_ocr": True},
                            {"force_full_page_ocr": True, "det_limit_side_len": 2880},
                            {"force_full_page_ocr": False}
                        ]
                        
                        page_doc = None
                        for variant_idx, variant_options in enumerate(ocr_variants):
                            try:
                                logger.info(f"Trying OCR variant {variant_idx+1}/{len(ocr_variants)} for page {page_num + 1}")
                                # Create variant-specific options
                                variant_ocr_options = RapidOcrOptions(**variant_options)
                                variant_pipeline_options = PdfPipelineOptions()
                                variant_pipeline_options.do_ocr = True
                                variant_pipeline_options.do_table_structure = self.config.enable_table_detection
                                variant_pipeline_options.table_structure_options.do_cell_matching = True
                                variant_pipeline_options.ocr_options = variant_ocr_options
                                
                                # Create a dedicated converter for this variant
                                variant_converter = DocumentConverter(
                                    format_options={
                                        InputFormat.PDF: PdfFormatOption(
                                            pipeline_options=variant_pipeline_options,
                                        )
                                    }
                                )
                                
                                # Try converting with this variant
                                page_doc = variant_converter.convert(Path(temp_pdf_path)).document
                                ocr_success = True
                                break
                            except Exception as e:
                                logger.warning(f"OCR variant {variant_idx+1} failed for page {page_num + 1}: {str(e)}")
                        
                        # If all variants failed, try fallback engines
                        if not ocr_success and self.config.fallback_engines:
                            for fallback_engine in self.config.fallback_engines:
                                logger.info(f"Trying fallback OCR engine {fallback_engine} for page {page_num + 1}")
                                # Add implementation for fallback engines if available
                        
                        # If all OCR attempts failed, raise exception
                        if not ocr_success or page_doc is None:
                            raise Exception("All OCR engines and variants failed")
                        
                        # Extract element boxes and markdown
                        page_element_boxes = self.extract_element_boxes(page_doc)
                        page_markdown = page_doc.export_to_markdown()
                        
                        # Assess OCR quality
                        ocr_confidence = self.quality_assurance.assess_confidence(page_markdown)
                        logger.info(f"Page {page_num + 1} OCR confidence: {ocr_confidence:.2f}")
                        
                        # Ensure the page number is included in the markdown
                        if not page_markdown.strip().startswith(f"# Page {page_num + 1}"):
                            page_markdown = f"# Page {page_num + 1}\n\n{page_markdown}"
                        
                        result = {
                            "page_num": page_num,
                            "element_boxes": page_element_boxes,
                            "markdown": page_markdown,
                            "ocr_confidence": ocr_confidence
                        }
                        
                        # Cache the result
                        self.cache_manager.save_to_cache(cache_key, result)
                        
                        # Mark page as completed in checkpoint
                        checkpoint_manager.mark_page_completed(page_num, result)
                        page_results.append(result)
                        
                        # Log progress after each page
                        logger.info(f"OCR Progress: {len(page_results)}/{num_pages} pages ({len(page_results)/num_pages*100:.1f}%)")
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                        page_results.append({
                            "page_num": page_num,
                            "error": str(e)
                        })
                
                # Verify we have actual content if pages were loaded from checkpoint
                if checkpoint_loaded_pages > 0:
                    logger.info(f"Loaded {checkpoint_loaded_pages} pages from checkpoint")
                
                # Step 2: Combine results in sequential order
                logger.info("Combining OCR results in sequential order")
                all_element_boxes = {"text_boxes": [], "image_boxes": []}
                combined_markdown = ""
                
                # Check for any failed pages
                failed_pages = [result for result in page_results if "error" in result]
                if failed_pages:
                    logger.warning(f"{len(failed_pages)} pages failed OCR processing")
                
                # Combine results in original page order
                for page_result in page_results:
                    if page_result and "error" not in page_result and "markdown" in page_result:
                        # Add element boxes if available
                        if "element_boxes" in page_result:
                            all_element_boxes["text_boxes"].extend(page_result["element_boxes"]["text_boxes"])
                            all_element_boxes["image_boxes"].extend(page_result["element_boxes"]["image_boxes"])
                        
                        # Add markdown with separator
                        combined_markdown += page_result["markdown"] + "\n\n"
                
                # Verify the combined markdown has content
                if not combined_markdown.strip():
                    logger.error("No markdown content generated after combining results")
                    # Force reprocess by clearing checkpoint
                    checkpoint_manager.checkpoint_data = {"processed_pages": [], "last_updated": None, "page_data": {}}
                    checkpoint_manager._save_checkpoint()
                    raise Exception("Empty markdown content after checkpoint restoration, please retry")
                
                # Step 3: Save the combined results
                element_boxes_path = ""
                markdown_path = ""
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w', encoding='utf-8') as f_boxes:
                    json.dump(all_element_boxes, f_boxes, indent=2, ensure_ascii=False)
                    element_boxes_path = f_boxes.name
                    
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False, mode='w', encoding='utf-8') as f_md:
                    f_md.write(combined_markdown)
                    markdown_path = f_md.name
                
                pdf.close()
                
                # Finalize checkpoint
                checkpoint_manager.finalize()
                
                # Calculate processing stats
                total_time = time.time() - start_time
                pages_per_second = num_pages / max(total_time, 0.001)
                
                logger.info(f"Document processed successfully. Total processing time: {total_time:.2f} seconds")
                logger.info(f"Processing speed: {pages_per_second:.2f} pages/second")
                
                resource_stats = self.resource_monitor.get_resource_stats()
                logger.info(f"Resource usage - CPU: {resource_stats['cpu_percent']}%, Memory: {resource_stats['memory_percent']}%")
                
                return {
                    "element_boxes": all_element_boxes,
                    "markdown_content": combined_markdown,
                    "element_boxes_path": element_boxes_path,
                    "markdown_path": markdown_path,
                    "stats": {
                        "total_time": total_time,
                        "pages_per_second": pages_per_second,
                        "successful_pages": num_pages - len(failed_pages),
                        "failed_pages": len(failed_pages),
                        "checkpoint_loaded_pages": checkpoint_loaded_pages,
                        "resources": resource_stats
                    }
                }
        except Exception as e:
            logger.error(f"Error processing document with docling: {str(e)}")
            raise

    def markdown_to_json(self, markdown_content: str, system_prompt: str) -> Dict[str, Any]:
        """
        Convert markdown to JSON using OpenAI with improved retry logic and handling of large documents.
        Focus on comprehensive extraction of all data without arbitrary token limits.
        """
        logger.info("Converting markdown to JSON using OpenAI (quality-focused mode)")
        
        # Estimate tokens in markdown content (for logging only)
        estimated_tokens = len(markdown_content) // 4
        logger.info(f"Estimated token count: {estimated_tokens}")
        
        # Add improved retry logic with exponential backoff
        max_retries = 5
        retry_count = 0
        backoff_time = 1  # Initial backoff time in seconds
        
        while retry_count < max_retries:
            try:
                logger.info(f"JSON extraction attempt {retry_count + 1}/{max_retries}")
                
                # Clear instruction to ensure comprehensive extraction
                user_prompt = f"Extract ALL structured data from the following text into JSON. Include ALL account details, transactions, dates, and amounts. Make sure to capture EVERY transaction and data point.\n\n{markdown_content}"
                
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    # Don't set arbitrary max_tokens limit
                    temperature=0.1,  # Lower temperature for more deterministic output
                    top_p=0.95,
                )
                
                try:
                    # Parse response
                    parsed_response = json.loads(response.choices[0].message.content)
                    
                    # Validate result with QA
                    validation_result = self.quality_assurance.validate_extraction(parsed_response)
                    
                    # Log validation results
                    logger.info(f"Extraction quality (attempt {retry_count + 1}): {validation_result['overall_quality']} (confidence: {validation_result['confidence_score']:.2f})")
                    if validation_result["warnings"]:
                        logger.warning(f"Extraction warnings: {', '.join(validation_result['warnings'])}")
                    if validation_result["issues"]:
                        logger.error(f"Extraction issues: {', '.join(validation_result['issues'])}")
                    
                    # Add validation info to response
                    if isinstance(parsed_response, dict):
                        parsed_response["_quality"] = {"score": validation_result['confidence_score']}
                    elif isinstance(parsed_response, list) and parsed_response and isinstance(parsed_response[0], dict):
                        parsed_response[0]["_quality"] = {"score": validation_result['confidence_score']}
                    
                    logger.info("Successfully parsed JSON response")
                    return parsed_response
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response on attempt {retry_count+1}")
                
                retry_count += 1
                # Only sleep if we're going to retry
                if retry_count < max_retries:
                    # Exponential backoff
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                
            except Exception as e:
                logger.error(f"API call failed on attempt {retry_count+1}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff
                    time.sleep(backoff_time)
                    backoff_time *= 2
        
        # If all attempts failed, return empty result with error
        logger.error("All extraction attempts failed")
        return {"error": "Failed to extract structured data after multiple attempts", "page_number": 1}

    def split_markdown_into_sections(self, markdown_content: str) -> List[str]:
        """
        Split markdown content into logical sections based on content structure.
        Tries to identify natural document sections like pages, account information, 
        and transaction blocks.
        """
        logger.info("Splitting markdown into logical sections")
        
        # Try to split by page markers first
        sections = []
        page_pattern = re.compile(r'#{1,2}\s+Page\s+(\d+)', re.IGNORECASE)
        
        # Find all page markers
        matches = list(page_pattern.finditer(markdown_content))
        
        if len(matches) > 1:
            # If we have page markers, split by them
            logger.info(f"Found {len(matches)} page markers")
            for i in range(len(matches)):
                start = matches[i].start()
                # If it's the last match, go to the end of the content
                end = matches[i+1].start() if i < len(matches) - 1 else len(markdown_content)
                section = markdown_content[start:end].strip()
                if section:
                    sections.append(section)
        else:
            # If no page markers, try to split by content boundaries
            logger.info("No page markers found, splitting by content boundaries")
            
            # Look for transaction tables or account sections
            transaction_patterns = [
                r'#{1,3}\s+Transactions', 
                r'#{1,3}\s+Statement', 
                r'(Date|Transaction Date|Value Date)[\s\|]+(Description|Particulars)[\s\|]+(Amount|Debit|Credit)[\s\|]+Balance'
            ]
            
            account_patterns = [
                r'#{1,3}\s+Account\s+Details',
                r'#{1,3}\s+Customer\s+Information',
                r'Account Number\s*:',
                r'Branch\s*:'
            ]
            
            # Combine all patterns to find section boundaries
            all_patterns = transaction_patterns + account_patterns
            
            # Build a list of potential section start positions
            section_positions = []
            for pattern in all_patterns:
                for match in re.finditer(pattern, markdown_content, re.IGNORECASE | re.MULTILINE):
                    section_positions.append(match.start())
            
            # Sort positions and add document start/end
            section_positions = sorted(list(set([0] + section_positions + [len(markdown_content)])))
            
            if len(section_positions) > 1:
                # Create sections from positions
                for i in range(len(section_positions) - 1):
                    start = section_positions[i]
                    end = section_positions[i + 1]
                    section = markdown_content[start:end].strip()
                    if section:
                        sections.append(section)
            else:
                # If no clear sections, try splitting by blank lines
                content_sections = re.split(r'\n{3,}', markdown_content)
                sections.extend([s.strip() for s in content_sections if s.strip()])
        
        # Ensure we have at least one section
        if not sections:
            sections = [markdown_content]
        
        # If sections are still very large, further split them based on size
        processed_sections = []
        max_chars_per_section = 30000  # ~7500 tokens
        
        for section in sections:
            if len(section) > max_chars_per_section:
                # Split large sections by paragraphs
                paragraphs = re.split(r'\n{2,}', section)
                
                current_section = ""
                current_size = 0
                
                for para in paragraphs:
                    para_size = len(para)
                    
                    if current_size + para_size > max_chars_per_section:
                        # Save current section if not empty
                        if current_section:
                            processed_sections.append(current_section.strip())
                        
                        # Start new section
                        current_section = para
                        current_size = para_size
                    else:
                        # Add to current section
                        if current_section:
                            current_section += "\n\n" + para
                        else:
                            current_section = para
                        current_size += para_size
                
                # Add any remaining content
                if current_section:
                    processed_sections.append(current_section.strip())
            else:
                # Section is small enough to process as-is
                processed_sections.append(section)
        
        # Add page number context to each section if not present
        for i, section in enumerate(processed_sections):
            if not page_pattern.search(section):
                processed_sections[i] = f"# Page/Section {i+1}\n\n{section}"
        
        logger.info(f"Split markdown into {len(processed_sections)} logical sections")
        return processed_sections

    def process_sections_in_batches(self, sections: List[str]) -> List[Dict[str, Any]]:
        """
        Process sections in batches with optimal parallelization.
        Similar to CUDA batch processing with proper resource management.
        """
        logger.info(f"Processing {len(sections)} sections in batches")
        
        # Determine optimal batch size based on available resources
        cpu_count = os.cpu_count() or 4
        batch_size = min(cpu_count * 2, len(sections))  # 2 tasks per CPU core is often optimal
        
        # Create batches of sections
        batches = [sections[i:i + batch_size] for i in range(0, len(sections), batch_size)]
        logger.info(f"Created {len(batches)} batches with batch size {batch_size}")
        
        all_results = []
        
        # Process each batch with progress tracking
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            batch_results = []
            
            # Use ThreadPoolExecutor for parallel processing within the batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Submit tasks
                futures = {
                    executor.submit(self.process_section_with_retry, section): (idx, section) 
                    for idx, section in enumerate(batch)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    idx, section = futures[future]
                    try:
                        result = future.result()
                        # Store result with its original index for proper ordering
                        batch_results.append((idx, result))
                        logger.info(f"Successfully processed section {batch_idx * batch_size + idx + 1}/{len(sections)}")
                    except Exception as e:
                        logger.error(f"Error processing section {batch_idx * batch_size + idx + 1}: {str(e)}")
                        # Add an error placeholder with original index
                        batch_results.append((idx, {"error": str(e), "page_number": batch_idx * batch_size + idx + 1}))
            
            # Sort batch results by their original index to maintain order
            batch_results.sort(key=lambda x: x[0])
            # Extract just the results (without indices)
            all_results.extend([result for _, result in batch_results])
        
        logger.info(f"Completed processing all {len(sections)} sections")
        return all_results

    def consolidate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate results from multiple API calls into a single JSON structure.
        Ensures proper sequence reconstruction and handles page numbers.
        """
        logger.info(f"Consolidating {len(results)} results")
        
        # Filter out any error results
        valid_results = [result for result in results if 'error' not in result]
        
        # Check if we have any valid results
        if not valid_results:
            logger.error("No valid results to consolidate")
            return [{"error": "No valid results were extracted", "page_number": 1}]
        
        # Normalize result structure
        normalized_results = []
        for result in valid_results:
            # Handle both single object and array results
            if isinstance(result, list):
                normalized_results.extend(result)
            else:
                normalized_results.append(result)
        
        # Sort by page number if available
        normalized_results.sort(key=lambda x: int(x.get('page_number', 0)) if isinstance(x.get('page_number'), (int, str)) and str(x.get('page_number', '')).isdigit() else 0)
        
        # Ensure each result has a page number
        for i, result in enumerate(normalized_results):
            if 'page_number' not in result or not str(result['page_number']).isdigit():
                result['page_number'] = i + 1
        
        logger.info(f"Consolidated into {len(normalized_results)} final results")
        return normalized_results

    def process_section_with_retry(self, section: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Process a section with OpenAI API, with retry logic and error handling for robustness.
        """
        retry_count = 0
        backoff_time = 1  # Initial backoff time in seconds
        
        while retry_count < max_retries:
            try:
                return self.markdown_to_json(section, self.system_prompt)
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to process section after {max_retries} retries: {str(e)}")
                    raise
                
                logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                # Exponential backoff
                time.sleep(backoff_time)
                backoff_time *= 2

    def main_extract(self, read_path: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Main extraction function using docling for OCR and OpenAI for JSON structuring.
        Prioritizes quality and comprehensive extraction over speed with improved content handling.
        
        Args:
            read_path: Path to the document to process
            retry_count: Internal parameter for tracking retries on checkpoint issues
        """
        logger.info(f"Starting high-quality extraction for: {read_path}")
        
        # Record start time for overall process
        overall_start_time = time.time()
        
        # Create output directory for permanent files
        base_dir = os.path.dirname(read_path)
        output_dir = os.path.join(base_dir, "extraction_output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory at: {output_dir}")
        
        # Process file based on extension
        file_extension = os.path.splitext(read_path)[1].lower()
        
        # Max retries for checkpoint issues (avoid infinite loops)
        max_retries = 2
        
        if file_extension in ['.jpg', '.jpeg', '.png', '.pdf']:
            try:
                # Process the document with docling in quality-focused mode
                logger.info("Phase 1: OCR processing with docling (quality mode)")
                doc_result = self.process_document_with_docling(read_path)
                
                # Save markdown to a permanent file
                base_name = os.path.splitext(os.path.basename(read_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                markdown_output_path = f"{output_dir}/{base_name}_extracted_{timestamp}.md"
                with open(markdown_output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(doc_result["markdown_content"])
                  # Save element boxes to permanent file
                element_boxes_output_path = f"{output_dir}/{base_name}_element_boxes_{timestamp}.json"
                with open(element_boxes_output_path, "w", encoding="utf-8") as boxes_file:
                    json.dump(doc_result["element_boxes"], boxes_file, indent=2, ensure_ascii=False)
                
                # Log the file paths
                logger.info(f"Temporary markdown file: {doc_result['markdown_path']}")
                logger.info(f"Permanent markdown file saved to: {markdown_output_path}")
                logger.info(f"Element boxes saved to: {element_boxes_output_path}")
                
                # Check if markdown is empty or minimal
                if not doc_result["markdown_content"].strip() or len(doc_result["markdown_content"]) < 50:
                    logger.error("OCR produced minimal or empty markdown content")
                    
                    # If we're dealing with a checkpoint issue, try again after clearing checkpoint
                    if "checkpoint_loaded_pages" in doc_result.get("stats", {}) and retry_count < max_retries:
                        # Clear checkpoint and retry
                        logger.warning(f"Attempting to clear checkpoint and retry (attempt {retry_count+1}/{max_retries})")
                        
                        # Generate the same document ID
                        document_hash = hashlib.md5(read_path.encode()).hexdigest()
                        document_id = f"{os.path.basename(read_path)}_{document_hash[:8]}"
                        
                        # Remove checkpoint file
                        checkpoint_dir = os.path.join(tempfile.gettempdir(), "ocr_checkpoints")
                        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{document_id}.json")
                        if os.path.exists(checkpoint_file):
                            try:
                                os.remove(checkpoint_file)
                                logger.info(f"Removed checkpoint file: {checkpoint_file}")
                            except Exception as e:
                                logger.warning(f"Failed to remove checkpoint file: {e}")
                        
                        # Retry with incremented retry count
                        return self.main_extract(read_path, retry_count + 1)
                    
                    return {
                        "error": "OCR produced minimal or empty content. The document may not be readable or may have security restrictions.",
                        "markdown_path": markdown_output_path,
                        "element_boxes_path": element_boxes_output_path
                    }
                
                # Only proceed with JSON extraction if we have content
                logger.info("Phase 2: Extracting structured data with OpenAI")
                batch_start_time = time.time()
                
                # Estimated token count (rough estimate: 4 chars ~= 1 token)
                estimated_tokens = len(doc_result["markdown_content"]) // 4
                logger.info(f"Estimated token count: {estimated_tokens}")
                
                # For large documents, split into logical sections based on content
                if estimated_tokens > 8000:  # Increased this threshold as models can handle more content
                    logger.info(f"Large document detected, splitting into logical sections")
                    
                    # Split the document by logical sections
                    sections = self.split_markdown_into_sections(doc_result["markdown_content"])
                    logger.info(f"Split document into {len(sections)} logical sections")
                    
                    # Process each section individually for best quality
                    results = []
                    for i, section in enumerate(sections):
                        logger.info(f"Processing section {i+1}/{len(sections)}")
                        # Skip sections that are too small to be meaningful
                        if len(section.strip()) < 50:
                            continue
                            
                        try:
                            section_result = self.markdown_to_json(section, self.system_prompt)
                            results.append(section_result)
                            logger.info(f"Successfully processed section {i+1}/{len(sections)}")
                        except Exception as e:
                            logger.error(f"Error processing section {i+1}: {str(e)}")
                            results.append({
                                "error": f"Failed to process section: {str(e)}",
                                "page_number": i+1
                            })
                            
                    # Merge results
                    logger.info(f"Merging {len(results)} section results")
                    structured_data = self.consolidate_results(results)
                else:
                    # Process entire content if it's small enough
                    structured_data = self.markdown_to_json(doc_result["markdown_content"], self.system_prompt)
                
                batch_time = time.time() - batch_start_time
                logger.info(f"JSON extraction completed in {batch_time:.2f} seconds")
                  # Save JSON output to permanent file
                json_output_path = f"{output_dir}/{base_name}_structured_{timestamp}.json"
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(structured_data, json_file, indent=2, ensure_ascii=False)
                
                # Calculate total processing time
                total_time = time.time() - overall_start_time
                logger.info(f"Total processing time: {total_time:.2f} seconds")
                
                # Return results with paths and statistics
                return {
                    "results": structured_data,
                    "markdown_path": markdown_output_path,
                    "element_boxes_path": element_boxes_output_path,
                    "json_path": json_output_path,
                    "output_dir": output_dir,
                    "stats": {
                        "total_time": total_time,
                        "ocr_stats": doc_result.get("stats", {}),
                        "extraction_time": batch_time,
                        "sections_processed": len(sections) if estimated_tokens > 8000 else 1,
                    }
                }
                
            except Exception as e:
                logger.error(f"Error processing file {read_path}: {str(e)}")
                
                # Check if this is a checkpoint-related error and we should retry
                if "checkpoint" in str(e).lower() and retry_count < max_retries:
                    # Generate the same document ID
                    document_hash = hashlib.md5(read_path.encode()).hexdigest()
                    document_id = f"{os.path.basename(read_path)}_{document_hash[:8]}"
                    
                    # Remove checkpoint file and retry
                    checkpoint_dir = os.path.join(tempfile.gettempdir(), "ocr_checkpoints")
                    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{document_id}.json")
                    if os.path.exists(checkpoint_file):
                        try:
                            os.remove(checkpoint_file)
                            logger.info(f"Removed checkpoint file due to error: {checkpoint_file}")
                        except Exception as e2:
                            logger.warning(f"Failed to remove checkpoint file: {e2}")
                    
                    logger.info(f"Retrying without checkpoint (attempt {retry_count+1}/{max_retries})")
                    return self.main_extract(read_path, retry_count + 1)
                
                return {"error": f"Failed to process document: {str(e)}", "page_number": 1}
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return {"error": f"Unsupported file type: {file_extension}", "page_number": 1}

# Note: Configure timeout settings in the WSGI server (e.g., Gunicorn) for long-running tasks.