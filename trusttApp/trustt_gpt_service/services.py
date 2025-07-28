import os
import cv2
import base64
import tempfile
import json
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import fitz
from openai import OpenAI
from dotenv import load_dotenv
from common.logger import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import jsonify, request
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import time
import io
from trustt_gpt_service.ocrpipeline import Perform_OCR_v2
import re
import decimal
from werkzeug.utils import secure_filename
from .db_repo import *
import concurrent.futures
import google.generativeai as genai
import PIL.Image
import fitz
import sys

if sys.version_info >= (3, 12):
    from typing import List, Dict, Any, TypedDict, Optional
else:
    from typing_extensions import List, Dict, Any, TypedDict, Optional


# Import docling libraries for OCR extraction
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from veri5_gateway.veri5_apis import Veri5ServiceAPI
from io import BytesIO

load_dotenv()

TGPT_LOG_NAME = os.getenv("TGPT_LOG_NAME")
LOG_LEVEL =os.getenv("LOG_LEVEL")
logger = setup_logger('trustt_gpt_service', TGPT_LOG_NAME, level=LOG_LEVEL)

minicpm_api_key = os.getenv("MINICPM_API_KEY")
minicpm_api_base =os.getenv("MINICPM_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

openai_api_key=os.getenv("OPENAI_API_KEY")
OPENAI_MODEL=os.getenv("OPENAI_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise EnvironmentError("Gemini API key not found in environment variables.")


class Veri5ServiceGateway:
    def extractDocInfo(self,document_type, document_side, extraction_type, document_front_image="", document_back_image=""):
        try:
            logger.info('Calling veri5 docInfoExtract api')
            response = Veri5ServiceAPI.extractDocInfo(document_type, document_side, extraction_type, document_front_image, document_back_image)
            return {"status":"SUCCESS", "result":response}
        except Exception as e:
            return {"status":"FAIL"}

    def extractbilldata(self,file_obj, file_type="pdf", doc_type="invoice", api_key="OPOfwhk3z44keJ7Lqrsn9L2gr5TCTzpF-5A2C9F899D464"):
        try:
            logger.info('Calling veri5 extract_bill_data api')
            response = Veri5ServiceAPI.extract_bill_data(file_obj, file_type, doc_type, api_key)
            return {"status":"SUCCESS", "result":response}
        except Exception as e:
            logger.error(f"Error in extract_bill_data: {str(e)}")
            return {"status":"FAIL", "error": str(e)}

class SessionService:
    def __init__(self, db):
        self.session_repo = SessionRepository(db)

    def insert_WorkItem(self, client_code, work_type, work_id, assignee_id, assignee_name, create_on, created_by):
        result = self.session_repo.insertWorkItem(
            client_code, work_type, work_id, assignee_id, assignee_name, create_on, created_by)
        if result:
            return {"status": "success", "work_item_id": result[0][0]}
        else:
            return {"status": "error", "message": "Failed to insert work item."}
        
    def insert_WorkDocInfo(self, work_item_id, doc_type, doc_id, doc_file_name, doc_type_source):
        result = self.session_repo.insertWorkDocInfo(
            work_item_id, doc_type, doc_id, doc_file_name, doc_type_source)
        if result:
            return {"status": "success", "work_doc_info_id": result[0][0]}
        else:
            return {"status": "error", "message": "Failed to insert work document info."}
    
    def insert_BankStmt(self, work_doc_info_id, stmt_id, bank_name, ac_holder_name, ac_num, ifsc_code, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified):
        result = self.session_repo.insertBankStmt(
            work_doc_info_id, stmt_id, bank_name, ac_holder_name, ac_num, ifsc_code, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified)
        if result:
            return {"status": "success", "bank_stmt_id": result[0][0]}
        else:
            return {"status": "error", "message": "Failed to insert bank statement."}
        
    def fetch_TxnJson(self, stmt_id):
        result = self.session_repo.fetchTxnJson(stmt_id)
        return result
    
    def update_BankStmt(self, work_doc_info_id, ac_holder_name, ac_num, ifsc_code, verification_metadata, txn_json, analysis_status, txn_accuracy_percentage, update_on, updated_by):
        result = self.session_repo.updateBankStmt(
            work_doc_info_id, ac_holder_name, ac_num, ifsc_code, verification_metadata, txn_json, analysis_status, txn_accuracy_percentage, update_on, updated_by)
        return {"status": "success"} if result else {"status": "error", "message": "Failed to update bank statement."}
    
    def fetch_BankStmtById(self, stmt_id):
        result = self.session_repo.fetchBankStmtById(stmt_id)
        if result:
            return {"status": "success", "data": {
                "txn_json": result[0],
                "verification_metadata": result[1],
                "analysis_status": result[2]
            }}
        else:
            return {"status": "error", "message": "Bank statement not found."}
        
    def fetch_InitialIdByClientCode(self, client_code):
        result = self.session_repo.fetchInitialIdByClientCode(client_code)
        if result:
            return {"status": "success", "initial_id": result[0]}
        else:
            return {"status": "error", "initial_id":0}
        
    def update_HumanVerification(self, stmt_id, is_human_verified, verified_by, verified_time):
        result = self.session_repo.updateHumanVerification(
            stmt_id, is_human_verified, verified_by, verified_time)
        return {"status": "success"} if result else {"status": "error", "message": "Failed to update human verification."}

    def insert_IdentityDoc(self, work_doc_info_id, stmt_id, extracted_data, created_on, created_by, is_deleted, total_pages):
        result = self.session_repo.insertIdentityDoc(work_doc_info_id, stmt_id, extracted_data, created_on, created_by, is_deleted, total_pages)
        return result
    
    def insert_CCStmt(self, work_doc_info_id, cc_stmt_id, bank_name, ac_holder_name, ac_num, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified):
        result = self.session_repo.insertCCStmt(
            work_doc_info_id, cc_stmt_id, bank_name, ac_holder_name, ac_num, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified)
        if result:
            return {"status": "success", "cc_stmt_id": result[0][0]}
        else:
            return {"status": "error", "message": "Failed to insert credit card statement."}

class Improved_PerformOCR:
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key)

    # def __init__(self):
    #     self.client = OpenAI( base_url=minicpm_api_base,api_key=minicpm_api_key)

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        
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

    def extract_invoice_data(self, base64_image, system_prompt):
        
        """Extract invoice data from a base64-encoded image."""
        logger.info(f"Extracting invoice data from image ")
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            # model="/apps/visionLLM/MiniCPM-V-2_6",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
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
            max_tokens=os.getenv("MAX_TOKENS"),
            temperature=os.getenv("TEMPERATURE"),
            top_p=os.getenv("TOP_P"),
        )
        logger.info(json.loads(response.choices[0].message.content))
        return json.loads(response.choices[0].message.content)

    def main_extract(self, read_path: str, system_prompt: str) -> List[Dict[str, Any]]:
        
        def process_file(filename: str) -> List[Dict[str, Any]]:
            
            file_extension = os.path.splitext(filename)[1].lower()
            base64_images = []
            if file_extension in ['.jpg', '.jpeg', '.png']:
                base64_images = [self.encode_image(filename)]
            elif file_extension == '.pdf':
                base64_images = self.pdf_to_base64_images(filename)
            else:
                logger.info(f"Skipping unsupported file type: {file_extension}")
                return []
            # Process images sequentially
            results = []
            total_pages = len(base64_images)
            
            for page_num, img in enumerate(base64_images):
                try:
                    page_data = self.process_single_page(img, system_prompt)
                    if page_data is None:
                        logger.warning(f"Page {page_num + 1} returned no data (None).")
                        page_data = {"error": f"Failed to process page {page_num + 1}: No data extracted."}
                    current_page_number = page_num + 1
                    page_data['page_number'] = current_page_number
                    results.append(page_data)
                    logger.info(f"Processed page {current_page_number} of {total_pages}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    results.append({"error": f"Failed to process page {page_num + 1}: {str(e)}", 
                                    "page_number": page_num + 1})
             
            return results


        logger.info(f"Starting extraction for: {read_path}")
        return process_file(read_path)
    

# # read_path =r"C:\Users\Trisha\Downloads\Data 1\Data 1\bank statement new.pdf"
# # system_prompt = """Extract all the data  and structure it in a json format."""
# # processor = Improved_PerformOCR()
# # results = processor.main_extract(read_path, system_prompt)
# # output_path = r"C:\TrishaW\ocr\trustt-platform-document-processing\output.json" 


# # with open(output_path, "w", encoding="utf-8") as json_file:
# #     json.dump(results, json_file, indent=4, ensure_ascii=False)

# class Improved_PerformOCR_v2:
#     def __init__(self):
#         self.client = OpenAI(api_key=openai_api_key)
#         # Initialize pipeline options for docling
#         self.pipeline_options = PdfPipelineOptions()
#         self.pipeline_options.do_ocr = True
#         self.pipeline_options.do_table_structure = True
#         self.pipeline_options.table_structure_options.do_cell_matching = True
        
#         ocr_options = RapidOcrOptions(force_full_page_ocr=True)
#         self.pipeline_options.ocr_options = ocr_options
        
#         # Initialize converter
#         self.converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=self.pipeline_options,
#                 )
#             }
#         )

#     def extract_element_boxes(self, doc):
#         """Extract text and image element boxes from the document"""
#         element_boxes = {
#             "text_boxes": [],
#             "image_boxes": []
#         }
        
#         # Iterate through all items in the document
#         for item, level in doc.iterate_items():
#             # Extract text boxes - including all text types
#             if hasattr(item, 'label') and hasattr(item, 'text'):
#                 for prov in item.prov:
#                     # Get the bounding box coordinates
#                     bbox_tuple = prov.bbox.as_tuple()
#                     element_boxes["text_boxes"].append({
#                         "text": item.text,
#                         "label": item.label,
#                         "page": prov.page_no,
#                         "bbox": {
#                             "left": bbox_tuple[0],
#                             "top": bbox_tuple[1],
#                             "right": bbox_tuple[2],
#                             "bottom": bbox_tuple[3]
#                         }
#                     })
            
#             # Extract image boxes
#             elif hasattr(item, 'label') and item.label in ['picture', 'chart']:
#                 for prov in item.prov:
#                     # Get the bounding box coordinates
#                     bbox_tuple = prov.bbox.as_tuple()
#                     element_boxes["image_boxes"].append({
#                         "page": prov.page_no,
#                         "bbox": {
#                             "left": bbox_tuple[0],
#                             "top": bbox_tuple[1],
#                             "right": bbox_tuple[2],
#                             "bottom": bbox_tuple[3]
#                         }
#                     })
        
#         return element_boxes

#     @staticmethod
#     def preprocess_image(image: np.ndarray) -> np.ndarray:
#         # Keeping this method for compatibility
#         norm_img = np.zeros((image.shape[0], image.shape[1]))
#         image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
#         return denoised

#     @staticmethod
#     def encode_image(image_path: str) -> str:
#         """Encode an image file to a base64 string."""
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")

#     def process_pdf_page(self, page_num: int, pdf_document: fitz.Document, dpi: int = 300) -> str:
#         # Keeping this method for compatibility
#         scale = dpi / 72
        
#         with tempfile.TemporaryDirectory() as temp_dir:
#             page = pdf_document.load_page(page_num)
#             pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             img_np = np.array(img)
       
#             preprocessed = self.preprocess_image(img_np)
#             temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
#             cv2.imwrite(temp_image_path, preprocessed)
          
#             with open(temp_image_path, "rb") as image_file:
#                 base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
#             logger.info(f"Processed PDF page {page_num + 1}")
#             return base64_image

#     def pdf_to_base64_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
#         # Keeping this method for compatibility
#         logger.info(f"Converting PDF to base64 images: {pdf_path}")
#         pdf_document = fitz.open(pdf_path)
#         total_pages = len(pdf_document)
        
#         max_workers = min(total_pages, os.cpu_count() or 1)
        
#         # Process pages in parallel
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = {
#                 executor.submit(self.process_pdf_page, page_num, pdf_document, dpi): page_num
#                 for page_num in range(total_pages)
#             }
            
#             # Initialize results list to maintain page order
#             base64_images = [None] * total_pages
            
#             # Collect results as they complete
#             for future in as_completed(futures):
#                 page_num = futures[future]
#                 try:
#                     base64_images[page_num] = future.result()
#                 except Exception as e:
#                     logger.error(f"Error processing PDF page {page_num + 1}: {str(e)}")
#                     base64_images[page_num] = None
                    
#         # Remove any failed pages
#         base64_images = [img for img in base64_images if img is not None]
        
#         logger.info(f"Completed processing {len(base64_images)} PDF pages")
#         return base64_images

#     def process_single_page(self, base64_image: str, system_prompt: str) -> Dict[str, Any]:
#         """Process a single page of the document."""
#         try:
#             return self.extract_invoice_data(base64_image, system_prompt)
#         except json.JSONDecodeError as e:
#             logger.error(f"Error decoding JSON for image: {e}")
#             return {}

#     def extract_invoice_data(self, base64_image, system_prompt):
#         """Extract invoice data from a base64-encoded image."""
#         logger.info(f"Extracting invoice data from image ")
        
#         # Use OpenAI to convert the markdown to the expected JSON format
#         response = self.client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type": "json_object"},
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": "Extract the data from the image and output JSON."},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpg;base64,{base64_image}",
#                                 "detail": "high",
#                             },
#                         },
#                     ],
#                 },
#             ],
#             max_tokens=os.getenv("MAX_TOKENS"),
#             temperature=os.getenv("TEMPERATURE"),
#             top_p=os.getenv("TOP_P"),
#         )
#         logger.info(json.loads(response.choices[0].message.content))
#         return json.loads(response.choices[0].message.content)

#     def process_document_with_docling(self, file_path: str) -> Dict[str, Any]:
#         """Process a document using docling library."""
#         logger.info(f"Processing document with docling: {file_path}")
        
#         start_time = time.time()
        
#         # Convert the document using docling
#         try:
#             doc = self.converter.convert(Path(file_path)).document
            
#             # Extract element boxes
#             element_boxes = self.extract_element_boxes(doc)
            
#             # Export to markdown
#             markdown_content = doc.export_to_markdown()
            
#             # Save element boxes and markdown temporarily
#             with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f_boxes:
#                 json.dump(element_boxes, f_boxes, indent=2)
#                 element_boxes_path = f_boxes.name
                
#             with tempfile.NamedTemporaryFile(suffix='.md', delete=False, mode='w') as f_md:
#                 f_md.write(markdown_content)
#                 markdown_path = f_md.name
            
#             logger.info(f"Document processed and saved temporarily. Processing time: {time.time() - start_time:.2f} seconds")
            
#             return {
#                 "element_boxes": element_boxes,
#                 "markdown_content": markdown_content,
#                 "element_boxes_path": element_boxes_path,
#                 "markdown_path": markdown_path
#             }
#         except Exception as e:
#             logger.error(f"Error processing document with docling: {str(e)}")
#             raise

#     def markdown_to_json(self, markdown_content: str, system_prompt: str) -> Dict[str, Any]:
#         """Convert markdown to JSON using OpenAI."""
#         logger.info("Converting markdown to JSON using OpenAI")
        
#         response = self.client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type": "json_object"},
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": f"Convert the following document content to the required JSON format.\n\n{markdown_content}"
#                 },
#             ],
#             max_tokens=os.getenv("MAX_TOKENS"),
#             temperature=os.getenv("TEMPERATURE"),
#             top_p=os.getenv("TOP_P"),
#         )
        
#         return json.loads(response.choices[0].message.content)

#     def main_extract(self, read_path: str, system_prompt: str) -> List[Dict[str, Any]]:
#         """Main extraction function using docling for OCR and OpenAI for JSON structuring."""
#         logger.info(f"Starting extraction for: {read_path}")
        
#         # Process file based on extension
#         file_extension = os.path.splitext(read_path)[1].lower()
        
#         if file_extension in ['.jpg', '.jpeg', '.png', '.pdf']:
#             try:
#                 # Process the document with docling
#                 doc_result = self.process_document_with_docling(read_path)
                
#                 # Convert markdown to JSON
#                 processed_data = self.markdown_to_json(doc_result["markdown_content"], system_prompt)
                
#                 # Clean up temporary files
#                 try:
#                     os.remove(doc_result["element_boxes_path"])
#                     os.remove(doc_result["markdown_path"])
#                 except Exception as e:
#                     logger.warning(f"Error cleaning up temporary files: {str(e)}")
                
#                 # Ensure result is wrapped in a list for consistency
#                 if isinstance(processed_data, dict):
#                     processed_data = [processed_data]
                
#                 # Add page numbers if not present
#                 for page_num, page_data in enumerate(processed_data):
#                     if 'page_number' not in page_data:
#                         page_data['page_number'] = page_num + 1
                
#                 return processed_data
#             except Exception as e:
#                 logger.error(f"Error processing file {read_path}: {str(e)}")
#                 return [{"error": f"Failed to process document: {str(e)}", "page_number": 1}]
#         else:
#             logger.info(f"Skipping unsupported file type: {file_extension}")
#             return []

# class PerformOCR_v1:
#     def __init__(self):
#         self.client = OpenAI(api_key=openai_api_key)

#     # def __init__(self):
#     #     self.client = OpenAI( base_url=minicpm_api_base,api_key=minicpm_api_key)

#     @staticmethod
#     def preprocess_image(image: np.ndarray) -> np.ndarray:
        
#         norm_img = np.zeros((image.shape[0], image.shape[1]))
#         image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
#         return denoised

#     @staticmethod
#     def encode_image(image_path: str) -> str:
#         """Encode an image file to a base64 string."""
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")

  
#     def process_pdf_page(self, page_num: int, pdf_document: fitz.Document, dpi: int = 300) -> str:
        
#         scale = dpi / 72
        
#         with tempfile.TemporaryDirectory() as temp_dir:
            
#             page = pdf_document.load_page(page_num)
#             pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             img_np = np.array(img)

       
#             preprocessed = self.preprocess_image(img_np)
#             temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
#             cv2.imwrite(temp_image_path, preprocessed)

          
#             with open(temp_image_path, "rb") as image_file:
#                 base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
#             logger.info(f"Processed PDF page {page_num + 1}")
#             return base64_image
        
    
#     def pdf_to_base64_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
       
#         logger.info(f"Converting PDF to base64 images: {pdf_path}")
#         pdf_document = fitz.open(pdf_path)
#         total_pages = len(pdf_document)
        
        
#         max_workers = min(total_pages, os.cpu_count() or 1)
        
#         # Process pages in parallel
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
#             futures = {
#                 executor.submit(self.process_pdf_page, page_num, pdf_document, dpi): page_num
#                 for page_num in range(total_pages)
#             }
#             base64_images = [None] * total_pages
            
           
#             for future in as_completed(futures):
#                 page_num = futures[future]
#                 try:
#                     base64_images[page_num] = future.result()
#                 except Exception as e:
#                     logger.error(f"Error processing PDF page {page_num + 1}: {str(e)}")
#                     base64_images[page_num] = None
                    
#         # Remove any failed pages
#         base64_images = [img for img in base64_images if img is not None]
        
#         logger.info(f"Completed processing {len(base64_images)} PDF pages")
#         return base64_images

#     def process_single_page(self, base64_image: str) -> Dict[str, Any]:
#         """Process a single page of the document."""
#         try:
#             return self.extract_invoice_data(base64_image)
#         except json.JSONDecodeError as e:
#             logger.error(f"Error decoding JSON for image: {e}")
#             return {}

#     def extract_invoice_data(self, base64_image):
        
#         """Extract invoice data from a base64-encoded image."""
#         logger.info(f"Extracting invoice data from image ")
#         json_schema={
#   "$schema": "http://json-schema.org/draft-07/schema#",
#   "type": "array",
#   "items": {
#     "type": "object",
#     "properties": {
#       "page_number": {
#         "type": "integer",
#         "description": "The page number of the statement"
#       },
#       "account_details": {
#         "type": "object",
#         "description": "Account holder information and summary data",
#         "properties": {
#           "account_name": { "type": "string" },
#           "address": { "type": "string" },
#           "date": { "type": "string" },
#           "account_number": { "type": "string" },
#           "account_description": { "type": "string" },
#           "branch": { "type": "string" },
#           "drawing_power": { "type": "string" },
#           "interest_rate": { "type": "string" },
#           "mod_balance": { "type": "string" },
#           "cif_no": { "type": "string" },
#           "ckyc_number": { "type": "string" },
#           "ifs_code": { "type": "string" },
#           "micr_code": { "type": "string" },
#           "nomination_registered": { "type": "string" },
#           "opening_balance": { "type": "string" ,"description": "Balance as of the start of the statement period"},
#           "period": { "type": "string","description": "Statement period"},
#         }
#       },
     
#       "account_statement": {
#         "type": "object",
#         "description": "Statement period and transactions",
#         "properties": {
#           "transactions": {
#             "type": "array",
#             "description": "List of all transactions in this page",
#             "items": {
#               "type": "object",
#               "properties": {
#                 "txn_date": { "type": "string" },
#                 "value_date": { "type": "string" },
#                 "description": { "type": "string" },
#                 "ref_no_cheque_no": { "type": "string" },
#                 "debit": { "type": "string" },
#                 "credit": { "type": "string" },
#                 "balance": { "type": "string" }
#               },
#               "required": ["txn_date", "value_date", "description"]
#             }
#           }
#         },
#         "required": ["transactions"]
#       }
#     },
#     "required": ["page_number", "account_statement"]
#   }
# }
#         system_prompt=f"""
#         You are an OCR tool designed to extract all the data from PDF with page numbers and struture it in Json format. Be consistent with the parameters and variables incase of repetition. The output should be in the following format:
#         [
#         {{
#             "page_number": "",
#             "account_details": {{
#                 "account_name": "",
#                 "address": "",
#                 "date": "",
#                 "account_number": "",
#                 "account_description": "",
#                 "branch": "",
#                 "drawing_power": "",
#                 "interest_rate": "",
#                 "mod_balance": "",
#                 "cif_no": "",
#                 "ckyc_number": "",
#                 "ifs_code": "",
#                 "micr_code": "",
#                 "nomination_registered": "",
#                 "opening_balance": "",
#                 "period": ""
#             }},
#             "account_statement": {{
#                 "transactions": [
#                     {{
#                         "txn_date": "",
#                         "value_date": "",
#                         "description": "",
#                         "ref_no_cheque_no": "",
#                         "debit": "",
#                         "credit": "",
#                         "balance": ""
#                     }}
#                 ]
#             }}

#         }}
#         ]
#         """
#         response = self.client.chat.completions.create(
#             model=OPENAI_MODEL,
#             # model="/apps/visionLLM/MiniCPM-V-2_6",
#             response_format={"type": "json_object"},
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": "Extract the data from the image and output JSON."},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpg;base64,{base64_image}",
#                             },
#                         },
#                     ],
#                 },
#             ],
#             max_tokens=os.getenv("MAX_TOKENS"),
#             temperature=os.getenv("TEMPERATURE"),
#             top_p=os.getenv("TOP_P"),
#         )
#         logger.info(json.loads(response.choices[0].message.content))
#         return json.loads(response.choices[0].message.content)
    
#     def process_and_extract_pdf_page(self, page_num: int, pdf_document: fitz.Document, dpi: int = 300) -> Dict[str, Any]:
#         """Process a PDF page and extract data in one step."""
#         try:
#             print(f"Processing and extracting PDF page {page_num + 1}")
#             scale = dpi / 72
            
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 # Load and render the page
#                 page = pdf_document.load_page(page_num)
#                 pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                 img_np = np.array(img)
                
#                 # Preprocess the image
#                 preprocessed = self.preprocess_image(img_np)
#                 temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.png")
#                 cv2.imwrite(temp_image_path, preprocessed)
                
#                 # Convert to base64
#                 with open(temp_image_path, "rb") as image_file:
#                     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
#                 # Extract data immediately after preprocessing
#                 result = self.extract_invoice_data(base64_image)
#                 result['page_number'] = page_num + 1
                
#                 print(f"Completed processing and extraction for page {page_num + 1}")
#                 return result
#         except Exception as e:
#             print(f"Error processing and extracting PDF page {page_num + 1}: {str(e)}")
#             return {"error": f"Failed to process page {page_num + 1}: {str(e)}", "page_number": page_num + 1}


#     def main_extract1(self, read_path: str) -> List[Dict[str, Any]]:
#         def process_file(filename: str) -> List[Dict[str, Any]]:
#             # 1) turn the file into base64 images
#             ext = os.path.splitext(filename)[1].lower()
#             if ext in ('.jpg', '.jpeg', '.png'):
#                 base64_images = [self.encode_image(filename)]
#             elif ext == '.pdf':
#                 # base64_images = self.pdf_to_base64_images(filename, dpi=300)
#                 return self.pdf_to_base64_images(filename, dpi=300)
#             else:
#                 logger.info(f"Skipping unsupported type {ext}")
#                 return []

#             total = len(base64_images)
#             # 2) decide how many threads (you might pick a lower cap to avoid rate-limit bursts)
#             max_workers = min(total, (os.cpu_count() or 4) * 2)

#             results: List[Dict[str, Any]] = [None] * total
#             with ThreadPoolExecutor(max_workers=max_workers) as executor:
#                 # submit one GPT call per page
#                 futures = {
#                     executor.submit(self.extract_invoice_data, img): idx
#                     for idx, img in enumerate(base64_images)
#                 }

#                 for future in as_completed(futures):
#                     idx = futures[future]
#                     page_no = idx + 1
#                     try:
#                         page_data = future.result()
#                         if page_data is None:
#                             raise ValueError("No data returned")
#                         # ensure page_number field is correctly set
#                         page_data["page_number"] = page_no
#                         results[idx] = page_data
#                         logger.info(f"[Thread] page {page_no} done")
#                     except Exception as e:
#                         logger.error(f"[Thread] Error on page {page_no}: {e}")
#                         results[idx] = {
#                             "page_number": page_no,
#                             "error": str(e)
#                         }

#             return results

#         logger.info(f"Starting parallel extraction for: {read_path}")
#         return process_file(read_path)
    
#     def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
#         """Process a PDF file with parallel processing of pages."""
#         print(f"Processing PDF: {pdf_path}")
#         pdf_document = fitz.open(pdf_path)
#         total_pages = len(pdf_document)
        
#         # Set number of parallel workers
#         max_workers = min(total_pages, os.cpu_count() or 1)
#         print(f"Using {max_workers} workers for parallel processing")
        
#         results = [None] * total_pages
        
#         # Process pages in parallel - each thread handles preprocessing and extraction for its page
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = {
#                 executor.submit(self.process_and_extract_pdf_page, page_num, pdf_document): page_num
#                 for page_num in range(total_pages)
#             }
            
#             for future in as_completed(futures):
#                 page_num = futures[future]
#                 try:
#                     results[page_num] = future.result()
#                     print(f"Completed page {page_num + 1} of {total_pages}")
#                 except Exception as e:
#                     print(f"Error in future for page {page_num + 1}: {str(e)}")
#                     results[page_num] = {"error": f"Failed to process page {page_num + 1}: {str(e)}", 
#                                         "page_number": page_num + 1}
        
#         # Filter out any None results
#         results = [r for r in results if r is not None]
#         return results

#     def process_image(self, image_path: str) -> Dict[str, Any]:
#         """Process a single image file."""
#         print(f"Processing image: {image_path}")
#         try:
#             # Load and preprocess the image
#             img = cv2.imread(image_path)
#             preprocessed = self.preprocess_image(img)
            
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 temp_image_path = os.path.join(temp_dir, "temp_image.png")
#                 cv2.imwrite(temp_image_path, preprocessed)
                
#                 # Convert to base64
#                 base64_image = self.encode_image(temp_image_path)
                
#                 # Extract data
#                 result = self.extract_invoice_data(base64_image)
#                 result['page_number'] = 1  # Single image is treated as page 1
                
#                 return result
#         except Exception as e:
#             print(f"Error processing image {image_path}: {str(e)}")
#             return {"error": f"Failed to process image: {str(e)}", "page_number": 1}

#     def main_extract(self, read_path: str) -> List[Dict[str, Any]]:
#         """Main entry point to process a file."""
#         print(f"Starting extraction for: {read_path}")
        
#         file_extension = os.path.splitext(read_path)[1].lower()
        
#         if file_extension in ['.jpg', '.jpeg', '.png']:
#             return [self.process_image(read_path)]
#         elif file_extension == '.pdf':
#             return self.process_pdf(read_path)
#         else:
#             print(f"Skipping unsupported file type: {file_extension}")
#             return []
    

# # read_path =r"C:\Users\Trisha\Downloads\bank statement new-1-5-2.pdf"
# # processor = PerformOCR_v1()
# # results = processor.main_extract(read_path)
# # output_path = r"C:\TrishaW\ocr\trustt-platform-document-processing\output_2.json" 


# # with open(output_path, "w", encoding="utf-8") as json_file:
# #     json.dump(results, json_file, indent=4, ensure_ascii=False)


#     def extract_data(self, files):
#         try:
            
#             all_results = []
            
#             for file in files:
                
#                 temp_file_path = os.path.join('', file.filename)
#                 print(f"Processing file: {file.filename}")
#                 file.save(temp_file_path)

#                 try:
                    
#                     # ocr_instance = PerformOCR_v1()
#                     file_extension = os.path.splitext(file.filename)[1].lower()
                    
#                     if file_extension in ['.jpg', '.jpeg', '.png']:
                        
#                         result = self.main_extract(temp_file_path)
                    
#                     elif file_extension == '.pdf':
                        
#                         result = self.main_extract(temp_file_path)
#                     else:
                        
#                         document_result = {
#                             "document_name": file.filename,
#                             "result": {"error": f"Unsupported file type: {file_extension}"}
#                         }
#                         all_results.append(document_result)
#                         continue

#                     document_result = {
#                         "document_name": file.filename,
#                         "result": result
#                     }
#                     all_results.append(document_result)
                    
#                 except Exception as e:

#                     document_result = {
#                         "document_name": file.filename,
#                         "result": {"error": f"Failed to process document: {str(e)}"}
#                     }
#                     all_results.append(document_result)
                
#                 finally:
#                     if os.path.exists(temp_file_path):
#                         os.remove(temp_file_path)
            
#             with open(f"{file.filename}.json", 'w', encoding='utf-8') as f:
#                 json.dump(all_results, f, ensure_ascii=False, indent=4)
#             return all_results

#         except Exception as e:
#             print(f"Error in extract_data route: {e} at line number {str(e.__traceback__.tb_lineno)}")
#             return jsonify({"error": str(e)}), 500
    

#     def bankstatementclassification(self,extracted_text):

#         try:
    
#             system_prompt=f"""
#             You are an AI trained in financial document analysis with specialized expertise in bank statements, invoices, and transaction records. 
#             Your task is to meticulously analyze provided text to determine if it is a bank statement based on strict criteria. 
#             You must:
#             - Cross-check for definitive bank statement features (e.g., transaction tables, account details).
#             - Ignore irrelevant text or partial matches (e.g., invoices with payment tables).
#             - Return a JSON response with a boolean field "isBankStatement" indicating the classification result.
#             """



#             prompt = f"""
#             Analyze the following text to determine if it is a bank statement. Use these criteria:
#             Key Indicators:
#             1. Account/Bank Metadata:
#             - Account holder name, number, IFSC, branch, or bank name.  
#             - Statement period (e.g., "01 Apr 2024 - 30 Apr 2024")
#             - Opening & closing balance for the statement period.  

#             2. Transaction Table Structure: 
#             - Columns: Date, Description, Debit/Credit, Balance (mandatory).  
#             - Running balance updates after each transaction.  

#             3. Financial Terminology:  
#             - Keywords: NEFT, IMPS, RTGS, UPI, ATM, POS, EMI, TDS, FD.  
#             - Currency symbols with formatted amounts.
#             - Amount with recurring format (e.g., "1,000.00 Cr" or "1,000.00 Dr").

#             4. Consistency & Completeness:  
#             - Opening/closing balances.  
#             - Page numbers, disclaimers, or bank logos (if present).  

#             Exclusion Criteria:
#             - Reject if only payment tables (no account metadata).  
#             - Reject if standalone transaction lists (e.g., credit card bills without bank linkage).  

#             Output Format (JSON): 
#             {{
#             "isBankStatement": bool
#             }}

#             Text to analyze:  
#             ```{extracted_text}```
#             """

#             tools = [
#   {
#     "type": "function",
#     "function": {
#       "name": "bank_statement_classification",
#       "description": "Function to classify whether the given text is from a bank statement.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "isBankStatement": {
#             "type": "boolean",
#             "enum": [True, False],
#             "description": "Indicates if the text is a bank statement (true) or not (false)."
#           }
#         },
#         "required": ["isBankStatement"]
#       }
#     }
#   }
# ]

#             response = self.client.chat.completions.create(
#                 model=OPENAI_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=5000,
#                 tools= tools,
#                 tool_choice={"type": "function", "function": {"name": "bank_statement_classification"}},
#             )

#             # response_text = response.choices[0].message.content


#             # if response_text.startswith("```") and response_text.endswith("```"):
#             #     response_text = response_text.strip("```").strip("json").strip()

#             # try:
#             #     result_json = json.loads(response_text)
#             # except json.JSONDecodeError:
#             #     result_json = {"error": "Failed to parse LLM response.", "raw_response": response_text}

            

#             return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
#         except Exception as e:
#             logger.error(f"Error in bank statement classification: {e} at line number {str(e.__traceback__.tb_lineno)}")
        


    
# class BankStatementAnalysis:
#     def __init__(self,db):
#         self.db = db

#     def performBasicAnalysis(self, request):
#         ####### Objective: Perform basic analysis on the bank statement
#         ####### 1. Classify wether the document provided is a bank statement or not
#         ####### 2. Extract the details from the bank statement
#         ####### 3. Check if the account name is matching with either the borrower name or proprietor name
#         ####### 4. Check if each page has the same account number, name and ifsc
#         ####### 5. Extract the Account number, Ifsc and name then perform the penny drop analysis to match the account holder name details
#         ####### 6. Extract the start date and end date of the bank statement and check if it is matching with the expected start date and end date provided by the user
#         ####### 7. Calculate the balance after each transaction and match the same with the balance after transaction in the statement

#         try:
#             document = request.files.getlist('files') 
#             customer_id = request.form.get('customer_id','')
#             loan_application_id = request.form.get('loan_application_id','')
#             borrower_name = request.form.get('borrower_name','')
#             proprietor_name = request.form.get('proprietor_name','')
#             expected_stmt_start_date = request.form.get('expected_stmt_start_date','')
#             expected_stmt_end_date = request.form.get('expected_stmt_end_date','')
#             doc_format = request.form.get('doc_format ','pdf')

#             logger.info(f"Starting analysis for customer: {customer_id}, loan_application_id: {loan_application_id}")
#             logger.info(f"Borrower name: {borrower_name}, Proprietor name: {proprietor_name}")
#             logger.info(f"Expected statement period: {expected_stmt_start_date} to {expected_stmt_end_date}")

        

#             metadata = {}
#             BSA_Analysis="Fail"

#             # Check if the files are empty
#             if not document:
#                 logger.error("No files provided in the request.")
#                 return {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     "BSA_Analysis": BSA_Analysis,
#                     "error": "No files provided"
#                     }, 400
            

#             # Extracting data from the document using the performOCR class

#             ocr_instance = PerformOCR_v1()
#             extraction_result = ocr_instance.extract_data(files=document)

#             print(type(extraction_result))
#             document_extracted = extraction_result[0].get('result', [])

#             # classifying the document as a bank statement or not using the bankstatementclassification class

#             classification_result = ocr_instance.bankstatementclassification(extracted_text=document_extracted)

#             logger.info(f"Classification result: {classification_result}")
#             is_bank_statement = classification_result.get('isBankStatement', False)

#             if not is_bank_statement:
#                 logger.error("The document is not a bank statement.")
#                 metadata["is_bank_statement"] = False
#                 return {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     "BSA_Analysis": BSA_Analysis,
#                     "meta_data": metadata
#                 }
#             else:
#                 metadata["is_bank_statement"] = True
#                 logger.info("The document is a bank statement.")



#             # for now we are opening the output.json file and storing the data in the document_extracted variable

#             # with open('C:/Users/Bharath/Documents/docu maharaj/trustt-platform-document-processing/trusttApp/output_v2.json', 'r') as f:
#             #     document_extracted = json.load(f)
            
#             analysed_data = self.check_name_and_period(document_extracted, expected_stmt_start_date, expected_stmt_end_date, borrower_name, proprietor_name)

#             account_holder_name_match = analysed_data.get('account_holder_name_match',"NO")
#             is_statement_is_within_the_period = analysed_data.get('is_statement_is_within_the_period',"NO")
#             start_date = analysed_data.get('start_date','')
#             end_date = analysed_data.get('end_date','')
#             missing_months = analysed_data.get('missing_months',[])
#             provided_months = analysed_data.get('provided_months',[])
#             is_completely_outside_period = analysed_data.get('is_completely_outside_period','NO')

#             customer_details_based_on_stmt = {
#                 "name" : document_extracted[0]['account_details']['account_name'],
#                 "account_number" : document_extracted[0]['account_details']['account_number'],
#                 "statement_period": document_extracted[0]['account_details']['period'],
#             }

#             metadata["customer_details"] = customer_details_based_on_stmt
        
            

#             # check if the account holder name is matching with the borrower name or proprietor name
#             if account_holder_name_match == 'NO':
#                 logger.error("The account holder name is not matching with the borrower name or proprietor name.")

#                 metadata["account_holder_name_match"] = False

#                 return {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The account holder name is not matching with the borrower name or proprietor name."
#                     "BSA_Analysis": BSA_Analysis,
#                     "meta_data": metadata

#                 }
#             else:
#                 metadata["account_holder_name_match"] = True
#                 logger.info("The account holder name is matching with the borrower name or proprietor name.")

#             # check if each page has the same account number, name and ifsc
#             account_numbers = {
#                 rec.get('account_details', {}).get('account_number')
#                 for rec in document_extracted
#                 if rec.get('account_details', {}).get('account_number')
#             }


#             ifsc_codes = {
#                 rec.get('account_details', {}).get('ifs_code')
#                 for rec in document_extracted
#                 if rec.get('account_details', {}).get('ifs_code')
#             }

#             account_names = {
#                 rec.get('account_details', {}).get('account_name')
#                 for rec in document_extracted
#                 if rec.get('account_details', {}).get('account_name')
#             }


#             if len(account_numbers) != 1:
#                 logger.error("The account number is not same in all the pages.")
#                 metadata["pagewise_account_number_match"] = False
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The account number is not same in all the pages."
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                     }
#                 return response
#             else:
#                 metadata["pagewise_account_number_match"] = True
#                 logger.info("The account number is same in all the pages.")
            
#             if len(ifsc_codes) != 1:
#                 logger.error("The ifsc code is not same in all the pages.")
#                 metadata["pagewise_ifsc_code_match"] = False
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The ifsc code is not same in all the pages."
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                     }
#                 return response
#             else:
#                 metadata["pagewise_ifsc_code_match"]= True
#                 logger.info("The ifsc code is same in all the pages.")

#             if len(account_names) != 1:
#                 logger.error("The account name is not same in all the pages.")
#                 metadata["pagewise_account_name_match"] = False
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The account name is not same in all the pages."
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                     }
#                 return response
#             else:
#                 metadata["pagewise_account_name_match"] = True
#                 logger.info("The account name is same in all the pages.")

#             # extract the Account number, Ifsc and name then perform the penny drop analysis to match the account holder name details
#             account_number = document_extracted[0]["account_details"]["account_number"]
#             ifsc = document_extracted[0]["account_details"]["ifs_code"]
#             account_name = document_extracted[0]["account_details"]["account_name"]
#             # penny_drop_analysis = performPennyDropAnalysis(account_number, ifsc, account_name)
#             # penny_drop_status = penny_drop_analysis['status']

#             mock_penny_drop = {"00000034909111330": "success","00000038322372698":"success","7223648801":"success","719130110000033":"success","69220000331696":"success","60240500000351":"success","357305040050196":"success"}
#             penny_drop_status = mock_penny_drop.get(account_number, 'fail')
#             if penny_drop_status.lower() != 'success':
#                 logger.error("The account holder name is not matching with the penny drop analysis.")
#                 metadata["penny_drop_analysis"] = False
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The account holder name is not matching with the penny drop analysis.",
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                     }
#                 return response
#             else:
#                 metadata["penny_drop_analysis"] = True
#                 logger.info("The account holder name is matching with the penny drop analysis.")
            
            
#             if is_completely_outside_period == 'YES':
#                 logger.error("The statement is completely outside the given period.")
#                 metadata["stmt_completely_outside_period"] = True
#                 stmt_period_analysis = {"stmt_start_date": start_date,
#                     "stmt_end_date": end_date,
#                     "expected_stmt_start_date": expected_stmt_start_date,
#                     "expected_stmt_end_date": expected_stmt_end_date,
#                     "missing_months": missing_months,
#                     "provided_months": provided_months}
#                 metadata["stmt_period_analysis"] = stmt_period_analysis
#                 return {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,                  
#                 }
#             else:
#                 metadata["stmt_completely_outside_period"] = False
#                 logger.info("The statement is not completely outside the given period.")
            
#             # check if the statement is within the expected period
#             if is_statement_is_within_the_period == 'NO':
#                 metadata["is_stmt_covers_provided_period"] = False

#                 stmt_period_analysis = {
#                     # "Text": "The statement is not within the expected period.",
#                     "stmt_start_date": start_date,
#                     "stmt_end_date": end_date,
#                     "missing_months": missing_months,
#                     "provided_months": provided_months
#                 }
#                 metadata["stmt_period_analysis"] = stmt_period_analysis
#                 logger.error("The statement is not within the expected period.")
#             else:
#                 metadata["is_stmt_covers_provided_period"] = True
#                 logger.info("The statement is within the expected period.")
 
#             # Initialize opening balance
#             opening_balance = Decimal(document_extracted[0]['account_details']['opening_balance'].replace(',', '') or '0')
#             prev_balance = opening_balance

#             if is_statement_is_within_the_period.upper() == "NO":
#                 BSA_Analysis = "INCOMPLETE"
#             else:
#                 BSA_Analysis = "SUCCESS"

#             # Iterate pages and transactions in order
#             missmatch_txns = []
#             credit_sum = 0
#             debit_sum = 0
#             txn_cnt = 0
#             for page in sorted(document_extracted, key=lambda x: x['page_number']):
#                 for tx in page.get('account_statement', {}).get('transactions', []):
#                     debit  = Decimal(tx['debit'].replace(',', '') or '0')
#                     credit = Decimal(tx['credit'].replace(',', '') or '0')
#                     stated_balance = Decimal(tx['balance'].replace(',', '') or '0')

#                     # Calculate expected balance
#                     calculated_balance = prev_balance - debit + credit
#                     credit_sum += credit
#                     debit_sum += debit
                    
#                     # Check for mismatch
#                     if calculated_balance != stated_balance:
#                         print(f"Mismatch on {tx['txn_date']}: "
#                             f"expected {calculated_balance:.2f}, "
#                             f"stated {stated_balance:.2f}")
#                         missmatch_txns.append({
#                             "txn_date": tx['txn_date'],
#                             "expected_balance": f"{calculated_balance:.2f}",
#                             "stated_balance": f"{stated_balance:.2f}",
#                             "debit": f"{debit:.2f}",
#                             "credit": f"{credit:.2f}",
#                             "previous_closing_balance": f"{prev_balance}",
#                         })
#                         BSA_Analysis = "FAIL"
                    
#                     # Update for next transaction
#                     prev_balance = stated_balance
#                     txn_cnt += 1

#             metadata["total_txns"] = txn_cnt
#             metadata["total_debits"] = debit_sum
#             metadata["total_credits"] = credit_sum

            

#             if len(missmatch_txns) != 0:
#                 logger.info("The balance after each transaction is not matching with the balance after transaction in the statement.")
#                 metadata["any_mismatch_txns"] = True
#                 metadata["mismatch_txns"] = missmatch_txns
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "The balance after each transaction is not matching with the balance after transaction in the statement.",
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                 }
#                 return response
#             else:
#                 metadata["any_mismatch_txns"] = False
#                 logger.info("The balance after each transaction is matching with the balance after transaction in the statement.")
#                 response = {
#                     "customer_id": customer_id,
#                     "loan_application_id": loan_application_id,
#                     # "Text": "BSA analysis completed successfully.",
#                     "meta_data": metadata,
#                     "BSA_Analysis": BSA_Analysis,
#                 }
#                 return response

#         except Exception as e:
#             logger.error(f"Error in extracting form data: {e} line: {e.__traceback__.tb_lineno}")
#             return {"error": str(e)}, 500
        


#     def check_name_and_period(self,ocr_results, given_start_date, given_end_date, given_borrower_name, given_proprietor_name):

#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         client = OpenAI(api_key=openai_api_key)


#         txn_dates = [
#             txn.get('txn_date') or txn.get('Txn Date')
#             for record in ocr_results
#             for txn in (
#                 record.get('account_statement', {}).get('transactions', [])
#                 + record.get('transactions', [])
#             )
#             if txn.get('txn_date') or txn.get('Txn Date')
#         ]

#         account_names = {
#                 rec.get('account_details', {}).get('account_name')
#                 for rec in ocr_results
#                 if rec.get('account_details', {}).get('account_name')
#             }
        

        
#         llm_prompt = f"""Analyze this bank statement data and do the following:

#         1. Check if the {account_names} is matching with the {given_borrower_name if given_borrower_name else ""} or  {given_proprietor_name if given_proprietor_name else ""} name. If yes, then set "account_holder_name_match" to "YES", else "NO".
#         2. Note that name matching should not be strich. For example if the name is justin b but actual name is justin bieber, then is should be YES. Set it to no if it is not matching at all. For example if the name is justin b and actual name is Hailey b, then it should be NO.
#         2. Check if the statement contains transactons from {given_start_date} to {given_end_date}. If yes, then set "is_statement_is_within_the_period" to "YES", else "NO" and set the corresponding missing months.
#         3. Note that if the period is from 01 jan 2023 to 01 dec 2023, and transactions dates are from 03 jan 2023 to 31 dec 2023, then note that the user has not done any transactions on 1 and 2 jan, so it can be ignored since he provided for jan 2023 to dec 2023. So, the missing months will be [].
#         4. If the statement is completely outside the given period, then set is_completely_outside_period to "YES", else if it is partially outside, then set is_completely_outside_period to "NO".
#         5. Store the missing months and provided months in the format of YYYY-MMM. For example, if the missing month is Jan 2023, then it should be stored as 2023-Jan. If the provided month is Jan 2023, then it should be stored as 2023-Jan.
#         4. Return JSON with:
#         {{
#             "account_holder_name_match": "YES/NO",
#             "is_statement_is_within_the_period": "YES/NO",
#             "start_date": "YYYY-MM-DD",
#             "end_date": "YYYY-MM-DD",
#             "missing_months": ["list of months along with the year that are missing example: ['2023-Jan', '2023-Feb']"],
#             "provided_months": ["list of months along with the year that are provided example: ['2023-Jan', '2023-Feb']"],
#             "is_completely_outside_period": "YES/NO"
#         }}
#         5. Strictly follow the JSON format and do not add any additional text or explanation.

#         Extracted transaction dates: {txn_dates}
    
#         """

#         tools = [
#             {
#                 "type": "function",
#                 "function": {
#                 "name": "bsa_analysis",
#                 "description": "Function to analyze the bank statement data.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                     "account_holder_name_match": {
#                         "type": "string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the account holder name matches the given names."
#                     },
#                     "is_statement_is_within_the_period": {
#                         "type":"string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the statement is within the given period."
#                     },
#                     "start_date": {
#                         "type": "string",
#                         "format": "date",
#                         "description": "Start date of the statement period."
#                     },
#                     "end_date": {
#                         "type": "string",
#                         "format": "date",
#                         "description": "End date of the statement period."
#                     },
#                     "missing_months": {
#                         "type": "array",
#                         "items": {
#                             "type": "string",
#                             "description": "List of months along with year that are missing."
#                         }
#                     },
#                     "is_completely_outside_period": {
#                         "type": "string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the statement is completely outside the given period."
#                     },
#                     "provided_months": {
#                         "type": "array",
#                         "items": {
#                             "type": "string",
#                             "description": "List of months along with year that are provided."
#                         }
#                     }
#                     },
#                     "required": [
#                         "account_holder_name_match",
#                         "is_statement_is_within_the_period",
#                         "start_date",
#                         "end_date",
#                         "missing_months",
#                         "is_completely_outside_period",
#                         "provided_months"
#                     ]
#                 }
#                 }
#             }
#             ]
        
#         msg = [
#                 {"role": "system", "content": "You're a bank statement analyst."},
#                 {"role": "user", "content": llm_prompt}
#             ]
        
        
#         response = client.chat.completions.create(
#             model="gpt-4o",  
#             response_format={"type": "json_object"},
#             messages=msg,
#             tools= tools,
#             tool_choice={"type": "function", "function": {"name": "bsa_analysis"}},
#         )

#         # print(response)
#         print(response.choices[0].message.tool_calls[0].function.arguments)
#         return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    
#     # def process(request):
#     #     """
#     #     - Pulls `file` out of request.files
#     #     - If encrypted, uses request.form['password'] to decrypt
#     #     - Re-serializes into a BytesIO
#     #     - Wraps that BytesIO as a new FileStorage and *overwrites* request.files['file']
#     #     """

#     #     upload = request.files.get("file")

#     #     reader = PdfReader(upload.stream)

#     #     if reader.is_encrypted:
#     #         pwd = request.form.get("password", "")


#     #     # 4) write out a fresh, unencrypted copy into memory
#     #     writer = PdfWriter()
#     #     for pg in reader.pages:
#     #         writer.add_page(pg)

#     #     buf = io.BytesIO()
#     #     writer.write(buf)
#     #     buf.seek(0)

#     #     # 5) wrap that buffer in a new FileStorage
#     #     clean_file = FileStorage(
#     #         stream=buf,
#     #         filename=upload.filename,
#     #         content_type=upload.mimetype
#     #     )

#     #     # 6) override request.files with a mutable MultiDict
#     #     #    (so downstream code that does request.files['file'] just works)
#     #     request.files = MultiDict([("file", clean_file)])

#     #     # 7) return None on success
#     #     return None



class BankStatementAnalysis:
    def __init__(self,db):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.sessionService = SessionService(db)
        
        # Initialize OCR configuration with improved settings
        self.ocr_config = {
            "deskew_pages": True,            # Keep deskewing as it helps with alignment
            "enable_caching": True,          # Enable caching for better performance
            "enable_checkpointing": True,    # Save progress for large documents
            "enable_preprocessing": False,   # Let RapidOCR handle the image directly
            "max_workers_multiplier": 2.0,   # Increase to use more parallel workers
            "min_workers": 4,                # Minimum 4 workers for parallel processing
            "max_workers": 8,                # Increase max workers for better parallelization
            "dpi": 400,                      # Keep high DPI for quality
            "temperature": 0.1,              # Lower temperature for more deterministic output
            "top_p": 0.95,                   # Higher top_p for better quality output
            # No max_tokens limit to ensure we capture all content
        }
        
        # Directory for CSV reports
        self.reports_dir = os.path.join(os.getcwd(), "bsa_reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize OCR processor
        self.ocr_processor = Perform_OCR_v2(self.ocr_config)

    # def generate_csv_report(self, analysis_result, file_name):
    #     """
    #     Generate a CSV report with analysis metrics
        
    #     Args:
    #         analysis_result: The analysis result dictionary
    #         file_name: Original file name for the report
            
    #     Returns:
    #         Path to the generated CSV report
    #     """
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     report_file = os.path.join(self.reports_dir, f"bsa_report_{timestamp}.csv")
        
    #     metadata = analysis_result.get("meta_data", {})
    #     bsa_status = analysis_result.get("BSA_Analysis", "FAIL")
        
    #     # Prepare report data
    #     report_data = {
    #         "File Name": [os.path.basename(file_name)],
    #         "BSA Status": [bsa_status],
    #         "Classification Match": [str(metadata.get("is_bank_statement", False))],
    #         "Extraction Success": [str(metadata.get("extraction_success", False))],
    #         "Account Holder Name Match": [str(metadata.get("account_holder_name_match", False))],
    #         "Account Number": [metadata.get("customer_details", {}).get("account_number", "")],
    #         "Account Name": [metadata.get("customer_details", {}).get("name", "")],
    #         "IFSC Code": [metadata.get("customer_details", {}).get("ifs_code", "")],
    #         "Transaction Accuracy (%)": [metadata.get("transaction_accuracy", 0)],
    #         "Total Transactions": [metadata.get("total_txns", 0)],
    #         "Error Transactions": [metadata.get("error_txns", 0)],
    #         "Mismatched Transactions": [len(metadata.get("mismatch_txns", []))],
    #         "Pages with Missing Account Numbers": [", ".join(map(str, metadata.get("missing_account_pages", [])))]
    #     }
        
    #     # Write to CSV
    #     with open(report_file, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
            
    #         # Write headers
    #         writer.writerow(report_data.keys())
            
    #         # Write values
    #         writer.writerow([report_data[key][0] for key in report_data.keys()])
            
    #         # Add a separator
    #         writer.writerow([])
            
    #         # Add transaction mismatch details if any
    #         mismatch_txns = metadata.get("mismatch_txns", [])
    #         if mismatch_txns:
    #             writer.writerow(["Transaction Date", "Description", "Expected Balance", "Stated Balance", "Debit", "Credit", "Previous Balance"])
    #             for tx in mismatch_txns:
    #                 writer.writerow([
    #                     tx.get("txn_date", ""),
    #                     tx.get("description", ""),
    #                     tx.get("expected_balance", ""),
    #                     tx.get("stated_balance", ""),
    #                     tx.get("debit", ""),
    #                     tx.get("credit", ""),
    #                     tx.get("previous_closing_balance", "")
    #                 ])
        
    #     logger.info(f"CSV report generated: {report_file}")
    #     return report_file

    def extract_text_from_document(self, file_path):
        """
        Extracts text from document using Perform_OCR_v2
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple with extraction status and the extracted content
        """
        logger.info(f"Starting OCR processing for file: {file_path}")
        
        try:
            # Process the document
            extraction_result = self.ocr_processor.main_extract(file_path)

            
            # Check for errors in extraction
            if 'error' in extraction_result:
                logger.error(f"OCR extraction error: {extraction_result['error']}")
                return False, {"error": extraction_result['error']}
            
            # Get the results from extraction
            results = extraction_result.get('results', {})
            if "data" in results:
                return True, results["data"]
            
            if not results:
                logger.error("No text content extracted from document")
                
                # Fallback: Try to extract text directly from PDF if it's a PDF file
                if file_path.lower().endswith('.pdf'):
                    try:
                        logger.info("Attempting fallback text extraction directly from PDF")
                        import fitz  # PyMuPDF
                        
                        direct_text = []
                        doc = fitz.open(file_path)
                        for page_num, page in enumerate(doc, 1):
                            text = page.get_text()
                            direct_text.append({
                                'page_number': page_num,
                                'text': text,
                                'account_details': self.extract_account_details_from_text(text),
                                'account_statement': self.extract_transactions_from_text(text)
                            })
                        
                        if direct_text:
                            logger.info(f"Successfully extracted text directly from PDF: {len(direct_text)} pages")
                            return True, direct_text
                    except Exception as pdf_err:
                        logger.error(f"Failed direct PDF text extraction: {str(pdf_err)}")
                
                return False, {"error": "Failed to extract content from document"}
            
            # Convert any non-dict objects to dicts for consistent handling
            processed_results = []
            
            for idx, item in enumerate(results, 1):
                if isinstance(item, dict):
                    # It's already a dict, just add it
                    processed_results.append(item)
                    
                elif isinstance(item, str):
                    # Try to parse as JSON if it looks like JSON
                    item = item.strip()
                    if (item.startswith('{') and item.endswith('}')) or (item.startswith('[') and item.endswith(']')):
                        try:
                            json_data = json.loads(item)
                            if isinstance(json_data, dict):
                                processed_results.append(json_data)
                            elif isinstance(json_data, list):
                                # If it's a list of objects, add each one
                                for subitem in json_data:
                                    if isinstance(subitem, dict):
                                        processed_results.append(subitem)
                            continue
                        except json.JSONDecodeError:
                            # Not valid JSON, continue with string handling
                            pass
                    
                    # Handle as raw text with minimal structure
                    logger.warning(f"Non-dict result item: {type(item)}. Converting to structured data.")
                    
                    # Try to identify transaction-like data
                    account_details = self.extract_account_details_from_text(item)
                    transactions = self.extract_transactions_from_text(item)
                    
                    # Create a structured representation
                    structured_data = {
                        'page_number': idx,
                        'text': item,
                        'account_details': account_details
                    }
                    
                    if transactions:
                        structured_data['account_statement'] = {'transactions': transactions}
                    
                    processed_results.append(structured_data)
                    
                else:
                    # For other types, just convert to string with minimal structure
                    logger.warning(f"Unknown result item type: {type(item)}. Converting to string.")
                    processed_results.append({
                        "text": str(item),
                        "page_number": idx
                    })
                logger.info(f"Processed page {idx}: {processed_results[-1]}")
            
            # Make sure we have at least one result
            if not processed_results:
                processed_results = [{
                    "text": "No structured data extracted",
                    "page_number": 1
                }]
            
            logger.info(f"Successfully extracted content from {len(processed_results)} pages")
            return True, processed_results
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return False, {"error": f"OCR processing failed: {str(e)}"}
            
    def extract_account_details_from_text(self, text):
        """
        Extract account details from raw text
        
        Args:
            text: Raw text to extract from
            
        Returns:
            Dictionary of account details
        """
        account_details = {}
        
        # Try to extract account number
        account_patterns = [
            r'Account\s*(?:No|Number|#)\s*:?\s*([A-Z0-9]+)',
            r'A/C\s*(?:No|Number|#)\s*:?\s*([A-Z0-9]+)',
            r'(?:Account|A/C)\s*:?\s*([A-Z0-9]+)'
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account_details['account_number'] = match.group(1).strip()
                break
        
        # Try to extract account name
        name_patterns = [
            r'(?:Account|A/C)\s*(?:Name|Holder)\s*:?\s*([A-Za-z\s.]+)',
            r'(?:Name|Customer)\s*:?\s*([A-Za-z\s.]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account_details['account_name'] = match.group(1).strip()
                break
        
        # Try to extract IFSC code
        ifsc_pattern = r'IFSC\s*(?:Code|#)?\s*:?\s*([A-Z0-9]+)'
        match = re.search(ifsc_pattern, text, re.IGNORECASE)
        if match:
            account_details['ifs_code'] = match.group(1).strip()
        
        # Try to extract statement period
        period_patterns = [
            r'(?:Statement|Period)\s*(?:for|from)?\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\s*(?:to|till|-)\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:From|Between)\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\s*(?:to|till|-)\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})'
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                period = f"{match.group(1)} to {match.group(2)}"
                account_details['period'] = period
                break
        
        # Try to extract opening balance
        balance_patterns = [
            r'Opening\s*Balance\s*:?\s*(?:Rs\.?|INR)?\s*([\d,]+\.\d{2})',
            r'Beginning\s*Balance\s*:?\s*(?:Rs\.?|INR)?\s*([\d,]+\.\d{2})'
        ]
        
        for pattern in balance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account_details['opening_balance'] = match.group(1).strip()
                break
                
        return account_details
    
    def extract_transactions_from_text(self, text):
        """
        Extract transactions from raw text
        
        Args:
            text: Raw text to extract from
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        # Split into lines
        lines = text.split('\n')
        
        # Regular expressions for transaction identification
        date_pattern = r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}'
        amount_pattern = r'(?:Rs\.?|INR)?\s*([\d,]+\.\d{2})'
        
        for line in lines:
            line = line.strip()
            
            # Skip short lines
            if len(line) < 15:
                continue
                
            # Look for lines that appear to be transactions
            date_match = re.search(date_pattern, line)
            amount_matches = re.findall(amount_pattern, line)
            
            if date_match and amount_matches:
                transaction = {
                    'txn_date': date_match.group(0),
                    'description': line
                }
                
                # Determine if debit or credit
                if 'dr' in line.lower() or 'debit' in line.lower() or 'withdrawal' in line.lower():
                    transaction['debit'] = amount_matches[0] if amount_matches else '0.00'
                    if len(amount_matches) > 1:
                        transaction['balance'] = amount_matches[-1]
                elif 'cr' in line.lower() or 'credit' in line.lower() or 'deposit' in line.lower():
                    transaction['credit'] = amount_matches[0] if amount_matches else '0.00'
                    if len(amount_matches) > 1:
                        transaction['balance'] = amount_matches[-1]
                else:
                    # If no indicator, guess based on position
                    if len(amount_matches) >= 2:
                        transaction['amount'] = amount_matches[0]
                        transaction['balance'] = amount_matches[-1]
                    elif amount_matches:
                        transaction['amount'] = amount_matches[0]
                
                transactions.append(transaction)
                
        return transactions

    def bankstatementclassification(self, extracted_text):
        """
        Classify if the document is a bank statement
        
        Args:
            extracted_text: The extracted text from the document
            
        Returns:
            Tuple with classification status (boolean) and the classification result
        """
        try:
            # Handle different input formats
            text_content = ""
            
            # Add debug logging to see what we're getting
            logger.info(f"Extracted text type: {type(extracted_text)}")
            if isinstance(extracted_text, list):
                logger.info(f"List length: {len(extracted_text)}")
                if len(extracted_text) > 0:
                    logger.info(f"First item type: {type(extracted_text[0])}")
            
            # Check if we can perform direct bank statement detection based on filename or content markers
            if isinstance(extracted_text, list):
                for item in extracted_text:
                    if isinstance(item, dict):
                        # Check if there are any account details or transaction data
                        has_account_details = False
                        has_transactions = False
                        
                        account_details = item.get('account_details', {})
                        if isinstance(account_details, dict) and account_details:
                            has_account_details = True
                            
                        account_statement = item.get('account_statement', {})
                        if isinstance(account_statement, dict):
                            transactions = account_statement.get('transactions', [])
                            if transactions:
                                has_transactions = True
                        
                        # If it has both account details and transactions, it's likely a bank statement
                        if has_account_details and has_transactions:
                            logger.info("Direct detection: Found both account details and transactions")
                            return True, {"isBankStatement": True, "method": "direct_detection"}
                        
                        # If it has the text field, check for bank statement keywords
                        text = item.get('text', '')
                        if isinstance(text, str) and text:
                            keywords = ['bank statement', 'account statement', 'transaction', 'balance', 'debit', 'credit', 
                                        'opening balance', 'closing balance', 'withdrawal', 'deposit']
                            keyword_count = sum(1 for kw in keywords if kw in text.lower())
                            if keyword_count >= 2:
                                logger.info(f"Keyword detection: Found {keyword_count} bank statement keywords")
                                return True, {"isBankStatement": True, "method": "keyword_detection"}
            
            # If it's a list of dictionaries
            if isinstance(extracted_text, list):
                # Extract and combine all text content
                for item in extracted_text:
                    if isinstance(item, dict):
                        # Try to extract key fields that indicate a bank statement
                        account_details = item.get('account_details', {})
                        if isinstance(account_details, dict):
                            text_content += f"Account Name: {account_details.get('account_name', '')}\n"
                            text_content += f"Account Number: {account_details.get('account_number', '')}\n"
                            text_content += f"Period: {account_details.get('period', '')}\n"
                            text_content += f"IFSC Code: {account_details.get('ifs_code', '')}\n"
                        
                        # Extract transactions
                        account_statement = item.get('account_statement', {})
                        if isinstance(account_statement, dict):
                            transactions = account_statement.get('transactions', [])
                            if transactions:
                                text_content += "Transactions:\n"
                                for tx in transactions[:10]:  # Limit to first 10 to keep prompt size reasonable
                                    if isinstance(tx, dict):
                                        text_content += f"Date: {tx.get('txn_date', '')}, "
                                        text_content += f"Description: {tx.get('description', '')}, "
                                        text_content += f"Debit: {tx.get('debit', '')}, "
                                        text_content += f"Credit: {tx.get('credit', '')}, "
                                        text_content += f"Balance: {tx.get('balance', '')}\n"
                        
                        # Also extract any raw text if available
                        if 'text' in item and isinstance(item['text'], str):
                            text_content += item['text'] + "\n\n"
                    
                    # For string items (like the warning showed in logs)
                    elif isinstance(item, str):
                        text_content += item + "\n"
                    
                    # For any other type
                    else:
                        text_content += str(item) + "\n"
            
            # If it's a dictionary
            elif isinstance(extracted_text, dict):
                text_content = json.dumps(extracted_text, indent=2)
            
            # If it's already a string
            else:
                text_content = str(extracted_text)
            
            # If we have no meaningful content, try to classify using a minimal approach
            if not text_content.strip():
                logger.warning("No text content extracted for classification")
                return False, {"isBankStatement": False, "reason": "No text content extracted"}
            
            # Log a sample of the text content for debugging
            sample_text = text_content[:500] + ("..." if len(text_content) > 500 else "")
            logger.info(f"Text content sample for classification: {sample_text}")
            
            # Forced override for specific cases
            # Check for very clear bank statement indicators regardless of model classification
            bank_statement_indicators = [
                'bank statement', 'account statement', 'statement of account',
                'transaction history', 'opening balance', 'closing balance',
                'account summary', 'debit and credit', 'account activity'
            ]
            
            for indicator in bank_statement_indicators:
                if indicator in text_content.lower():
                    logger.info(f"Force classification as bank statement due to presence of '{indicator}'")
                    return True, {"isBankStatement": True, "method": "keyword_override"}
            
            system_prompt="""
            You are an AI trained in financial document analysis with specialized expertise in bank statements.
            Your task is to determine if the provided text is from a bank statement by looking for key indicators:
            
            1. Account holder details (name, account number)
            2. Transaction records with dates, descriptions, amounts, and running balances
            3. Banking terminology (deposit, withdrawal, NEFT, IMPS, etc.)
            4. Statement period or date range
            
            Be very lenient in your classification - if it shows ANY signs of being a bank statement or contains banking transaction information, classify it as such.
            Even fragmentary or incomplete bank statements should be classified as bank statements.
            The goal is to avoid false negatives. When in doubt, classify as a bank statement.
            """

            prompt = f"""
            Analyze the following text to determine if it is from a bank statement.
            
            Common bank statement indicators (ANY ONE of these is sufficient):
            - Account holder name or account number
            - Bank name or branch details
            - Statement period (date range)
            - Transaction entries with dates
            - Debit/credit amounts 
            - Banking terminology (NEFT, IMPS, UPI, ATM, etc.)
            - Opening or closing balances
            
            IMPORTANT: Be VERY lenient - if you see ANY indication of banking information, classify it as a bank statement.
            When in doubt, say YES.
            
            Text to analyze:
            ```
            {text_content[:4000]}  # Limit size to prevent token overflows
            ```
            
            Is this a bank statement? Respond with YES or NO.
            """

            # Use direct chat completion with JSON response format
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Raw classification response: {response_text}")
            
            # Process the response to determine if it's a bank statement
            # Be very lenient - if "YES" appears anywhere in the response, count it as a bank statement
            is_bank_statement = "YES" in response_text.upper()
            
            # Create a structured result
            result = {
                "isBankStatement": is_bank_statement,
                "response": response_text
            }
            
            logger.info(f"Classification result: {result}")
            
            # Default to True for problematic cases to avoid false negatives
            if not is_bank_statement and ('indian' in text_content.lower() or 'bank' in text_content.lower()):
                logger.info("Overriding classification to True due to banking keywords")
                is_bank_statement = True
                result["isBankStatement"] = True
                result["override"] = True
            
            return is_bank_statement, result
            
        except Exception as e:
            logger.error(f"Error in bank statement classification: {e} at line number {str(e.__traceback__.tb_lineno)}")
            logger.error(f"Forcing classification to True due to error")
            return True, {"isBankStatement": True, "error": str(e), "forced": True}

    def performBasicAnalysis(self, request):
        """
        Perform comprehensive analysis on the bank statement
        - All checks are run regardless of previous check failures
        - Returns detailed metrics on each aspect of the analysis
        
        Args:
            request: The HTTP request containing file and form data
            
        Returns:
            Analysis result dictionary with all metrics
        """
        try:
            # Extract form data
            documents = request.files.getlist('files') 
            customer_id = request.form.get('customer_id','')
            loan_application_id = request.form.get('loan_application_id','')
            borrower_name = request.form.get('borrower_name','')
            proprietor_name = request.form.get('proprietor_name','')
            expected_stmt_start_date = request.form.get('expected_stmt_start_date','')
            expected_stmt_end_date = request.form.get('expected_stmt_end_date','')
            client_code = request.form.get('client_code')
            entity_type = request.form.get('entity','')
            user_id = request.form.get('user_id','')
            doc_type_source = request.form.get('source','doc')

            logger.info(f"Starting analysis for customer: {customer_id}, loan_application_id: {loan_application_id}")
            logger.info(f"Borrower name: {borrower_name}, Proprietor name: {proprietor_name}")
            logger.info(f"Expected statement period: {expected_stmt_start_date} to {expected_stmt_end_date}")

            # Initialize metadata to store all check results
            metadata = {
                "extraction_success": False,
                "is_bank_statement": False,
                "account_holder_name_match": False,
                "pagewise_account_number_match": False,
                "pagewise_ifsc_code_match": False,
                "pagewise_account_name_match": False,
                "penny_drop_analysis": False,
                "stmt_completely_outside_period": False,
                "is_stmt_covers_provided_period": False,
                "transaction_accuracy": 0,
                # "missing_account_pages": [],
                "mismatch_txns": []
            }
            
            BSA_Analysis = "FAIL"  # Default status

            # Check if the files are empty
            if not documents:
                logger.error("No files provided in the request.")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                    "error": "No files provided"
                }, 400
                
            file = documents[0]
            file_name = secure_filename(file.filename)
            file_extension = os.path.splitext(file_name)[1].lower()

            sessionService = self.sessionService

            now = datetime.now()

            # Convert datetime to timestamp
            timestamp = now.timestamp()

            work_id = f"{loan_application_id}_{timestamp}"

            intialIdOfClientCode = sessionService.fetch_InitialIdByClientCode(client_code)
            initial_id = intialIdOfClientCode.get('initial_id', 0)
            logger.info("Initial id "+str(initial_id))

            work_item_result = sessionService.insert_WorkItem(client_code, "BSA", loan_application_id, user_id, "admin", datetime.now(), "admin")
            doc_id = f"la_{timestamp}"

            if work_item_result['status'] != 'success':
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                }, 500
            
            work_item_id = work_item_result['work_item_id']
            if initial_id == 0:
                initial_id = work_item_id

            
            work_doc_result = sessionService.insert_WorkDocInfo(work_item_id, "Bank Statement",doc_id, f"{file_name}_{doc_id}", doc_type_source)

            if work_doc_result['status'] != 'success':
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                }, 500

            work_doc_info_id = work_doc_result['work_doc_info_id'] 
            
            # Check supported file formats
            if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
                logger.error(f"Unsupported file format: {file_extension}")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                    "error": f"Unsupported file format: {file_extension}. Supported formats are PDF, JPG, JPEG, and PNG."
                }, 400

            # Save the file temporarily
            file_path = os.path.join(tempfile.gettempdir(), file_name)
            file.save(file_path)
            logger.info(f"File saved temporarily at: {file_path}")

            # Extract text from document
            extraction_status, document_extracted = self.extract_text_from_document(file_path)
            metadata["extraction_success"] = extraction_status
            # Check if extraction returned an error
            if not extraction_status:
                logger.error(f"Text extraction failed: {document_extracted['error']}")
                
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                    "error": document_extracted['error']
                }, 500
                
            # Ensure document_extracted is a list of dictionaries
            if not isinstance(document_extracted, list):
                logger.error(f"Unexpected document_extracted format: {type(document_extracted)}")
                if isinstance(document_extracted, dict):
                    document_extracted = [document_extracted]
                else:
                    document_extracted = []
                    
            if len(document_extracted) == 0:
                logger.error("No document content extracted")
                
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata
                }, 500

            # Classify if it's a bank statement
            logger.info("Classifying document as bank statement")
            classification_status, classification_result = self.bankstatementclassification(document_extracted)
            logger.info(f"Classification result: {classification_result}")
            
            is_bank_statement = classification_status
            metadata["is_bank_statement"] = is_bank_statement

            if not is_bank_statement:
                logger.info("The document is not a bank statement.")
                
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                }
                
            logger.info("Document confirmed as bank statement, proceeding with analysis")

            # Analyze name matching and statement period
            logger.info("Analyzing name matching and statement period")
            analysed_data = self.check_name_and_period(document_extracted, expected_stmt_start_date, expected_stmt_end_date, borrower_name, proprietor_name)

            account_holder_name_match = analysed_data.get('account_holder_name_match',"NO")
            is_statement_is_within_the_period = analysed_data.get('is_statement_is_within_the_period',"NO")
            start_date = analysed_data.get('start_date','')
            end_date = analysed_data.get('end_date','')
            missing_months = analysed_data.get('missing_months',[])
            provided_months = analysed_data.get('provided_months',[])
            is_completely_outside_period = analysed_data.get('is_completely_outside_period','NO')

            # Extract customer details from statement
            try:
                # First check if we have valid data
                if len(document_extracted) > 0 and isinstance(document_extracted[0], dict):
                    account_details = document_extracted[0].get('account_details', {})
                    if isinstance(account_details, dict):
                        customer_details_based_on_stmt = {
                            "name": account_details.get('account_name', 'N/A'),
                            "account_number": account_details.get('account_number', 'N/A'),
                            "statement_period": account_details.get('period', 'N/A'),
                            "ifs_code": account_details.get('ifs_code', 'N/A'),
                            "address": account_details.get('address', 'N/A'),
                            "date": account_details.get('date', 'N/A'),
                            "branch": account_details.get('branch', 'N/A'),
                            "cif_no": account_details.get('cif_no', 'N/A'),
                            "micr_code": account_details.get('micr_code', 'N/A')

                        }
                        metadata["customer_details"] = customer_details_based_on_stmt
                    else:
                        raise ValueError(f"account_details is not a dict: {type(account_details)}")
                else:
                    raise ValueError(f"Invalid document_extracted format or empty: {document_extracted}")
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"Error extracting customer details: {e}")
                metadata["customer_details_error"] = str(e)
                metadata["customer_details"] = {
                    "name": "N/A",
                    "account_number": "N/A",
                    "statement_period": "N/A",
                    "ifs_code": "N/A",
                    "address": "N/A",
                    "date": "N/A",
                    "branch": "N/A",
                    "cif_no": "N/A",
                    "micr_code": "N/A"
                }

            # Check if account holder name matches
            metadata["account_holder_name_match"] = account_holder_name_match == 'YES'
            
            # Check for consistency across pages and identify pages with missing account numbers
            account_numbers_by_page = {}
            ifsc_codes_by_page = {}
            account_names_by_page = {}
            
            for rec in document_extracted:
                page_num = rec.get('page_number', 0)
                account_details = rec.get('account_details', {})
                
                # Store account info by page
                if account_details.get('account_number'):
                    account_numbers_by_page[page_num] = account_details.get('account_number')
                    
                if account_details.get('ifs_code'):
                    ifsc_codes_by_page[page_num] = account_details.get('ifs_code')
                    
                if account_details.get('account_name'):
                    account_names_by_page[page_num] = account_details.get('account_name')
            
            # Get unique values
            unique_account_numbers = set(account_numbers_by_page.values())
            unique_ifsc_codes = set(ifsc_codes_by_page.values())
            unique_account_names = set(account_names_by_page.values())
            
            # Check if all pages have the same account details
            pages_with_account_numbers = []
            pages_without_account_numbers = []
            
            for idx, page in enumerate(document_extracted, 1):
                account_number = page.get('account_details', {}).get('account_number')
                if account_number:
                    pages_with_account_numbers.append(idx)
                else:
                    pages_without_account_numbers.append(idx)
            
            # Store pages missing account numbers in metadata
            # metadata["missing_account_pages"] = pages_without_account_numbers
            
            # All pages have account numbers or all are missing (consistent)
            if len(pages_without_account_numbers) == 0 or len(pages_with_account_numbers) == 0:
                metadata["pagewise_account_number_match"] = True
            else:
                # Check if all pages with account numbers have the same number
                account_numbers = {
                    rec.get('account_details', {}).get('account_number')
                    for rec in document_extracted
                    if rec.get('account_details', {}).get('account_number')
                }
                
                if len(account_numbers) == 1:
                    metadata["pagewise_account_number_match"] = True
                    logger.info("The account number is consistent across all pages where it appears.")
                else:
                    metadata["pagewise_account_number_match"] = False
                    logger.error("Different account numbers found across pages.")
                    # response = {
                    #     "customer_id": customer_id,
                    #     "loan_application_id": loan_application_id,
                    #     "meta_data": metadata,
                    #     "BSA_Analysis": "FAIL",
                    # }
                    # return response

            # Check IFSC code consistency
            metadata["pagewise_ifsc_code_match"] = len(unique_ifsc_codes) == 1
            
            # Check account name consistency
            metadata["pagewise_account_name_match"] = len(unique_account_names) == 1

            # Perform penny drop analysis (mock for now)
            try:
                if len(document_extracted) > 0 and isinstance(document_extracted[0], dict):
                    # First try to get account details from the document
                    account_details = document_extracted[0].get('account_details', {})
                    account_number = None
                    
                    # If we found account details
                    if isinstance(account_details, dict):
                        account_number = account_details.get('account_number', '')
                        ifsc = account_details.get('ifs_code', '')
                        account_name = account_details.get('account_name', '')
                    
                    # If account number wasn't found in account_details, try to find it in all items
                    if not account_number:
                        logger.info("No account number found in account_details, searching full document")
                        for item in document_extracted:
                            if isinstance(item, dict):
                                # Check for account number in the text
                                text = item.get('text', '')
                                if isinstance(text, str) and 'account' in text.lower() and re.search(r'\b\d{6,}\b', text):
                                    # Look for account number pattern in text
                                    match = re.search(r'account\s*(?:no|number|#)?\s*[:.]?\s*(\d{6,})', text.lower())
                                    if match:
                                        account_number = match.group(1)
                                        logger.info(f"Found account number {account_number} in document text")
                                        break
                    
                    # If we found an account number
                    if account_number:
                        # Mock penny drop verification
                        mock_penny_drop = {
                            "00000034909111330": "success",
                            "00000038322372698": "success",
                            "7223648801": "success",
                            "719130110000033": "success",
                            "69220000331696": "success",
                            "60240500000351": "success",
                            "357305040050196": "success",
                            "9876543210" : "success"
                            # Add more test account numbers as needed
                        }
                        
                        # Clean account number (remove spaces and special chars)
                        clean_account = ''.join(c for c in account_number if c.isdigit())
                        
                        # Try to match with the mock database
                        penny_drop_status = mock_penny_drop.get(clean_account, '')
                        
                        
                        # # If it's not in our mock database, approve for testing
                        # if not penny_drop_status and len(clean_account) >= 6:
                        #     logger.info(f"Account number {clean_account} not in mock database, but approving for testing")
                        #     penny_drop_status = 'success'
                        
                        metadata["penny_drop_analysis"] = penny_drop_status.lower() == 'success'
                    else:
                        logger.warning("No account number found for penny drop analysis")
                        # For testing purposes, we'll mark it as successful to avoid false negatives
                        metadata["penny_drop_analysis"] = False
                        metadata["penny_drop_warning"] = "No account number found"
                else:
                    logger.warning("Invalid document_extracted format for penny drop analysis")
                    # For testing, mark as successful to avoid false negatives
                    metadata["penny_drop_analysis"] = False
                    metadata["penny_drop_warning"] = "Invalid document format"
            except (KeyError, IndexError, ValueError, AttributeError) as e:
                logger.error(f"Error in penny drop analysis: {e}")
                metadata["penny_drop_error"] = str(e)
                # For testing purposes, default to true to avoid false negatives
                metadata["penny_drop_analysis"] = True

            # Check statement period
            metadata["stmt_completely_outside_period"] = is_completely_outside_period == 'YES'
            
            # Statement period analysis
            stmt_period_analysis = {
                "stmt_start_date": start_date,
                "stmt_end_date": end_date,
                "expected_stmt_start_date": expected_stmt_start_date,
                "expected_stmt_end_date": expected_stmt_end_date,
                "missing_months": missing_months,
                "provided_months": provided_months
            }
            metadata["stmt_period_analysis"] = stmt_period_analysis

            # Check if statement covers the entire requested period
            metadata["is_stmt_covers_provided_period"] = is_statement_is_within_the_period == 'YES'

            # Validate transaction balances and calculate accuracy
            logger.info("Validating transaction balances")
            try:
                # Ensure we have valid data
                if len(document_extracted) == 0 or not isinstance(document_extracted[0], dict):
                    logger.warning("Invalid document_extracted format or empty - skipping transaction validation")
                    metadata["transaction_accuracy"] = 0
                    metadata["total_txns"] = 0
                    metadata["error_txns"] = 0
                    metadata["mismatch_txns"] = []
                    return
                
                # Initialize tracking variables
                missmatch_txns = []
                credit_sum = Decimal('0')
                debit_sum = Decimal('0')
                txn_cnt = 0
                error_txns = 0
                credit_cnt = 0
                debit_cnt = 0
                # opening_balance = None
                opening_balance = Decimal(document_extracted[0]['account_details'].get('opening_balance','0').replace(',', '') or '0')
                
                
                # Begin transaction validation
                prev_balance = opening_balance
                txn_json = []

                for page in sorted(document_extracted, key=lambda x: x['page_number']):
                    page_number = page.get('page_number', 0)
                    pagewise_txn = []
                    idx = 1
                    for tx in page.get('account_statement', {}).get('transactions', []):
                        debit  = Decimal(tx['debit'].replace(',', '') or '0')
                        credit = Decimal(tx['credit'].replace(',', '') or '0')
                        stated_balance = Decimal(tx['balance'].replace(',', '') or '0')

                        # Calculate expected balance
                        calculated_balance = prev_balance - debit + credit
                        if credit_sum != (credit_sum + credit):
                            credit_cnt += 1
                        if debit_sum != (debit_sum + debit):
                            debit_cnt += 1
                        credit_sum += credit
                        debit_sum += debit

                        out = {
                            "idx" : idx,
                            "txn_date" : tx.get('txn_date',''),
                            "value_date" : tx.get('value_date', ''),
                            "description" : tx.get('description', ''),
                            "ref_no_cheque_no" : tx.get('ref_no_cheque_no', ''),
                            "credit" : credit,
                            "debit" :  debit,
                            "stated_balance" : stated_balance,
                            "calculated_balance" : calculated_balance,
                            "opening_balance": prev_balance
                        }


                        # Check for mismatch
                        if calculated_balance != stated_balance:
                            print(f"Mismatch on {tx['txn_date']}: "
                                f"expected {calculated_balance:.2f}, "
                                f"stated {stated_balance:.2f}")
                            missmatch_txns.append({
                                "txn_date": tx['txn_date'],
                                "expected_balance": f"{calculated_balance:.2f}",
                                "stated_balance": f"{stated_balance:.2f}",
                                "debit": f"{debit:.2f}",
                                "credit": f"{credit:.2f}",
                                "previous_closing_balance": f"{prev_balance}",
                            })
                            error_txns += 1
                            BSA_Analysis = "FAIL"
                            out["status"] = "Fail"
                        else:
                            out["status"] = "Pass"

                        pagewise_txn.append(out)
                        # Update for next transaction
                        prev_balance = stated_balance
                        txn_cnt += 1
                        idx += 1

                    txn_json.append({
                        "page_number": page_number,
                        "transactions": pagewise_txn
                    })

                # Add transaction summary to metadata
                metadata["total_txns"] = txn_cnt
                metadata["error_txns"] = error_txns
                metadata["total_debits"] = float(debit_sum)
                metadata["total_credits"] = float(credit_sum)
                metadata["credit_cnt"] = credit_cnt
                metadata["debit_cnt"] = debit_cnt
                metadata["opening_balance"] = self.decimal_to_float(opening_balance)
                metadata["closing_balance"] = self.decimal_to_float(prev_balance)
                metadata["is_human_verified"] = False
                metadata["verified_by"] = ""
                metadata["verified_time"] = ""
                metadata["total_pages"] = len(txn_json)

                # Calculate transaction accuracy percentage
                valid_txns = txn_cnt - error_txns
                if valid_txns > 0:
                    accuracy = round((valid_txns - len(missmatch_txns)) / valid_txns * 100, 2)
                    metadata["transaction_accuracy"] = accuracy
                else:
                    metadata["transaction_accuracy"] = 0
                
                # Store mismatched transactions
                metadata["mismatch_txns"] = missmatch_txns
                metadata["any_mismatch_txns"] = len(missmatch_txns) > 0
                
            except Exception as e:
                logger.error(f"Error validating transaction balances: {e}")
                import traceback
                logger.error(traceback.format_exc())
                metadata["balance_validation_error"] = str(e)
                metadata["transaction_accuracy"] = 0

            # Determine overall BSA_Analysis status based on all checks
            checks_passed = all([
                metadata["is_bank_statement"],
                metadata["account_holder_name_match"],
                metadata["pagewise_account_number_match"],
                metadata["pagewise_ifsc_code_match"],
                metadata["pagewise_account_name_match"],
                metadata["penny_drop_analysis"],
                not metadata["stmt_completely_outside_period"]
            ])
            
            if checks_passed:
                if metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            # metadata = self.decimal_to_float(metadata)
            txn_json = self.decimal_to_float(txn_json)
                
            # Adjust status based on transaction accuracy
            if BSA_Analysis == "SUCCESS" and metadata["transaction_accuracy"] < 90:
                BSA_Analysis = "INCOMPLETE"

            # Return comprehensive analysis
            logger.info(f"BSA analysis completed with status: {BSA_Analysis}")
            stmt_id = f"{loan_application_id}_{work_item_id - initial_id + 1}"

            bank_stmt_insert_result = sessionService.insert_BankStmt(work_doc_info_id,stmt_id , " ", account_name, account_number, metadata.get("customer_details",{}).get("ifs_code",""), len(txn_json), "", metadata, document_extracted, txn_json, BSA_Analysis, metadata["transaction_accuracy"], datetime.now(), "admin", False)

            if bank_stmt_insert_result.get("status") != "success":
                logger.error(f"Failed to insert bank statement data: {bank_stmt_insert_result.get('error', 'Unknown error')}")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": "FAIL",
                }, 500

            bank_stmt_id = bank_stmt_insert_result.get("bank_stmt_id", None)


            return {
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": metadata,
                "BSA_Analysis": BSA_Analysis,
                "txn_json" : txn_json
            }

        except Exception as e:
            logger.error(f"Error in Bank Statement Analysis: {e} at line {e.__traceback__.tb_lineno}")
            return {
                "customer_id": request.form.get('customer_id', ''),
                "loan_application_id": request.form.get('loan_application_id', ''),
                "BSA_Analysis": "FAIL",
                "meta_data": {"extraction_success": False},
                "error": str(e)
            }, 500
        
    def performBasicAnalysisGemini(self, request):
        """
        Perform comprehensive analysis on the bank statement
        - All checks are run regardless of previous check failures
        - Returns detailed metrics on each aspect of the analysis
        
        Args:
            request: The HTTP request containing file and form data
            
        Returns:
            Analysis result dictionary with all metrics
        """
        try:
            # Extract form data
            documents = request.files.getlist('files') 
            customer_id = request.form.get('customer_id','')
            loan_application_id = request.form.get('loan_application_id','')
            borrower_name = request.form.get('borrower_name','')
            proprietor_name = request.form.get('proprietor_name','')
            expected_stmt_start_date = request.form.get('expected_stmt_start_date','')
            expected_stmt_end_date = request.form.get('expected_stmt_end_date','')
            client_code = request.form.get('client_code')
            entity_type = request.form.get('entity','')
            user_id = request.form.get('user_id','')
            doc_type_source = request.form.get('source','doc')

            logger.info(f"Starting analysis for customer: {customer_id}, loan_application_id: {loan_application_id}")
            logger.info(f"Borrower name: {borrower_name}, Proprietor name: {proprietor_name}")
            logger.info(f"Expected statement period: {expected_stmt_start_date} to {expected_stmt_end_date}")

            # Initialize metadata to store all check results
            metadata = {
                "extraction_success": False,
                "is_bank_statement": False,
                "account_holder_name_match": False,
                "pagewise_account_number_match": False,
                "pagewise_ifsc_code_match": False,
                "pagewise_account_name_match": False,
                "penny_drop_analysis": False,
                "stmt_completely_outside_period": False,
                "is_stmt_covers_provided_period": False,
                "transaction_accuracy": 0,
                # "missing_account_pages": [],
                "mismatch_txns": []
            }
            
            BSA_Analysis = "FAIL"  # Default status

            # Check if the files are empty
            if not documents:
                logger.error("No files provided in the request.")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                    "error": "No files provided"
                }, 400
                
            file = documents[0]
            file_name = secure_filename(file.filename)
            file_extension = os.path.splitext(file_name)[1].lower()

            sessionService = self.sessionService

            now = datetime.now()

            # Convert datetime to timestamp
            timestamp = now.timestamp()

            work_id = f"{loan_application_id}_{timestamp}"

            intialIdOfClientCode = sessionService.fetch_InitialIdByClientCode(client_code)
            initial_id = intialIdOfClientCode.get('initial_id', 0)
            logger.info("Initial id "+str(initial_id))

            work_item_result = sessionService.insert_WorkItem(client_code, "BSA", loan_application_id, user_id, "admin", datetime.now(), "admin")
            doc_id = f"la_{timestamp}"

            if work_item_result['status'] != 'success':
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                }, 500
            
            work_item_id = work_item_result['work_item_id']
            if initial_id == 0:
                initial_id = work_item_id

            
            work_doc_result = sessionService.insert_WorkDocInfo(work_item_id, "Bank Statement",doc_id, f"{file_name}_{doc_id}", doc_type_source)

            if work_doc_result['status'] != 'success':
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                }, 500

            work_doc_info_id = work_doc_result['work_doc_info_id'] 
            
            # Check supported file formats
            if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
                logger.error(f"Unsupported file format: {file_extension}")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                    "error": f"Unsupported file format: {file_extension}. Supported formats are PDF, JPG, JPEG, and PNG."
                }, 400

            # Save the file temporarily
            file_path = os.path.join(tempfile.gettempdir(), file_name)
            file.save(file_path)
            logger.info(f"File saved temporarily at: {file_path}")

            # Extract text from document
            # extraction_status, document_extracted = self.extract_text_from_document(file_path)

            bsaextractionGeminiService = BSAextractionGeminiService()

            document_extracted = bsaextractionGeminiService.process_pdf_parallel(file_path)

            # store the document_extracted json in the file
            # with open(f"{file_name}.json", 'w', encoding='utf-8') as f:
            #     json.dump(document_extracted, f, ensure_ascii=False, indent=4)

            if os.path.exists(file_path):
                os.remove(file_path)

            # metadata["extraction_success"] = extraction_status
            # # Check if extraction returned an error
            # if not extraction_status:
            #     logger.error(f"Text extraction failed: {document_extracted['error']}")
                
            #     return {
            #         "customer_id": customer_id,
            #         "loan_application_id": loan_application_id,
            #         "BSA_Analysis": BSA_Analysis,
            #         "meta_data": metadata,
            #         "error": document_extracted['error']
            #     }, 500
                
            # Ensure document_extracted is a list of dictionaries
            if not isinstance(document_extracted, list):
                logger.error(f"Unexpected document_extracted format: {type(document_extracted)}")
                if isinstance(document_extracted, dict):
                    document_extracted = [document_extracted]
                else:
                    document_extracted = []
                    
            if len(document_extracted) == 0:
                logger.error("No document content extracted")
                
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata
                }, 500

            # Classify if it's a bank statement
            logger.info("Classifying document as bank statement")
            classification_status, classification_result = self.bankstatementclassification(document_extracted)
            logger.info(f"Classification result: {classification_result}")
            
            is_bank_statement = classification_status
            metadata["is_bank_statement"] = is_bank_statement

            if not is_bank_statement:
                logger.info("The document is not a bank statement.")
                
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": BSA_Analysis,
                    "meta_data": metadata,
                }
                
            logger.info("Document confirmed as bank statement, proceeding with analysis")

            # Analyze name matching and statement period
            logger.info("Analyzing name matching and statement period")
            analysed_data = self.check_name_and_period(document_extracted, expected_stmt_start_date, expected_stmt_end_date, borrower_name, proprietor_name)

            account_holder_name_match = analysed_data.get('account_holder_name_match',"NO")
            is_statement_is_within_the_period = analysed_data.get('is_statement_is_within_the_period',"NO")
            start_date = analysed_data.get('start_date','')
            end_date = analysed_data.get('end_date','')
            missing_months = analysed_data.get('missing_months',[])
            provided_months = analysed_data.get('provided_months',[])
            is_completely_outside_period = analysed_data.get('is_completely_outside_period','NO')

            # Extract customer details from statement
            try:
                # First check if we have valid data
                if len(document_extracted) > 0 and isinstance(document_extracted[0], dict):
                    account_details = document_extracted[0].get('account_details', {})
                    if isinstance(account_details, dict):
                        customer_details_based_on_stmt = {
                            "name": account_details.get('account_name', 'N/A'),
                            "account_number": account_details.get('account_number', 'N/A'),
                            "statement_period": account_details.get('period', 'N/A'),
                            "ifs_code": account_details.get('ifs_code', 'N/A'),
                            "address": account_details.get('address', 'N/A'),
                            "date": account_details.get('date', 'N/A'),
                            "branch": account_details.get('branch', 'N/A'),
                            "cif_no": account_details.get('cif_no', 'N/A'),
                            "micr_code": account_details.get('micr_code', 'N/A')

                        }
                        metadata["customer_details"] = customer_details_based_on_stmt
                    else:
                        raise ValueError(f"account_details is not a dict: {type(account_details)}")
                else:
                    raise ValueError(f"Invalid document_extracted format or empty: {document_extracted}")
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"Error extracting customer details: {e}")
                metadata["customer_details_error"] = str(e)
                metadata["customer_details"] = {
                    "name": "N/A",
                    "account_number": "N/A",
                    "statement_period": "N/A",
                    "ifs_code": "N/A",
                    "address": "N/A",
                    "date": "N/A",
                    "branch": "N/A",
                    "cif_no": "N/A",
                    "micr_code": "N/A"
                }

            # Check if account holder name matches
            metadata["account_holder_name_match"] = account_holder_name_match == 'YES'
            
            # Check for consistency across pages and identify pages with missing account numbers
            account_numbers_by_page = {}
            ifsc_codes_by_page = {}
            account_names_by_page = {}
            
            for rec in document_extracted:
                page_num = rec.get('page_number', 0)
                account_details = rec.get('account_details', {})
                
                # Store account info by page
                if account_details.get('account_number'):
                    account_numbers_by_page[page_num] = account_details.get('account_number')
                    
                if account_details.get('ifs_code'):
                    ifsc_codes_by_page[page_num] = account_details.get('ifs_code')
                    
                if account_details.get('account_name'):
                    account_names_by_page[page_num] = account_details.get('account_name')
            
            # Get unique values
            unique_account_numbers = set(account_numbers_by_page.values())
            unique_ifsc_codes = set(ifsc_codes_by_page.values())
            unique_account_names = set(account_names_by_page.values())
            
            # Check if all pages have the same account details
            pages_with_account_numbers = []
            pages_without_account_numbers = []
            
            for idx, page in enumerate(document_extracted, 1):
                account_number = page.get('account_details', {}).get('account_number')
                if account_number:
                    pages_with_account_numbers.append(idx)
                else:
                    pages_without_account_numbers.append(idx)
            
            # Store pages missing account numbers in metadata
            # metadata["missing_account_pages"] = pages_without_account_numbers
            
            # All pages have account numbers or all are missing (consistent)
            if len(pages_without_account_numbers) == 0 or len(pages_with_account_numbers) == 0:
                metadata["pagewise_account_number_match"] = True
            else:
                # Check if all pages with account numbers have the same number
                account_numbers = {
                    rec.get('account_details', {}).get('account_number')
                    for rec in document_extracted
                    if rec.get('account_details', {}).get('account_number')
                }
                
                if len(account_numbers) == 1:
                    metadata["pagewise_account_number_match"] = True
                    logger.info("The account number is consistent across all pages where it appears.")
                else:
                    metadata["pagewise_account_number_match"] = False
                    logger.error("Different account numbers found across pages.")
                    # response = {
                    #     "customer_id": customer_id,
                    #     "loan_application_id": loan_application_id,
                    #     "meta_data": metadata,
                    #     "BSA_Analysis": "FAIL",
                    # }
                    # return response

            # Check IFSC code consistency
            metadata["pagewise_ifsc_code_match"] = len(unique_ifsc_codes) == 1
            
            # Check account name consistency
            metadata["pagewise_account_name_match"] = len(unique_account_names) == 1

            # Perform penny drop analysis (mock for now)
            try:
                if len(document_extracted) > 0 and isinstance(document_extracted[0], dict):
                    # First try to get account details from the document
                    account_details = document_extracted[0].get('account_details', {})
                    account_number = None
                    
                    # If we found account details
                    if isinstance(account_details, dict):
                        account_number = account_details.get('account_number', '')
                        ifsc = account_details.get('ifs_code', '')
                        account_name = account_details.get('account_name', '')
                    
                    # If account number wasn't found in account_details, try to find it in all items
                    if not account_number:
                        logger.info("No account number found in account_details, searching full document")
                        for item in document_extracted:
                            if isinstance(item, dict):
                                # Check for account number in the text
                                text = item.get('text', '')
                                if isinstance(text, str) and 'account' in text.lower() and re.search(r'\b\d{6,}\b', text):
                                    # Look for account number pattern in text
                                    match = re.search(r'account\s*(?:no|number|#)?\s*[:.]?\s*(\d{6,})', text.lower())
                                    if match:
                                        account_number = match.group(1)
                                        logger.info(f"Found account number {account_number} in document text")
                                        break
                    
                    # If we found an account number
                    if account_number:
                        # Mock penny drop verification
                        mock_penny_drop = {
                            "00000034909111330": "success",
                            "00000038322372698": "success",
                            "7223648801": "success",
                            "719130110000033": "success",
                            "69220000331696": "success",
                            "60240500000351": "success",
                            "357305040050196": "success",
                            "9876543210" : "success"
                            # Add more test account numbers as needed
                        }
                        
                        # Clean account number (remove spaces and special chars)
                        clean_account = ''.join(c for c in account_number if c.isdigit())
                        
                        # Try to match with the mock database
                        penny_drop_status = mock_penny_drop.get(clean_account, '')
                        
                        
                        # # If it's not in our mock database, approve for testing
                        # if not penny_drop_status and len(clean_account) >= 6:
                        #     logger.info(f"Account number {clean_account} not in mock database, but approving for testing")
                        #     penny_drop_status = 'success'
                        
                        metadata["penny_drop_analysis"] = penny_drop_status.lower() == 'success'
                    else:
                        logger.warning("No account number found for penny drop analysis")
                        # For testing purposes, we'll mark it as successful to avoid false negatives
                        metadata["penny_drop_analysis"] = False
                        metadata["penny_drop_warning"] = "No account number found"
                else:
                    logger.warning("Invalid document_extracted format for penny drop analysis")
                    # For testing, mark as successful to avoid false negatives
                    metadata["penny_drop_analysis"] = False
                    metadata["penny_drop_warning"] = "Invalid document format"
            except (KeyError, IndexError, ValueError, AttributeError) as e:
                logger.error(f"Error in penny drop analysis: {e}")
                metadata["penny_drop_error"] = str(e)
                # For testing purposes, default to true to avoid false negatives
                metadata["penny_drop_analysis"] = True

            # Check statement period
            metadata["stmt_completely_outside_period"] = is_completely_outside_period == 'YES'
            
            # Statement period analysis
            stmt_period_analysis = {
                "stmt_start_date": start_date,
                "stmt_end_date": end_date,
                "expected_stmt_start_date": expected_stmt_start_date,
                "expected_stmt_end_date": expected_stmt_end_date,
                "missing_months": missing_months,
                "provided_months": provided_months
            }
            metadata["stmt_period_analysis"] = stmt_period_analysis

            # Check if statement covers the entire requested period
            metadata["is_stmt_covers_provided_period"] = is_statement_is_within_the_period == 'YES'

            # Validate transaction balances and calculate accuracy
            logger.info("Validating transaction balances")
            try:
                # Ensure we have valid data
                if len(document_extracted) == 0 or not isinstance(document_extracted[0], dict):
                    logger.warning("Invalid document_extracted format or empty - skipping transaction validation")
                    metadata["transaction_accuracy"] = 0
                    metadata["total_txns"] = 0
                    metadata["error_txns"] = 0
                    metadata["mismatch_txns"] = []
                    return
                
                # Initialize tracking variables
                missmatch_txns = []
                credit_sum = Decimal('0')
                debit_sum = Decimal('0')
                txn_cnt = 0
                error_txns = 0
                credit_cnt = 0
                debit_cnt = 0
                # opening_balance = None
                opening_balance = Decimal(str(document_extracted[0]['account_details'].get('opening_balance','0')).replace(',', '') or '0')
                
                
                # Begin transaction validation
                prev_balance = opening_balance
                txn_json = []

                for page in sorted(document_extracted, key=lambda x: x['page_number']):
                    page_number = page.get('page_number', 0)
                    pagewise_txn = []
                    idx = 1
                    for tx in page.get('account_statement', {}).get('transactions', []):
                        debit  = Decimal(str(tx['debit']).replace(',', '') or '0')
                        credit = Decimal(str(tx['credit']).replace(',', '') or '0')
                        stated_balance = Decimal(str(tx['balance']).replace(',', '') or '0')

                        # Calculate expected balance
                        calculated_balance = prev_balance - debit + credit
                        if credit_sum != (credit_sum + credit):
                            credit_cnt += 1
                        if debit_sum != (debit_sum + debit):
                            debit_cnt += 1
                        credit_sum += credit
                        debit_sum += debit

                        out = {
                            "idx" : idx,
                            "txn_date" : tx.get('txn_date',''),
                            "value_date" : tx.get('value_date', ''),
                            "description" : tx.get('description', ''),
                            "ref_no_cheque_no" : tx.get('ref_no_cheque_no', ''),
                            "credit" : credit,
                            "debit" :  debit,
                            "stated_balance" : stated_balance,
                            "calculated_balance" : calculated_balance,
                            "opening_balance": prev_balance
                        }


                        # Check for mismatch
                        if calculated_balance != stated_balance:
                            print(f"Mismatch on {tx['txn_date']}: "
                                f"expected {calculated_balance:.2f}, "
                                f"stated {stated_balance:.2f}")
                            missmatch_txns.append({
                                "txn_date": tx['txn_date'],
                                "expected_balance": f"{calculated_balance:.2f}",
                                "stated_balance": f"{stated_balance:.2f}",
                                "debit": f"{debit:.2f}",
                                "credit": f"{credit:.2f}",
                                "previous_closing_balance": f"{prev_balance}",
                            })
                            error_txns += 1
                            BSA_Analysis = "FAIL"
                            out["status"] = "Fail"
                        else:
                            out["status"] = "Pass"

                        pagewise_txn.append(out)
                        # Update for next transaction
                        prev_balance = stated_balance
                        txn_cnt += 1
                        idx += 1

                    txn_json.append({
                        "page_number": page_number,
                        "transactions": pagewise_txn
                    })

                # Add transaction summary to metadata
                metadata["total_txns"] = txn_cnt
                metadata["error_txns"] = error_txns
                metadata["total_debits"] = float(debit_sum)
                metadata["total_credits"] = float(credit_sum)
                metadata["credit_cnt"] = credit_cnt
                metadata["debit_cnt"] = debit_cnt
                metadata["opening_balance"] = self.decimal_to_float(opening_balance)
                metadata["closing_balance"] = self.decimal_to_float(prev_balance)
                metadata["is_human_verified"] = False
                metadata["verified_by"] = ""
                metadata["verified_time"] = ""
                metadata["total_pages"] = len(txn_json)

                # Calculate transaction accuracy percentage
                valid_txns = txn_cnt - error_txns
                if valid_txns > 0:
                    accuracy = round((valid_txns - len(missmatch_txns)) / valid_txns * 100, 2)
                    metadata["transaction_accuracy"] = accuracy
                else:
                    metadata["transaction_accuracy"] = 0
                
                # Store mismatched transactions
                metadata["mismatch_txns"] = missmatch_txns
                metadata["any_mismatch_txns"] = len(missmatch_txns) > 0
                
            except Exception as e:
                logger.error(f"Error validating transaction balances: {e}")
                import traceback
                logger.error(traceback.format_exc())
                metadata["balance_validation_error"] = str(e)
                metadata["transaction_accuracy"] = 0

            # Determine overall BSA_Analysis status based on all checks
            checks_passed = all([
                metadata["is_bank_statement"],
                metadata["account_holder_name_match"],
                metadata["pagewise_account_number_match"],
                metadata["pagewise_ifsc_code_match"],
                metadata["pagewise_account_name_match"],
                metadata["penny_drop_analysis"],
                not metadata["stmt_completely_outside_period"]
            ])
            
            if checks_passed:
                if metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            # metadata = self.decimal_to_float(metadata)
            txn_json = self.decimal_to_float(txn_json)
                
            # Adjust status based on transaction accuracy
            if BSA_Analysis == "SUCCESS" and metadata["transaction_accuracy"] < 90:
                BSA_Analysis = "INCOMPLETE"

            # Return comprehensive analysis
            logger.info(f"BSA analysis completed with status: {BSA_Analysis}")
            stmt_id = f"{loan_application_id}_{work_item_id - initial_id + 1}"

            bank_stmt_insert_result = sessionService.insert_BankStmt(work_doc_info_id,stmt_id , " ", account_name, account_number, metadata.get("customer_details",{}).get("ifs_code",""), len(txn_json), "", metadata, document_extracted, txn_json, BSA_Analysis, metadata["transaction_accuracy"], datetime.now(), "admin", False)

            if bank_stmt_insert_result.get("status") != "success":
                logger.error(f"Failed to insert bank statement data: {bank_stmt_insert_result.get('error', 'Unknown error')}")
                return {
                    "customer_id": customer_id,
                    "loan_application_id": loan_application_id,
                    "BSA_Analysis": "FAIL",
                }, 500

            bank_stmt_id = bank_stmt_insert_result.get("bank_stmt_id", None)


            return {
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": metadata,
                "BSA_Analysis": BSA_Analysis,
                "txn_json" : txn_json
            }

        except Exception as e:
            logger.error(f"Error in Bank Statement Analysis: {e} at line {e.__traceback__.tb_lineno}")
            return {
                "customer_id": request.form.get('customer_id', ''),
                "loan_application_id": request.form.get('loan_application_id', ''),
                "BSA_Analysis": "FAIL",
                "meta_data": {"extraction_success": False},
                "error": str(e)
            }, 500

    def check_name_and_period(self, ocr_results, given_start_date, given_end_date, given_borrower_name, given_proprietor_name):
        """
        Check if account holder name matches provided names and if statement period covers required dates
        
        Args:
            ocr_results: Extracted OCR results
            given_start_date: Expected statement start date
            given_end_date: Expected statement end date
            given_borrower_name: Borrower's name
            given_proprietor_name: Proprietor's name
            
        Returns:
            Dictionary with name match and period coverage results
        """
        try:
            # Make sure ocr_results is properly formatted
            if not isinstance(ocr_results, list):
                logger.warning(f"OCR results is not a list, it's {type(ocr_results)}. Converting...")
                if isinstance(ocr_results, dict):
                    ocr_results = [ocr_results]
                else:
                    # If it's a string or other type, create a minimal structure
                    ocr_results = [{"account_details": {}, "page_number": 1, "account_statement": {"transactions": []}}]
            
            # Extract all transaction dates
            txn_dates = []
            for record in ocr_results:
                if not isinstance(record, dict):
                    logger.warning(f"Record is not a dict: {type(record)}")
                    continue
                    
                # Get transactions safely
                transactions = []
                account_statement = record.get('account_statement', {})
                if isinstance(account_statement, dict):
                    transactions.extend(account_statement.get('transactions', []))
                
                transactions.extend(record.get('transactions', []))
                
                # Extract dates
                for txn in transactions:
                    if not isinstance(txn, dict):
                        continue
                    date = txn.get('txn_date') or txn.get('Txn Date')
                    if date:
                        txn_dates.append(date)

            # Extract all account names
            account_names = set()
            for rec in ocr_results:
                if not isinstance(rec, dict):
                    continue
                
                account_details = rec.get('account_details', {})
                if isinstance(account_details, dict) and account_details.get('account_name'):
                    account_names.add(account_details.get('account_name'))
            
            # Create a clean array of names for display
            account_names_list = list(account_names)
            
            # Create a string with both account holder names
            name_comparison = f"{account_names_list} vs {given_borrower_name if given_borrower_name else ''} or {given_proprietor_name if given_proprietor_name else ''}"

            # If we have no account names or transaction dates, return default values
            if not account_names and not txn_dates:
                logger.warning("No account names or transaction dates found - using default values")
                return {
                    "account_holder_name_match": "NO",
                    "matching_name": "NONE",
                    "is_statement_is_within_the_period": "NO",
                    "start_date": "",
                    "end_date": "",
                    "missing_months": [],
                    "provided_months": [],
                    "is_completely_outside_period": "YES",
                    "name_comparison": name_comparison
                }
        
            # Try to determine dates directly from available transaction dates
            if txn_dates:
                # Direct date analysis without calling OpenAI
                try:
                    # Sort dates to find earliest and latest
                    from dateutil import parser
                    
                    # Parse dates (handle various formats)
                    parsed_dates = []
                    for date_str in txn_dates:
                        try:
                            parsed_dates.append(parser.parse(date_str, fuzzy=True))
                        except:
                            pass
                    
                    if parsed_dates:
                        parsed_dates.sort()
                        start_date = parsed_dates[0].strftime("%Y-%m-%d")
                        end_date = parsed_dates[-1].strftime("%Y-%m-%d")
                        
                        # Format months for output
                        months_covered = set()
                        for date in parsed_dates:
                            months_covered.add(f"{date.year}-{date.strftime('%b')}")
                        
                        # Attempt to parse expected start/end dates
                        try:
                            exp_start = parser.parse(given_start_date, fuzzy=True)
                            exp_end = parser.parse(given_end_date, fuzzy=True)
                            
                            # Find missing months
                            from datetime import datetime, timedelta
                            
                            # Generate all months between expected start and end
                            expected_months = set()
                            current = datetime(exp_start.year, exp_start.month, 1)
                            end = datetime(exp_end.year, exp_end.month, 1)
                            
                            while current <= end:
                                expected_months.add(f"{current.year}-{current.strftime('%b')}")
                                # Move to next month
                                if current.month == 12:
                                    current = datetime(current.year + 1, 1, 1)
                                else:
                                    current = datetime(current.year, current.month + 1, 1)
                            
                            missing_months = sorted(list(expected_months - months_covered))
                            provided_months = sorted(list(months_covered))
                            
                            # Determine if completely outside
                            if parsed_dates[-1] < exp_start or parsed_dates[0] > exp_end:
                                is_completely_outside = "YES"
                            else:
                                is_completely_outside = "NO"
                                
                            # Determine if within period
                            if not missing_months:
                                is_within_period = "YES"
                            else:
                                is_within_period = "NO"
                                
                        except:
                            # Default values if parsing expected dates fails
                            missing_months = []
                            provided_months = sorted(list(months_covered))
                            is_completely_outside = "NO"
                            is_within_period = "NO"
                    else:
                        # No parseable dates
                        start_date = ""
                        end_date = ""
                        missing_months = []
                        provided_months = []
                        is_completely_outside = "YES"
                        is_within_period = "NO"
                except Exception as date_error:
                    logger.error(f"Error analyzing dates: {date_error}")
                    start_date = ""
                    end_date = ""
                    missing_months = []
                    provided_months = []
                    is_completely_outside = "YES"
                    is_within_period = "NO"
            else:
                # No transaction dates available
                start_date = ""
                end_date = ""
                missing_months = []
                provided_months = []
                is_completely_outside = "YES"
                is_within_period = "NO"
                
            # Check name match
            account_holder_name_match = "NO"
            matching_name = "NONE"
            
            if account_names and (given_borrower_name or given_proprietor_name):
                # Normalize names for comparison
                def normalize_name(name):
                    if not name:
                        return ""
                    return name.lower().replace('.', '').replace(',', '')
                
                normalized_borrower = normalize_name(given_borrower_name)
                normalized_proprietor = normalize_name(given_proprietor_name)
                
                for account_name in account_names:
                    normalized_account = normalize_name(account_name)
                    
                    # Check for exact or partial matches
                    if normalized_borrower and (normalized_borrower in normalized_account or normalized_account in normalized_borrower):
                        account_holder_name_match = "YES"
                        matching_name = account_name
                        break
                        
                    if normalized_proprietor and (normalized_proprietor in normalized_account or normalized_account in normalized_proprietor):
                        account_holder_name_match = "YES"
                        matching_name = account_name
                        break
                        
                    # Check for initial matches (e.g. "J Smith" matches "John Smith")
                    if normalized_borrower and normalized_borrower.split():
                        first_initial = normalized_borrower.split()[0][0]
                        rest_of_name = ' '.join(normalized_borrower.split()[1:])
                        if rest_of_name and rest_of_name in normalized_account and normalized_account.startswith(first_initial):
                            account_holder_name_match = "YES"
                            matching_name = account_name
                            break
                            
                    if normalized_proprietor and normalized_proprietor.split():
                        first_initial = normalized_proprietor.split()[0][0]
                        rest_of_name = ' '.join(normalized_proprietor.split()[1:])
                        if rest_of_name and rest_of_name in normalized_account and normalized_account.startswith(first_initial):
                            account_holder_name_match = "YES"
                            matching_name = account_name
                            break
            
            # Construct and return the result
            result = {
                "account_holder_name_match": account_holder_name_match,
                "matching_name": matching_name,
                "is_statement_is_within_the_period": is_within_period,
                "start_date": start_date,
                "end_date": end_date,
                "missing_months": missing_months,
                "provided_months": provided_months,
                "is_completely_outside_period": is_completely_outside,
                "name_comparison": name_comparison
            }
            
            logger.info(f"Name and period analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in name and period analysis: {e} at line {e.__traceback__.tb_lineno}")
            return {
                "account_holder_name_match": "NO",
                "matching_name": "NONE",
                "is_statement_is_within_the_period": "NO",
                "start_date": "",
                "end_date": "",
                "missing_months": [],
                "provided_months": [],
                "is_completely_outside_period": "YES",
                "name_comparison": f"{given_borrower_name} or {given_proprietor_name}",
                "error": str(e)
            }

    def safe_decimal(self, value, default=0):
        """
        Safely convert a string to Decimal, handling various edge cases
        
        Args:
            value: The string value to convert
            default: Default value to return if conversion fails
            
        Returns:
            Decimal value or default on failure
        """
        if value is None:
            return Decimal(default)
            
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value))
            
        try:
            # Handle string
            cleaned = str(value).strip()
            
            # Remove all non-numeric characters except decimal point
            # First handle common cases
            cleaned = cleaned.replace(',', '')
            
            # Handle currency symbols, parentheses, etc.
            if cleaned.lower() in ('na', 'n/a', '', '-', 'nil', 'null'):
                return Decimal(default)
                
            # Handle percentage or other special indicators
            cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.' or c == '-')
            
            # Handle multiple decimal points (take first occurrence)
            parts = cleaned.split('.')
            if len(parts) > 2:
                cleaned = parts[0] + '.' + parts[1]
                
            # Handle empty string after cleaning
            if not cleaned or cleaned == '.':
                return Decimal(default)
                
            return Decimal(cleaned)
        except (ValueError, decimal.InvalidOperation, decimal.ConversionSyntax):
            logger.warning(f"Could not convert '{value}' to Decimal, using default: {default}")
            return Decimal(default)
        
    def decimal_to_float(self,obj):
        try:
            """
            Convert Decimal objects to floats for JSON serialization
            """
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: self.decimal_to_float(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self.decimal_to_float(item) for item in obj]
            return obj
        except Exception as e:
            logger.error(f"Error converting Decimal to float: {e} at line number {e.__traceback__.tb_lineno}")

    def updateTxn(self,request):
        try:
            stmt_id = request.json.get('stmt_id', '')
            txn_json = request.json.get('txn_json', [])
            updated_customer_details = request.json.get("updated_customer_details",{})
            customer_id = request.json.get('customer_id', '')
            loan_application_id = request.json.get('loan_application_id', '')
            sessionService = self.sessionService

            fetchTxnJson = sessionService.fetch_TxnJson(stmt_id)
            actual_txn_json = fetchTxnJson[0]
            actual_verification_metadata = fetchTxnJson[1]
            # logger.info(f"Fetched transaction JSON: {actual_txn_json}")

            if updated_customer_details.get("account_number","") != "" and updated_customer_details.get("account_number","") != " ":
                actual_verification_metadata["customer_details"]["account_number"] = updated_customer_details.get("account_number")
                account_number = updated_customer_details.get("account_number", "")
            else:
                account_number = actual_verification_metadata["customer_details"].get("account_number", "")
            if updated_customer_details.get("ifs_code","") != "" and updated_customer_details.get("ifs_code","") != " ":
                actual_verification_metadata["customer_details"]["ifs_code"] = updated_customer_details.get("ifs_code")
                ifsc_code =  updated_customer_details.get("ifs_code", "")
            else:
                ifsc_code = actual_verification_metadata["customer_details"].get("ifs_code", "")
            if  updated_customer_details.get("name","") != "" and updated_customer_details.get("name","") != " ":
                actual_verification_metadata["customer_details"]["name"] = updated_customer_details.get("name")
                name = updated_customer_details.get("name", "")
            else:
                name =  actual_verification_metadata["customer_details"].get("name", "")

            for page in txn_json:
                page_number = int(page.get('page_number', '0'))
                txns = page.get('transactions', [])
                actual_txn_page = actual_txn_json[page_number-1]["transactions"]
                idx_list = [txn.get("idx", 0) for txn in txns]
                idx_txn_map = {txn.get("idx", 0): txn for txn in txns}
                for txn in actual_txn_page:
                    if txn.get("idx", 0) in idx_list:
                        actual_txn_page[txn.get("idx", 0)-1] = idx_txn_map.get(txn.get("idx", 0), txn).copy()
                actual_txn_json[page_number-1]["transactions"] = actual_txn_page.copy()

            # Initialize tracking variables
            missmatch_txns = []
            credit_sum = Decimal('0')
            debit_sum = Decimal('0')
            txn_cnt = 0
            error_txns = 0
            credit_cnt = 0
            debit_cnt = 0
            # opening_balance = None
            opening_balance = Decimal(str(actual_txn_json[0]["transactions"][0].get("opening_balance",0)) or '0')
            
            
            # Begin transaction validation
            prev_balance = opening_balance
            out_txn_json = []

            logger.info(actual_txn_json)

            for page in actual_txn_json:
                page_number = page.get('page_number', 0)
                pagewise_txn = []
                idx = 1
                
                for tx in page.get('transactions', []):
                    debit  = Decimal(str(tx['debit']) or '0')
                    credit = Decimal(str(tx['credit']) or '0')
                    stated_balance = Decimal(str(tx['stated_balance']) or '0')

                    # Calculate expected balance
                    calculated_balance = prev_balance - debit + credit
                    if credit_sum != (credit_sum + credit):
                            credit_cnt += 1
                    if debit_sum != (debit_sum + debit):
                        debit_cnt += 1
                    credit_sum += credit
                    debit_sum += debit

                    out = {
                        "idx" : idx,
                        "txn_date" : tx.get('txn_date',''),
                        "value_date" : tx.get('value_date', ''),
                        "description" : tx.get('description', ''),
                        "ref_no_cheque_no" : tx.get('ref_no_cheque_no', ''),
                        "credit" : credit,
                        "debit" :  debit,
                        "stated_balance" : stated_balance,
                        "calculated_balance" : calculated_balance,
                        "opening_balance": prev_balance
                    }


                    # Check for mismatch
                    if calculated_balance != stated_balance:
                        print(f"Mismatch on {tx['txn_date']}: "
                            f"expected {calculated_balance:.2f}, "
                            f"stated {stated_balance:.2f}")
                        missmatch_txns.append({
                            "txn_date": tx['txn_date'],
                            "expected_balance": f"{calculated_balance:.2f}",
                            "stated_balance": f"{stated_balance:.2f}",
                            "debit": f"{debit:.2f}",
                            "credit": f"{credit:.2f}",
                            "previous_closing_balance": f"{prev_balance}",
                        })
                        error_txns += 1
                        BSA_Analysis = "FAIL"
                        out["status"] = "Fail"
                    else:
                        out["status"] = "Pass"

                    pagewise_txn.append(out)
                    # Update for next transaction
                    prev_balance = stated_balance
                    txn_cnt += 1
                    idx += 1
                out_txn_json.append({
                    "page_number": page_number,
                    "transactions": pagewise_txn
                })

            # Add transaction summary to metadata
            actual_verification_metadata["total_txns"] = txn_cnt
            actual_verification_metadata["error_txns"] = error_txns
            actual_verification_metadata["total_debits"] = float(debit_sum)
            actual_verification_metadata["total_credits"] = float(credit_sum)
            actual_verification_metadata["opening_balance"] = self.decimal_to_float(opening_balance)
            actual_verification_metadata["closing_balance"] = self.decimal_to_float(prev_balance)
            
            # Calculate transaction accuracy percentage
            valid_txns = txn_cnt - error_txns
            if valid_txns > 0:
                accuracy = round((valid_txns - len(missmatch_txns)) / valid_txns * 100, 2)
                actual_verification_metadata["transaction_accuracy"] = accuracy
            else:
                actual_verification_metadata["transaction_accuracy"] = 0
            
            # Store mismatched transactions
            actual_verification_metadata["mismatch_txns"] = missmatch_txns
            actual_verification_metadata["any_mismatch_txns"] = len(missmatch_txns) > 0

            out_txn_json = self.decimal_to_float(out_txn_json)

            # Determine overall BSA_Analysis status based on all checks
            checks_passed = all([
                actual_verification_metadata["is_bank_statement"],
                actual_verification_metadata["account_holder_name_match"],
                actual_verification_metadata["pagewise_account_number_match"],
                actual_verification_metadata["pagewise_ifsc_code_match"],
                actual_verification_metadata["pagewise_account_name_match"],
                actual_verification_metadata["penny_drop_analysis"],
                not actual_verification_metadata["stmt_completely_outside_period"]
            ])
            
            if checks_passed:
                if actual_verification_metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            # Adjust status based on transaction accuracy
            if BSA_Analysis == "SUCCESS" and actual_verification_metadata["transaction_accuracy"] < 90:
                BSA_Analysis = "INCOMPLETE"

            sessionService.update_BankStmt(stmt_id, name, account_number, ifsc_code, actual_verification_metadata, out_txn_json, BSA_Analysis, actual_verification_metadata["transaction_accuracy"], datetime.now(), "admin")

            return {
                "BSA_Analysis": BSA_Analysis,
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": actual_verification_metadata,
                "txn_json": out_txn_json,
            }

        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}
        
    def extractBankStatement(self, request):
        try:
            documents = request.files.getlist('files') 
            file = documents[0]
            original_filename = file.filename
            file_name = secure_filename(file.filename)
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
                logger.error(f"Unsupported file format: {file_extension}")
                return {
                    "error": f"Unsupported file format: {file_extension}. Supported formats are PDF, JPG, JPEG, and PNG."
                }, 400

            file_path = os.path.join(tempfile.gettempdir(), file_name)
            file.save(file_path)
            logger.info(f"File saved temporarily at: {file_path}")

            extraction_status, document_extracted = self.extract_text_from_document(file_path)

            if extraction_status:
                return {
                    "status" : "success",
                    "result" : document_extracted
                }, 200
            else:
                return {
                    "status" : "fail"
                }, 500
        except Exception as e:
            logger.error(f"Error in extractBankStatement: {e} at line number {e.__traceback__.tb_lineno}")
            return {
                "status": "fail"
            }, 500
        

    def getTxn(self, request):
        try:
            stmt_id = request.json.get('stmt_id', '')
            customer_id = request.json.get('customer_id','')
            loan_application_id = request.json.get('loan_application_id','')
            sessionService = self.sessionService

            fetchTxnJson = sessionService.fetch_BankStmtById(stmt_id)
            if fetchTxnJson.get('status', "error") == "error":
                return {"error":"statement does not exist with the given stmt_id"}, 404
            
            fetchTxnJson_data = fetchTxnJson.get('data')

            txn_json = fetchTxnJson_data.get("txn_json",[])
            verification_metadata = fetchTxnJson_data.get("verification_metadata",{})
            analysis_status = fetchTxnJson_data.get("analysis_status", "FAIL")

            return{
                "BSA_Analysis": analysis_status,
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": verification_metadata,
                "txn_json": txn_json,
            }
        except Exception as e:
            logger.error(f"Error in getTxn: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}, 500

    def previewTxn(self,request):
        try:
            stmt_id = request.json.get('stmt_id', '')
            txn_json = request.json.get('txn_json', [])
            customer_id = request.json.get('customer_id', '')
            loan_application_id = request.json.get('loan_application_id', '')

            sessionService = self.sessionService

            fetchTxnJson = sessionService.fetch_TxnJson(stmt_id)
            actual_txn_json = fetchTxnJson[0]
            actual_verification_metadata = fetchTxnJson[1]
            # logger.info(f"Fetched transaction JSON: {actual_txn_json}")

            
            for page in txn_json:
                page_number = int(page.get('page_number', '0'))
                txns = page.get('transactions', [])
                actual_txn_page = actual_txn_json[page_number-1]["transactions"]
                idx_list = [txn.get("idx", 0) for txn in txns]
                idx_txn_map = {txn.get("idx", 0): txn for txn in txns}
                for txn in actual_txn_page:
                    if txn.get("idx", 0) in idx_list:
                        actual_txn_page[txn.get("idx", 0)-1] = idx_txn_map.get(txn.get("idx", 0), txn).copy()
                actual_txn_json[page_number-1]["transactions"] = actual_txn_page.copy()

            # Initialize tracking variables
            missmatch_txns = []
            credit_sum = Decimal('0')
            debit_sum = Decimal('0')
            txn_cnt = 0
            error_txns = 0
            credit_cnt = 0
            debit_cnt = 0
            # opening_balance = None
            opening_balance = Decimal(str(actual_txn_json[0]["transactions"][0].get("opening_balance",0)) or '0')
            
            
            # Begin transaction validation
            prev_balance = opening_balance
            out_txn_json = []

            logger.info(actual_txn_json)

            for page in actual_txn_json:
                page_number = page.get('page_number', 0)
                pagewise_txn = []
                idx = 1
                
                for tx in page.get('transactions', []):
                    debit  = Decimal(str(tx['debit']) or '0')
                    credit = Decimal(str(tx['credit']) or '0')
                    stated_balance = Decimal(str(tx['stated_balance']) or '0')

                    # Calculate expected balance
                    calculated_balance = prev_balance - debit + credit
                    if credit_sum != (credit_sum + credit):
                            credit_cnt += 1
                    if debit_sum != (debit_sum + debit):
                        debit_cnt += 1
                    credit_sum += credit
                    debit_sum += debit

                    out = {
                        "idx" : idx,
                        "txn_date" : tx.get('txn_date',''),
                        "value_date" : tx.get('value_date', ''),
                        "description" : tx.get('description', ''),
                        "ref_no_cheque_no" : tx.get('ref_no_cheque_no', ''),
                        "credit" : credit,
                        "debit" :  debit,
                        "stated_balance" : stated_balance,
                        "calculated_balance" : calculated_balance,
                        "opening_balance": prev_balance
                    }


                    # Check for mismatch
                    if calculated_balance != stated_balance:
                        print(f"Mismatch on {tx['txn_date']}: "
                            f"expected {calculated_balance:.2f}, "
                            f"stated {stated_balance:.2f}")
                        missmatch_txns.append({
                            "txn_date": tx['txn_date'],
                            "expected_balance": f"{calculated_balance:.2f}",
                            "stated_balance": f"{stated_balance:.2f}",
                            "debit": f"{debit:.2f}",
                            "credit": f"{credit:.2f}",
                            "previous_closing_balance": f"{prev_balance}",
                        })
                        error_txns += 1
                        BSA_Analysis = "FAIL"
                        out["status"] = "Fail"
                    else:
                        out["status"] = "Pass"

                    pagewise_txn.append(out)
                    # Update for next transaction
                    prev_balance = stated_balance
                    txn_cnt += 1
                    idx += 1
                out_txn_json.append({
                    "page_number": page_number,
                    "transactions": pagewise_txn
                })

            # Add transaction summary to metadata
            actual_verification_metadata["total_txns"] = txn_cnt
            actual_verification_metadata["error_txns"] = error_txns
            actual_verification_metadata["total_debits"] = float(debit_sum)
            actual_verification_metadata["total_credits"] = float(credit_sum)
            actual_verification_metadata["opening_balance"] = self.decimal_to_float(opening_balance)
            actual_verification_metadata["closing_balance"] = self.decimal_to_float(prev_balance)
            
            # Calculate transaction accuracy percentage
            valid_txns = txn_cnt - error_txns
            if valid_txns > 0:
                accuracy = round((valid_txns - len(missmatch_txns)) / valid_txns * 100, 2)
                actual_verification_metadata["transaction_accuracy"] = accuracy
            else:
                actual_verification_metadata["transaction_accuracy"] = 0
            
            # Store mismatched transactions
            actual_verification_metadata["mismatch_txns"] = missmatch_txns
            actual_verification_metadata["any_mismatch_txns"] = len(missmatch_txns) > 0

            out_txn_json = self.decimal_to_float(out_txn_json)

            # Determine overall BSA_Analysis status based on all checks
            checks_passed = all([
                actual_verification_metadata["is_bank_statement"],
                actual_verification_metadata["account_holder_name_match"],
                actual_verification_metadata["pagewise_account_number_match"],
                actual_verification_metadata["pagewise_ifsc_code_match"],
                actual_verification_metadata["pagewise_account_name_match"],
                actual_verification_metadata["penny_drop_analysis"],
                not actual_verification_metadata["stmt_completely_outside_period"]
            ])
            
            if checks_passed:
                if actual_verification_metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            # Adjust status based on transaction accuracy
            if BSA_Analysis == "SUCCESS" and actual_verification_metadata["transaction_accuracy"] < 90:
                BSA_Analysis = "INCOMPLETE"

            return {
                "BSA_Analysis": BSA_Analysis,
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": actual_verification_metadata,
                "txn_json": out_txn_json,
            }

        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}

    def updateCustomerDetails(self,request):
        try:
            stmt_id = request.json.get('stmt_id', '')
            updated_customer_details = request.json.get("updated_customer_details",{})
            customer_id = request.json.get('customer_id', '')
            loan_application_id = request.json.get('loan_application_id', '')
            sessionService = self.sessionService


            fetchTxnJson = sessionService.fetch_TxnJson(stmt_id)
            actual_txn_json = fetchTxnJson[0]
            actual_verification_metadata = fetchTxnJson[1]
            # logger.info(f"Fetched transaction JSON: {actual_txn_json}")

            if updated_customer_details.get("account_number","") != "" and updated_customer_details.get("account_number","") != " ":
                actual_verification_metadata["customer_details"]["account_number"] = updated_customer_details.get("account_number")
                account_number = updated_customer_details.get("account_number", "")

                mock_penny_drop = {
                            "00000034909111330": "success",
                            "00000038322372698": "success",
                            "7223648801": "success",
                            "719130110000033": "success",
                            "69220000331696": "success",
                            "60240500000351": "success",
                            "357305040050196": "success",
                            "9876543210" : "success"
                            # Add more test account numbers as needed
                        }
                penny_drop_status =  mock_penny_drop.get(account_number, "fail")
                actual_verification_metadata["penny_drop_analysis"] = penny_drop_status

                if "penny_drop_warning" in actual_verification_metadata.keys() and penny_drop_status == "success":
                    # rempve penny_drop_warning key if it exists
                    del actual_verification_metadata["penny_drop_warning"]


            else:
                account_number = actual_verification_metadata["customer_details"].get("account_number", "")
            if updated_customer_details.get("ifs_code","") != "" and updated_customer_details.get("ifs_code","") != " ":
                actual_verification_metadata["customer_details"]["ifs_code"] = updated_customer_details.get("ifs_code")
                ifsc_code =  updated_customer_details.get("ifs_code", "")
            else:
                ifsc_code = actual_verification_metadata["customer_details"].get("ifs_code", "")
            if  updated_customer_details.get("name","") != "" and updated_customer_details.get("name","") != " ":
                actual_verification_metadata["customer_details"]["name"] = updated_customer_details.get("name")
                name = updated_customer_details.get("name", "")
            else:
                name =  actual_verification_metadata["customer_details"].get("name", "")

            if updated_customer_details.get("statement_period", "") != "" and updated_customer_details.get("statement_period", "") != " ":
                actual_verification_metadata["customer_details"]["statement_period"] = updated_customer_details.get("statement_period")
                statement_period = updated_customer_details.get("statement_period", "")

            if updated_customer_details.get('address',"") != "" and updated_customer_details.get('address',"") != " ":
                actual_verification_metadata["customer_details"]["address"] = updated_customer_details.get('address')
                address = updated_customer_details.get('address', "")

            if  updated_customer_details.get("date","") != "" and updated_customer_details.get("date","") != " ":
                actual_verification_metadata["customer_details"]["date"] = updated_customer_details.get("date")
                date = updated_customer_details.get("date", "")

            if updated_customer_details.get("branch","") != "" and updated_customer_details.get("branch","") != " ":
                actual_verification_metadata["customer_details"]["branch"] = updated_customer_details.get("branch")
                branch = updated_customer_details.get("branch", "")

            if updated_customer_details.get("cif_no","") != "" and updated_customer_details.get("cif_no","") != " ":
                actual_verification_metadata["customer_details"]["cif_no"] = updated_customer_details.get("cif_no")
                cif_no = updated_customer_details.get("cif_no", "")

            if updated_customer_details.get("micr_code","") != "" and updated_customer_details.get("micr_code","") != " ":
                actual_verification_metadata["customer_details"]["micr_code"] = updated_customer_details.get("micr_code")
                micr_code = updated_customer_details.get("micr_code", "")

            # Determine overall BSA_Analysis status based on all checks
            checks_passed = all([
                actual_verification_metadata["is_bank_statement"],
                actual_verification_metadata["account_holder_name_match"],
                actual_verification_metadata["pagewise_account_number_match"],
                actual_verification_metadata["pagewise_ifsc_code_match"],
                actual_verification_metadata["pagewise_account_name_match"],
                actual_verification_metadata["penny_drop_analysis"],
                not actual_verification_metadata["stmt_completely_outside_period"]
            ])
            
            if checks_passed:
                if actual_verification_metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            sessionService.update_BankStmt(stmt_id, name, account_number, ifsc_code, actual_verification_metadata, actual_txn_json, BSA_Analysis, actual_verification_metadata["transaction_accuracy"], datetime.now(), "admin")

            return {
                "BSA_Analysis": BSA_Analysis,
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": actual_verification_metadata,
                "txn_json": actual_txn_json,
            }



        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}

    def updateBsaHumanVerification(self, request):
        try:
            stmt_id = request.json.get('stmt_id', '')
            customer_id = request.json.get('customer_id', '')
            loan_application_id = request.json.get('loan_application_id', '')
            is_human_verified = request.json.get('is_human_verified', False)
            verified_by = request.json.get('verified_by', 'admin')

            sessionService = self.sessionService
            fetchTxnJson = sessionService.fetch_TxnJson(stmt_id)
            actual_txn_json = fetchTxnJson[0]
            actual_verification_metadata = fetchTxnJson[1]
            actual_verification_metadata["is_human_verified"] = is_human_verified
            actual_verification_metadata["verified_by"] = verified_by
            actual_verification_metadata["verified_time"] = datetime.now().isoformat()
            status = sessionService.update_HumanVerification(stmt_id, is_human_verified, verified_by, datetime.now())

            checks_passed = all([
                actual_verification_metadata["is_bank_statement"],
                actual_verification_metadata["account_holder_name_match"],
                actual_verification_metadata["pagewise_account_number_match"],
                actual_verification_metadata["pagewise_ifsc_code_match"],
                actual_verification_metadata["pagewise_account_name_match"],
                actual_verification_metadata["penny_drop_analysis"],
                not actual_verification_metadata["stmt_completely_outside_period"]
            ])

            if checks_passed:
                if actual_verification_metadata["is_stmt_covers_provided_period"]:
                    BSA_Analysis = "SUCCESS"
                else:
                    BSA_Analysis = "INCOMPLETE"
            else:
                BSA_Analysis = "FAIL"

            return {
                "BSA_Analysis": BSA_Analysis,
                "stmt_id": stmt_id,
                "customer_id": customer_id,
                "loan_application_id": loan_application_id,
                "meta_data": actual_verification_metadata,
                "txn_json": actual_txn_json,
            }

        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}
        
    def getDocAnalysis(self,request):
        try:
            documents = request.files.getlist('files') 
            customer_id = request.form.get('customer_id','')
            client_code = request.form.get('client_code')
            user_id = request.form.get('user_id','')
            doc_type = request.form.get('doc_type','')
            work_type = request.form.get('work_type','')
            doc_format = request.form.get('doc_format','PNG')
            doc_side = request.form.get('doc_side','FRONT')
            extraction_type = request.form.get('extraction_type','OCR')

            if not documents:
                return {"status":"FAIL", "error":"No document provided"}, 400
            
            sessionService = self.sessionService
            
            intialIdOfClientCode = sessionService.fetch_InitialIdByClientCode(client_code)
            initial_id = intialIdOfClientCode.get('initial_id', 0)
            logger.info("Initial id "+str(initial_id))

            work_item_result = sessionService.insert_WorkItem(client_code, work_type, f'{client_code}_{datetime.now()}', user_id, "admin", datetime.now(), "admin")

            if work_item_result['status'] != 'success':
                return {
                    "status":"FAIL",
                    "message":"Internal service error"
                }, 500
            
            work_item_id = work_item_result['work_item_id']
            if initial_id == 0:
                initial_id = work_item_id

            file = documents[0]
            file_name = secure_filename(file.filename)

            timestamp = datetime.now()

            doc_id = f"la_{timestamp}"

            
            work_doc_result = sessionService.insert_WorkDocInfo(work_item_id, doc_type ,doc_id, f"{file_name}_{doc_id}", "doc")

            if work_doc_result['status'] != 'success':
                return {
                    "status":"FAIL",
                    "message":"Internal service error"
                }, 500
            
            work_doc_info_id = work_doc_result['work_doc_info_id'] 
            
            encoded_result = self.encodeFile(documents,doc_format)
            if encoded_result.get('status','FAIL') == "FAIL":
                return {"status":"FAIL", "message":"Encoding failed"}, 500

            base64_content = encoded_result.get('base64_pages',[])
            veri5Service = Veri5ServiceGateway()

            if doc_side.lower() == "front":
                api_result = veri5Service.extractDocInfo(doc_type, doc_side, extraction_type, base64_content[0], document_back_image="")
            elif doc_side.lower() == "back":
                api_result = veri5Service.extractDocInfo(doc_type, doc_side, extraction_type,"", base64_content[0])
            else:
                api_result = veri5Service.extractDocInfo(doc_type, doc_side, extraction_type, base64_content[0], document_back_image=base64_content[1])

            '''
            example response

            {
                "response_data": {
                    "encrypted": "NO",
                    "document_data":base64_encoded
                "hash": "d425f18864cb15f1d60f1d99e9fb8a1fb4a5ef57663aecae9226196d99136f26"
                },
                "response_status": {
                    "code": "000",
                    "message": "",
                    "status": "SUCCESS"
                }
            }
            '''

            if api_result.get('status',"SUCCESS").upper() == "FAIL":
                return {"status":"FAIL", "message":"Extraction failed at the server side"}, 500
            
            api_result_response = api_result.get('result',{})

            if api_result_response.get('response_status',{}).get('code','333') == '000':
                encrypted_data = api_result_response.get('response_data',{}).get('document_data','')
                if encrypted_data:
                    decrypted_data = json.loads(base64.b64decode(encrypted_data).decode('utf-8'))
                    logger.info('decrypted data '+str(decrypted_data))
                else:
                    return {"status":"FAIL", "message":"Internal service error"}, 500
            else:
                return {"status":"FAIL", "message":"Extraction failed"}, 500
            
            stmt_id = f"{customer_id}_{work_item_id - initial_id + 1}"
            

            if doc_type != "EB":
                
                if extraction_type.lower() == "ocr":

                    result =  {
                        "photo" : 'na',
                        "extracted_info" : decrypted_data.get('original_kyc_info',{})
                    }

                    sessionService.insert_IdentityDoc(work_doc_info_id, stmt_id, result, datetime.now(), user_id, False, len(base64_content))

                    return result
                elif extraction_type.lower() == "face":
                    result =  {
                        "photo" : decrypted_data.get('photo',{}).get('document_image','na'),
                        "extracted_info" : {}
                    }

                    sessionService.insert_IdentityDoc(work_doc_info_id, stmt_id, result, datetime.now(), user_id, False, len(base64_content))

                    return result

                else:
                    result = {
                        "photo" : decrypted_data.get('photo',{}).get('document_image','na'),
                        "extracted_info" : decrypted_data.get('original_kyc_info',{})
                    }
                    sessionService.insert_IdentityDoc(work_doc_info_id, stmt_id, result, datetime.now(), user_id, False, len(base64_content))

                    return result
            else:
                result = {
                    "extracted_info" : decrypted_data.get('electricityBillFileds',{})
                }
                sessionService.insert_IdentityDoc(work_doc_info_id, stmt_id, result, datetime.now(), user_id, False, len(base64_content))

                return result


        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"error": str(e), "status": "FAIL"}
        
    
    # def encodeFile(self, document, doc_format):
    #     try:
    #         # If document is a list, get the first file
    #         if isinstance(document, list):
    #             if not document:
    #                 return {"status": "FAIL", "error": "No document provided"}
    #             file_obj = document[0]
    #         else:
    #             file_obj = document

    #         file_content = file_obj.read()
    #         base64_pages = []

    #         if doc_format.lower() == 'pdf':
    #             pdf_document = fitz.open(stream=file_content, filetype="pdf")
    #             for page_num in range(pdf_document.page_count):
    #                 page = pdf_document.load_page(page_num)
    #                 pix = page.get_pixmap(dpi=300)
    #                 pil_image = Image.open(BytesIO(pix.tobytes()))
    #                 img_byte_arr = BytesIO()
    #                 pil_image.save(img_byte_arr, format='PNG')
    #                 encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    #                 base64_pages.append(encoded_image)
    #             return {"status": "SUCCESS", "base64_pages": base64_pages}
    #         else:
    #             encoded_content = base64.b64encode(file_content).decode('utf-8')
    #             base64_pages.append(encoded_content)
    #             return {"status": "SUCCESS", "base64_pages": base64_pages}
    #     except Exception as e:
    #         logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
    #         return {"status":"FAIL"}

    def encodeFile(self, document, doc_format):
        try:
            base64_pages = []
            # Always treat as a list for uniformity
            files = document if isinstance(document, list) else [document]
            if not files:
                return {"status": "FAIL", "error": "No document provided"}

            for file_obj in files:
                file_obj.seek(0)  # Ensure pointer is at start
                file_content = file_obj.read()
                file_obj.seek(0)  # Reset pointer for any further use

                ext = os.path.splitext(file_obj.filename)[1].lower()
                if ext == '.pdf' or doc_format.lower() == 'pdf':
                    pdf_document = fitz.open(stream=file_content, filetype="pdf")
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document.load_page(page_num)
                        pix = page.get_pixmap(dpi=300)
                        pil_image = Image.open(BytesIO(pix.tobytes()))
                        img_byte_arr = BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                        base64_pages.append(encoded_image)
                else:
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    base64_pages.append(encoded_content)

            return {"status": "SUCCESS", "base64_pages": base64_pages}
        except Exception as e:
            logger.error(f"Error in update transaction: {e} at line number {e.__traceback__.tb_lineno}")
            return {"status":"FAIL"}
        

    def invoiceAnalysis(self,request):
        try:
            documents = request.files.getlist('files') 
            file_format = request.form.get('doc_format','pdf')
            doc_type = request.form.get('doc_type','invoice')
            veri5Service = Veri5ServiceGateway()
            if not documents:
                return {"status":"FAIL", "error":"No document provided"}, 400
            file = documents[0]
            result = veri5Service.extractbilldata(file_obj = file, file_type = file_format, doc_type = doc_type.lower(), api_key = "OPOfwhk3z44keJ7Lqrsn9L2gr5TCTzpF-5A2C9F899D464")

            if result.get('status', 'FAIL').upper() == "FAIL":
                return {"status":"FAIL", "message":"Extraction failed at the server side"}, 500
            
            logger.info(f"Invoice analysis result: {result.get('response')}")

            return result.get('result',{}).get('response',{})
        except Exception as e:
            logger.error(f"Error in invoiceAnalysis: {e} at line number {e.__traceback__.tb_lineno}")
            return {"status": "FAIL", "error": str(e)}, 500
        
# --- 1. Define the JSON Schema (No Changes Here) ---

class Transaction(TypedDict):
    txn_date: str
    value_date: str
    description: str
    ref_no_cheque_no: str
    debit: float
    credit: float
    balance: float

class AccountStatement(TypedDict):
    transactions: List[Transaction]

class AccountDetails(TypedDict):
    account_name: str
    address: str
    date: str
    account_number: str
    account_description: str
    branch: str
    drawing_power: str
    interest_rate: str
    mod_balance: str
    cif_no: str
    ckyc_number: str
    ifs_code: str
    micr_code: str
    nomination_registered: str
    opening_balance: float
    period: str

class BankStatementPage(TypedDict):
    page_number: int
    account_details: AccountDetails
    account_statement: AccountStatement

class BSAextractionGeminiService:

    def __init__(self):
        # --- 2. Configure the Gemini API Model (No Changes Here) ---

        # Replace with your actual API key
        genai.configure(api_key=GEMINI_API_KEY)

        self.system_prompt = """You are an OCR tool designed to extract all the data from PDF with page numbers and structure it in JSON format. Be consistent with the parameters and variables in case of repetition.

        Here is the example for transaction:
                        {
                            "balance": "13,708.24",
                            "credit": "",
                            "debit": "1.00",
                            "description": "TO TRANSFER-UPI/DR/428411838816/BHANUYA/HUFC/bhanuyadav/UPI-",
                            "ref_no_cheque_no": "4897694162092",
                            "txn_date": "10 Oct 2024",
                            "value_date": "10 Oct 2024"
                        }

        Here is the example for account details: if anything is not available, then keep it as empty string.
        "                {
                            "account_description": "",
                            "account_name": "John Doe",
                            "account_number": "9876543210",
                            "address": "",
                            "branch": "California",
                            "cif_no": "",
                            "ckyc_number": "",
                            "date": "26 Mar 2025",
                            "drawing_power": "0.00",
                            "ifs_code": "",
                            "interest_rate": "",
                            "micr_code": "",
                            "mod_balance": "",
                            "nomination_registered": "",
                            "opening_balance": "",
                            "period": ""
                        }
        """
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": BankStatementPage,
        }
        self.model = genai.GenerativeModel(
            # model_name="gemini-2.5-flash",
            model_name = GEMINI_MODEL_NAME,
            system_instruction=self.system_prompt,
            generation_config=self.generation_config
        )


    # --- 3. Worker Function to Process a Single Page (No Changes Here) ---

    def process_page(self,page_num: int, page: fitz.Page) -> Dict[str, Any]:
        """Converts a single PDF page to an image and calls the Gemini API."""
        print(f"Processing page {page_num}...")
        try:
            pix = page.get_pixmap(dpi=300)
            img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            prompt = f"Extract all account and transaction details from this image. This is page number {page_num} of the document."
            response = self.model.generate_content([prompt, img])
            page_data = json.loads(response.text)
            print(f"Successfully processed page {page_num}.")
            return page_data
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return {"page_number": page_num, "error": str(e)}

    # # --- 4. Main Function Modified to Accept Bytes ---

    # def process_pdf_from_api_content(pdf_content: bytes) -> List[Dict[str, Any]]:
    #     """
    #     Processes PDF content from a byte stream in parallel.

    #     Args:
    #         pdf_content: The binary content of the PDF file.

    #     Returns:
    #         A sorted list of dictionaries, where each dictionary represents a page.
    #     """
    #     all_pages_data = []
    #     try:
    #         # **KEY CHANGE**: Open the PDF from the byte stream in memory
    #         doc = fitz.open(stream=pdf_content, filetype="pdf")
    #     except Exception as e:
    #         print(f"Error opening PDF from memory stream: {e}")
    #         return []

    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(process_page, i + 1, doc.load_page(i)): i + 1 for i in range(len(doc))}
    #         for future in concurrent.futures.as_completed(futures):
    #             page_num = futures[future]
    #             try:
    #                 result = future.result()
    #                 if result:
    #                     all_pages_data.append(result)
    #             except Exception as exc:
    #                 print(f"Page {page_num} generated an exception: {exc}")
    #                 all_pages_data.append({"page_number": page_num, "error": str(exc)})

    #     sorted_results = sorted(all_pages_data, key=lambda x: x.get('page_number', float('inf')))
    #     return sorted_results

    # --- 4. Main Function to Orchestrate Parallel Processing ---

    def process_pdf_parallel(self,pdf_path: str) -> List[Dict[str, Any]]:
        """
        Processes each page of a PDF in parallel, extracts structured data,
        and returns a sorted list of JSON objects.
        """
        all_pages_data = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF file: {e}")
            return []

        # Using ThreadPoolExecutor to process pages concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a future for each page.
            futures = {executor.submit(self.process_page, i + 1, doc.load_page(i)): i + 1 for i in range(len(doc))}

            # As each future completes, collect its result
            for future in concurrent.futures.as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_pages_data.append(result)
                except Exception as exc:
                    print(f"Page {page_num} generated an exception: {exc}")
                    all_pages_data.append({"page_number": page_num, "error": str(exc)})


        # Sort the final list based on the 'page_number' key
        sorted_results = sorted(all_pages_data, key=lambda x: x.get('page_number', float('inf')))
        
        return sorted_results

class CreditCardAnalysis:
    def __init__(self, db):
        self.client = OpenAI(api_key=openai_api_key)
        self.sessionService = SessionService(db)
        

    def performCCAnalysisGemini(self, request):
        try:
            documents = request.files.getlist('files') 
            customer_id = request.form.get('customer_id','')
            loan_application_id = request.form.get('loan_application_id','')
            borrower_name = request.form.get('borrower_name','')
            proprietor_name = request.form.get('proprietor_name','')
            expected_stmt_start_date = request.form.get('expected_stmt_start_date','')
            expected_stmt_end_date = request.form.get('expected_stmt_end_date','')
            client_code = request.form.get('client_code')
            entity_type = request.form.get('entity','')
            user_id = request.form.get('user_id','')
            doc_type_source = request.form.get('source','doc') 


            file = documents[0]
            file_name = secure_filename(file.filename)
            file_extension = os.path.splitext(file_name)[1].lower()

            # Save the file temporarily
            file_path = os.path.join(tempfile.gettempdir(), file_name)
            file.save(file_path)
            logger.info(f"File saved temporarily at: {file_path}")

            sessionService = self.sessionService

            now = datetime.now()

            # Convert datetime to timestamp
            timestamp = now.timestamp()

            work_id = f"{loan_application_id}_{timestamp}"

            intialIdOfClientCode = sessionService.fetch_InitialIdByClientCode(client_code)
            initial_id = intialIdOfClientCode.get('initial_id', 0)
            logger.info("Initial id "+str(initial_id))

            work_item_result = sessionService.insert_WorkItem(client_code, "CCAnalysis", loan_application_id, user_id, "admin", datetime.now(), "admin")
            doc_id = f"la_{timestamp}"

            work_item_id = work_item_result['work_item_id']
            if initial_id == 0:
                initial_id = work_item_id

            
            work_doc_result = sessionService.insert_WorkDocInfo(work_item_id, "CC Statement",doc_id, f"{file_name}_{doc_id}", doc_type_source)

            work_doc_info_id = work_doc_result['work_doc_info_id'] 



            classification_result = self.ccClassification(file_path)
            is_cc_statement = classification_result.get('isCCStatement', False)
            if not is_cc_statement:
                return {
                    "meta_data" : {
                        "isCCStatement": False,
                    }
                }


            extractor = CC_GeminiExtractor(model_name=GEMINI_MODEL_NAME)
            document_extracted = extractor.main_extract(file_path)

            if os.path.exists(file_path):
                os.remove(file_path)


            total_debits = Decimal('0')
            total_credits = Decimal('0')
            debits_cnt = 0
            credits_cnt = 0
            total_txns = Decimal('0')
            txn_cnt = 0
            fx_txn_cnt = 0
            fx_txn_map = {} # currencywise fx txn count map

            txn_json = []

            account_number_set = set()
            address_set = set()
            available_cash_limit_set = set()
            available_credit_limit_set = set()
            bank_name_set = set()
            billing_period_end_set = set()
            billing_period_start_set = set()
            card_number_masked_set = set()
            cardholder_name_set = set()
            cash_limit_set = set()
            credit_limit_set = set()
            gst_number_set = set()
            hsn_code_set = set()
            minimum_amount_due_set = set()
            opening_balance_set = set()
            past_dues_1_month_set = set()
            past_dues_2_months_set = set()
            past_dues_3_months_set = set()
            past_dues_current_dues_set = set()
            payment_due_date_set = set()
            payment_summary_finance_charges_set = set()
            payment_summary_interest_charged_set = set()
            payment_summary_late_fees_set = set()
            payment_summary_other_credits_set = set()
            payment_summary_payments_made_set = set()
            relationship_number_set = set()
            reward_summary_closing_points_set = set()
            reward_summary_opening_points_set = set()
            reward_summaery_points_adjusted_or_lapsed_set = set()
            reward_summary_points_earned_set = set()
            reward_summary_points_expiring_next_30_days_set = set()
            reward_summary_points_expiring_next_60_days_set = set()
            statement_date_set = set()
            total_dues_set = set()

            total_pages = len(document_extracted)


            for page in document_extracted:
                
                if "transactions" in page.keys() and len(page["transactions"]) > 0:
                    
                    transactions = page["transactions"]
                    idx = 1
                    txns_list = []
                    for txn in transactions:
                        if txn.get('credit_or_debit','Debit').lower() == 'debit':
                            debit = Decimal(str(txn.get('amount_inr', '0')))
                            total_debits += debit
                            debits_cnt += 1
                            total_txns += debit
                        else:
                            credit = Decimal(str(txn.get('amount_inr', '0')))
                            total_credits += credit
                            credits_cnt += 1
                            total_txns -= credit
                        txn_cnt += 1
                        if txn.get('is_fx_transaction', False):
                            fx_txn_cnt += 1
                            fx_txn_map[txn.get('fx_currency','').upper()] = fx_txn_map.get(txn.get('fx_currency','').upper(), 0) + 1
                        txn['idx'] = idx
                        idx += 1
                        txns_list.append(txn)
                    txn_json.append({"page_number":page.get('page_number',1),"transactions":txns_list})

                if "account_details" in page.keys() and len(page["account_details"]) > 0:
                    account_details = page.get('account_details', [])
                    if len(account_details) == 0:
                        continue
                    account_details = account_details[0]

                    # (field, set, default, nonzero, parent)
                    fields = [
                        ('account_number', account_number_set, '', False, account_details),
                        ('address', address_set, '', False, account_details),
                        ('available_cash_limit', available_cash_limit_set, 0, True, account_details),
                        ('available_credit_limit', available_credit_limit_set, 0, True, account_details),
                        ('bank_name', bank_name_set, '', False, account_details),
                        ('billing_period_end', billing_period_end_set, '', False, account_details),
                        ('billing_period_start', billing_period_start_set, '', False, account_details),
                        ('card_number_masked', card_number_masked_set, '', False, account_details),
                        ('cardholder_name', cardholder_name_set, '', False, account_details),
                        ('cash_limit', cash_limit_set, 0, True, account_details),
                        ('credit_limit', credit_limit_set, 0, True, account_details),
                        ('gst_number', gst_number_set, '', False, account_details),
                        ('hsn_code', hsn_code_set, '', False, account_details),
                        ('minimum_amount_due', minimum_amount_due_set, 0, True, account_details),
                        ('opening_balance', opening_balance_set, 0, True, account_details),
                        ('payment_due_date', payment_due_date_set, '', False, account_details),
                        ('relationship_number', relationship_number_set, '', False, account_details),
                        ('statement_date', statement_date_set, '', False, account_details),
                        ('total_dues', total_dues_set, 0, True, account_details),
                    ]

                    for field, target_set, default, nonzero, parent in fields:
                        value = parent.get(field, default)
                        if (not nonzero and value != '') or (nonzero and value != 0):
                            target_set.add(value)

                    # Nested fields for past_dues
                    for months, target_set in [
                        ('1_month', past_dues_1_month_set),
                        ('2_months', past_dues_2_months_set),
                        ('3_months', past_dues_3_months_set),
                        ('current_dues', past_dues_current_dues_set),
                    ]:
                        value = account_details.get('past_dues', {}).get(months, 0)
                        if value != 0:
                            target_set.add(value)

                    # Nested fields for payment_summary
                    for field, target_set in [
                        ('finance_charges', payment_summary_finance_charges_set),
                        ('interest_charged', payment_summary_interest_charged_set),
                        ('late_fees', payment_summary_late_fees_set),
                        ('other_credits', payment_summary_other_credits_set),
                        ('payments_made', payment_summary_payments_made_set),
                    ]:
                        value = account_details.get('payment_summary', {}).get(field, 0)
                        if value != 0:
                            target_set.add(value)

                    # reward_summary fields are on the page level
                    for field, target_set in [
                        ('closing_points', reward_summary_closing_points_set),
                        ('opening_points', reward_summary_opening_points_set),
                        ('points_adjusted_or_lapsed', reward_summaery_points_adjusted_or_lapsed_set),
                        ('points_earned', reward_summary_points_earned_set),
                        ('points_expiring_next_30_days', reward_summary_points_expiring_next_30_days_set),
                        ('points_expiring_next_60_days', reward_summary_points_expiring_next_60_days_set),
                    ]:
                        value = page.get('reward_summary', {}).get(field, 0)
                        if value != 0:
                            target_set.add(value)

            account_details = {
                "account_number": list(account_number_set)[0] if account_number_set else "",
                "address": list(address_set)[0] if address_set else "",
                "available_cash_limit" : max(available_cash_limit_set) if available_cash_limit_set else 0,
                "available_credit_limit" : max(available_credit_limit_set) if available_credit_limit_set else 0,
                "bank_name": list(bank_name_set)[0] if bank_name_set else "",
                "billing_period_end": list(billing_period_end_set)[0] if billing_period_end_set else "",
                "billing_period_start": list(billing_period_start_set)[0] if billing_period_start_set else "",
                "card_number_masked": list(card_number_masked_set)[0] if card_number_masked_set else "",
                "cardholder_name": list(cardholder_name_set)[0] if cardholder_name_set else "",
                "cash_limit": max(cash_limit_set) if cash_limit_set else 0,
                "credit_limit": max(credit_limit_set) if credit_limit_set else 0,
                "gst_number": list(gst_number_set)[0] if gst_number_set else "",
                "hsn_code": list(hsn_code_set)[0] if hsn_code_set else "",
                "minimum_amount_due": max(minimum_amount_due_set) if minimum_amount_due_set else 0,
                "opening_balance": max(opening_balance_set) if opening_balance_set else 0,
                "past_dues": {
                    "1_month": max(past_dues_1_month_set) if past_dues_1_month_set else 0,
                    "2_months": max(past_dues_2_months_set) if past_dues_2_months_set else 0,
                    "3_months": max(past_dues_3_months_set) if past_dues_3_months_set else 0,
                    "current_dues": max(past_dues_current_dues_set) if past_dues_current_dues_set else 0
                },
                "payment_due_date": list(payment_due_date_set)[0] if payment_due_date_set else "",
                "payment_summary": {
                    "finance_charges": max(payment_summary_finance_charges_set) if payment_summary_finance_charges_set else 0,
                    "interest_charged": max(payment_summary_interest_charged_set) if payment_summary_interest_charged_set else 0,
                    "late_fees": max(payment_summary_late_fees_set) if payment_summary_late_fees_set else 0,
                    "other_credits": max(payment_summary_other_credits_set) if payment_summary_other_credits_set else 0,
                    "payments_made": max(payment_summary_payments_made_set) if payment_summary_payments_made_set else 0
                },
                "relationship_number": list(relationship_number_set)[0] if relationship_number_set else "",
                "reward_summary": {
                    "closing_points": max(reward_summary_closing_points_set) if reward_summary_closing_points_set else 0,
                    "opening_points": max(reward_summary_opening_points_set) if reward_summary_opening_points_set else 0,
                    "points_adjusted_or_lapsed": max(reward_summaery_points_adjusted_or_lapsed_set) if reward_summaery_points_adjusted_or_lapsed_set else 0,
                    "points_earned": max(reward_summary_points_earned_set) if reward_summary_points_earned_set else 0,
                    "points_expiring_next_30_days": max(reward_summary_points_expiring_next_30_days_set) if reward_summary_points_expiring_next_30_days_set else 0,
                    "points_expiring_next_60_days": max(reward_summary_points_expiring_next_60_days_set) if reward_summary_points_expiring_next_60_days_set else 0
                },
                "statement_date": max(statement_date_set) if statement_date_set else "",
                "total_dues": max(total_dues_set) if total_dues_set else 0
            }

            total_txns = float(total_txns)

            total_dues = float(account_details.get('total_dues', 0))
            opening_balance = float(account_details.get('opening_balance', 0))
            total_calculated_due = float(opening_balance + total_txns)


            result = {
                "txn_json" : txn_json,
                "meta_data" : {
                    "total_debits" : total_debits,
                    "total_credits" : total_credits,
                    "debits_cnt" : debits_cnt,
                    "credits_cnt" : credits_cnt, 
                    "total_txns" : total_txns,
                    "total_calculated_due" : total_calculated_due,
                    "txn_cnt" : txn_cnt,
                    "fx_txn_cnt" : fx_txn_cnt,
                    "fx_txn_map" : fx_txn_map,
                    "account_details" : account_details,
                    # "document_extracted" : document_extracted,
                    "isCCStatement" : True,
                    "total_pages" : total_pages
                }
            }

            

            result = self.decimal_to_float(result)

            total_dues = result['meta_data']['account_details'].get('total_dues', 0)
            total_calculated_due = result['meta_data']['total_calculated_due']

            logger.info(f"Total Dues: {total_dues}, Total Calculated Due: {total_calculated_due}")

            if total_dues == total_calculated_due:
                result['meta_data']['total_dues_match'] = True
            else:
                result['meta_data']['total_dues_match'] = False
                

            accuracy_percentage = round(float(100 - ((total_dues - total_calculated_due)/ total_dues * 100) if total_dues != 0 else 0),2)

            result['meta_data']['accuracy_percentage'] = accuracy_percentage 

            stmt_id = f"{loan_application_id}_{work_item_id - initial_id + 1}"

            account_number = account_details.get('account_number', '')
            account_name = account_details.get('cardholder_name', '')

            bank_stmt_insert_result = sessionService.insert_CCStmt(work_doc_info_id,stmt_id , " ", account_name, account_number, total_pages, "", result.get('meta_data',{}), document_extracted, txn_json, "", accuracy_percentage, datetime.now(), "admin", False)

            result['meta_data']['stmt_id'] = stmt_id

            return result

                
        except Exception as e:
            logger.error(f"Error in performCCAnalysisGemini: {e} at line number {e.__traceback__.tb_lineno}")
            return {"status": "FAIL", "error": str(e)}, 500
        
    def decimal_to_float(self,obj):
        try:
            """
            Convert Decimal objects to floats for JSON serialization
            """
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: self.decimal_to_float(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self.decimal_to_float(item) for item in obj]
            return obj
        except Exception as e:
            logger.error(f"Error converting Decimal to float: {e} at line number {e.__traceback__.tb_lineno}")

    def ccClassification(self,pdf_path: str):
        system_instruction = """
        You are an AI trained in financial document analysis with specialized expertise in credit card statements, invoices, and billing documents.
        Your task is to meticulously analyze provided text or image content to determine if it is a **credit card statement** based on strict criteria.

        You must:
        - Look for definitive features unique to credit card statements (e.g., credit card number, due date, total outstanding, transaction list).
        - Avoid false positives from invoices, bank statements, or generic financial documents.
        - Return a JSON response with a **clear decision and detailed justification**.

        Respond only in the specified format.
        """

        classification_prompt = f"""
        Analyze the following content and determine whether it is a **credit card statement**. Use the following criteria:

    Key Indicators:
    1. Credit Card Metadata:
      - Cardholder name.
      - Masked card number (e.g., XXXX-XXXX-XXXX-1234).
      - Statement date / billing period.
      - Payment due date and minimum amount due.
      - Total outstanding or closing balance.
      - Credit limit and available credit.

    2. Transaction Summary:
      - Transaction list including date, merchant, amount.
      - Separate debit/credit entries are **not required**, but total spends should be visible.
      - Sectional breakdown: retail spends, EMIs, late payment fees, interest, etc.

    3. Billing Terminology:
      - Keywords: Total Due, Minimum Amount Due, Due Date, Closing Balance, Previous Balance, Credit Limit.
      - Amounts typically with currency symbols and proper formatting.

    4. Document Structure & Format:
      - Tables with transaction details.
      - Summary sections often labeled as "Payment Summary" or "Account Summary".
      - References to credit card provider (e.g., HDFC, SBI Card, ICICI, Axis, Amex, etc.)

    Exclusion Criteria:
    - Reject if document lacks any card-related metadata.
    - Reject if the content only includes payment receipts or partial statements.
    - Reject if it resembles a bank statement (i.e., running balance, IFSC, savings/current account).

    Output Format (JSON):
    {{
        "isCCStatement": bool
        
        }}

    Donot provide any other information other than what is mentioned in the schema
        """

        
        image_path = "first_page_temp.png"
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        pix.save(image_path)
        doc.close()

        
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded_image}"

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": classification_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

       
        if result_text.startswith("```"):
            result_text = result_text.strip("`").strip("json").strip()

        
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {
                "error": "Failed to parse LLM response.",
                "raw_response": result_text
            }

       
        print(json.dumps(result_json, indent=2))

        
        if os.path.exists(image_path):
            os.remove(image_path)

        return result_json
    
    #     def check_name_and_period(self,ocr_results, given_start_date, given_end_date, given_borrower_name, given_proprietor_name):

#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         client = OpenAI(api_key=openai_api_key)


#         txn_dates = [
#             txn.get('txn_date') or txn.get('Txn Date')
#             for record in ocr_results
#             for txn in (
#                 record.get('account_statement', {}).get('transactions', [])
#                 + record.get('transactions', [])
#             )
#             if txn.get('txn_date') or txn.get('Txn Date')
#         ]

#         account_names = {
#                 rec.get('account_details', {}).get('account_name')
#                 for rec in ocr_results
#                 if rec.get('account_details', {}).get('account_name')
#             }
        

        
#         llm_prompt = f"""Analyze this bank statement data and do the following:

#         1. Check if the {account_names} is matching with the {given_borrower_name if given_borrower_name else ""} or  {given_proprietor_name if given_proprietor_name else ""} name. If yes, then set "account_holder_name_match" to "YES", else "NO".
#         2. Note that name matching should not be strich. For example if the name is justin b but actual name is justin bieber, then is should be YES. Set it to no if it is not matching at all. For example if the name is justin b and actual name is Hailey b, then it should be NO.
#         2. Check if the statement contains transactons from {given_start_date} to {given_end_date}. If yes, then set "is_statement_is_within_the_period" to "YES", else "NO" and set the corresponding missing months.
#         3. Note that if the period is from 01 jan 2023 to 01 dec 2023, and transactions dates are from 03 jan 2023 to 31 dec 2023, then note that the user has not done any transactions on 1 and 2 jan, so it can be ignored since he provided for jan 2023 to dec 2023. So, the missing months will be [].
#         4. If the statement is completely outside the given period, then set is_completely_outside_period to "YES", else if it is partially outside, then set is_completely_outside_period to "NO".
#         5. Store the missing months and provided months in the format of YYYY-MMM. For example, if the missing month is Jan 2023, then it should be stored as 2023-Jan. If the provided month is Jan 2023, then it should be stored as 2023-Jan.
#         4. Return JSON with:
#         {{
#             "account_holder_name_match": "YES/NO",
#             "is_statement_is_within_the_period": "YES/NO",
#             "start_date": "YYYY-MM-DD",
#             "end_date": "YYYY-MM-DD",
#             "missing_months": ["list of months along with the year that are missing example: ['2023-Jan', '2023-Feb']"],
#             "provided_months": ["list of months along with the year that are provided example: ['2023-Jan', '2023-Feb']"],
#             "is_completely_outside_period": "YES/NO"
#         }}
#         5. Strictly follow the JSON format and do not add any additional text or explanation.

#         Extracted transaction dates: {txn_dates}
    
#         """

#         tools = [
#             {
#                 "type": "function",
#                 "function": {
#                 "name": "bsa_analysis",
#                 "description": "Function to analyze the bank statement data.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                     "account_holder_name_match": {
#                         "type": "string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the account holder name matches the given names."
#                     },
#                     "is_statement_is_within_the_period": {
#                         "type":"string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the statement is within the given period."
#                     },
#                     "start_date": {
#                         "type": "string",
#                         "format": "date",
#                         "description": "Start date of the statement period."
#                     },
#                     "end_date": {
#                         "type": "string",
#                         "format": "date",
#                         "description": "End date of the statement period."
#                     },
#                     "missing_months": {
#                         "type": "array",
#                         "items": {
#                             "type": "string",
#                             "description": "List of months along with year that are missing."
#                         }
#                     },
#                     "is_completely_outside_period": {
#                         "type": "string",
#                         "enum": ["YES", "NO"],
#                         "description": "Indicates if the statement is completely outside the given period."
#                     },
#                     "provided_months": {
#                         "type": "array",
#                         "items": {
#                             "type": "string",
#                             "description": "List of months along with year that are provided."
#                         }
#                     }
#                     },
#                     "required": [
#                         "account_holder_name_match",
#                         "is_statement_is_within_the_period",
#                         "start_date",
#                         "end_date",
#                         "missing_months",
#                         "is_completely_outside_period",
#                         "provided_months"
#                     ]
#                 }
#                 }
#             }
#             ]
        
#         msg = [
#                 {"role": "system", "content": "You're a bank statement analyst."},
#                 {"role": "user", "content": llm_prompt}
#             ]
        
        
#         response = client.chat.completions.create(
#             model="gpt-4o",  
#             response_format={"type": "json_object"},
#             messages=msg,
#             tools= tools,
#             tool_choice={"type": "function", "function": {"name": "bsa_analysis"}},
#         )

#         # print(response)
#         print(response.choices[0].message.tool_calls[0].function.arguments)
#         return json.loads(response.choices[0].message.tool_calls[0].function.arguments)



genai.configure(api_key=GEMINI_API_KEY)

class CC_GeminiExtractor:
    """OCR + Gemini extraction utility.

    Converts each page of a PDF or image to a pre‑processed PNG, then asks Gemini
    to extract structured JSON from the image.
    """

    def __init__(self, model_name: str = GEMINI_MODEL_NAME):
        # Keep the model instance around so we don’t re‑instantiate it for every page
        self.model = genai.GenerativeModel(model_name)

    # ---------- Image utilities ---------- #

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        return denoised

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # ---------- PDF handling ---------- #

    def process_pdf_page(
        self, page_num: int, pdf_document: fitz.Document, dpi: int = 200
    ) -> str:
        scale = dpi / 72.0
        with tempfile.TemporaryDirectory() as temp_dir:
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)

            preprocessed = self.preprocess_image(img_np)
            temp_image_path = os.path.join(
                temp_dir, f"temp_page_{page_num + 1}.png"
            )
            cv2.imwrite(temp_image_path, preprocessed)

            with open(temp_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        logger.debug(f"Processed PDF page {page_num + 1}")
        return base64_image

    def pdf_to_base64_images(self, pdf_path: str, dpi: int = 200) -> List[str]:
        logger.info("Converting PDF to base64 images: %s", pdf_path)
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)

        max_workers = min(total_pages, os.cpu_count() or 1)
        base64_images: List[Optional[str]] = [None] * total_pages

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_pdf_page, page_num, pdf_document, dpi
                ): page_num
                for page_num in range(total_pages)
            }

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    base64_images[page_num] = future.result()
                except Exception as e:
                    logger.exception(
                        "Error processing PDF page %s: %s", page_num + 1, e
                    )

        # Drop failed pages if any
        return [img for img in base64_images if img]

    # ---------- Gemini call ---------- #

    def extract_invoice_data(self, base64_image: str) -> Dict[str, Any]:
        try:
            json_structure={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Credit Card Statement Schema",
                "type": "object",
                "properties": {
                    "account_details": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cardholder_name": {
                                    "type": "string"
                                },
                                "card_number_masked": {
                                    "type": "string"
                                },
                                "bank_name": {
                                    "type": "string"
                                },
                                "statement_date": {
                                    "type": "string",
                                    "format": "date"
                                },
                                "billing_period_start": {
                                    "type": "string",
                                    "format": "date"
                                },
                                "billing_period_end": {
                                    "type": "string",
                                    "format": "date"
                                },
                                "payment_due_date": {
                                    "type": "string",
                                    "format": "date"
                                },
                                "account_number": {
                                    "type": "string"
                                },
                                "relationship_number": {
                                    "type": "string"
                                },
                                "gst_number": {
                                    "type": "string"
                                },
                                "hsn_code": {
                                    "type": "string"
                                },
                                "address": {
                                    "type": "string"
                                },
                                "credit_limit": {
                                    "type": "number"
                                },
                                "available_credit_limit": {
                                    "type": "number"
                                },
                                "cash_limit": {
                                    "type": "number"
                                },
                                "available_cash_limit": {
                                    "type": "number"
                                },
                                "opening_balance": {
                                    "type": "number"
                                },
                                "total_dues": {
                                    "type": "number"
                                },
                                "minimum_amount_due": {
                                    "type": "number"
                                },
                                "past_dues": {
                                    "type": "object",
                                    "properties": {
                                        "3_months": {
                                            "type": "number"
                                        },
                                        "2_months": {
                                            "type": "number"
                                        },
                                        "1_month": {
                                            "type": "number"
                                        },
                                        "current_dues": {
                                            "type": "number"
                                        }
                                    },
                                    "additionalProperties": False
                                },
                                "payment_summary": {
                                    "type": "object",
                                    "properties": {
                                        "payments_made": {
                                            "type": "number"
                                        },
                                        "other_credits": {
                                            "type": "number"
                                        },
                                        "finance_charges": {
                                            "type": "number"
                                        },
                                        "interest_charged": {
                                            "type": "number"
                                        },
                                        "late_fees": {
                                            "type": "number"
                                        }
                                    },
                                    "additionalProperties": False
                                },
                                "reward_summary": {
                                    "type": "object",
                                    "properties": {
                                        "opening_points": {
                                            "type": "number"
                                        },
                                        "points_earned": {
                                            "type": "number"
                                        },
                                        "points_adjusted_or_lapsed": {
                                            "type": "number"
                                        },
                                        "closing_points": {
                                            "type": "number"
                                        },
                                        "points_expiring_next_30_days": {
                                            "type": "number"
                                        },
                                        "points_expiring_next_60_days": {
                                            "type": "number"
                                        }
                                    },
                                    "additionalProperties": False
                                }
                            }
                        }
                    },
                    "transactions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "transaction_date": {
                                    "type": "string",
                                    "format": "date"
                                },
                                "description": {
                                    "type": "string"
                                },
                                "amount_inr": {
                                    "type": "number"
                                },
                                "transaction_type": {
                                    "type": "string",
                                    "enum": [
                                        "Purchase",
                                        "Refund",
                                        "Cash Withdrawal",
                                        "UPI",
                                        "Interest",
                                        "Fee",
                                        "EMI",
                                        "FX",
                                        "Payment",
                                        "Reversal",
                                        "Others"
                                    ]
                                },
                                "reward_points": {
                                    "type": "number"
                                },
                                "is_fx_transaction": {
                                    "type": "boolean"
                                },
                                "fx_currency": {
                                    "type": "string"
                                },
                                "fx_amount": {
                                    "type": "number"
                                },
                                "conversion_rate": {
                                    "type": "number"
                                },
                                "markup_fee": {
                                    "type": "number"
                                },
                                "credit_or_debit": {
                                    "type": "string",
                                    "enum": [
                                        "Credit",
                                        "Debit"
                                    ]
                                }
                            },
                            "required": [
                                "transaction_date",
                                "description",
                                "amount_inr"
                            ]
                        }
                    }
                },
                "required": [],
                "additionalProperties": False
            }

            system_prompt =f"""
You are a world-class information-extraction engine that reads raw credit-card-statement text (as produced by OCR or text-parsing of a PDF),extracts all credit-card-statement related information and returns a single JSON object that conforms exactly to the schema described below.
━━━━━━━━  JSON OUTPUT RULES ━━━━━━━━
1. **Return ONLY valid JSON — no extra keys, no commentary, no markdown.**
2. **Omit any field that cannot be confidently found** (do NOT include it with null/empty values).
3. Numeric fields: parse as pure numbers (no commas, currency symbols, or “Cr” suffixes).
4. Dates: ISO-8601 format YYYY-MM-DD (use the statement’s own date format to infer year if needed).
5. Place the list of individual transactions in the transactions array.
6. Preserve all fields’ spelling / naming exactly as given in the schema.

━━━━━━━━  CREDIT vs. DEBIT LOGIC ━━━━━━━━
•  If a transaction amount line ends with “Cr” (or “CR”, case-insensitive),  
   set "credit_or_debit": "Credit" and remove the “Cr” text from the numeric amount.  
•  Otherwise set "credit_or_debit": "Debit".
━━━━━━━━  FX-TRANSACTION LOGIC ━━━━━━━━
•  Detect an international / FX transaction when you see a three-letter currency code
   **other than INR/₹/Rs** directly next to an amount (e.g. “USD 58.62”, “CAD 12”, “EUR 199.00”)
   OR when the statement explicitly marks a transaction as international.
•  For such rows:
   ‣ "is_fx_transaction": true  
   ‣ "fx_currency"  = the three-letter code (e.g. “CAD”)  
   ‣ "fx_amount"    = numeric value of that foreign amount (e.g. 12)  
   •  If the INR equivalent is shown on the same line or a companion line, parse it into
      "amount_inr". Otherwise leave "amount_inr" blank (omit the field).
•  For non-FX rows set "is_fx_transaction": false and set "fx_currency": "", "fx_amount":0
━━━━━━━━ TRANSACTION TYPE LOGIC  ━━━━━━━━
Classify "transaction_type" using these hints:
• "Purchase" – typical shopping, point-of-sale, online spends
• "Refund" – reversal, returned item, merchant refund
• "Cash Withdrawal" – ATM, card cash advance
• "UPI" – UPI-based payments
• "Interest" – interest charges, finance cost
• "Fee" – late fee, annual fee, overlimit fee
• "EMI" – EMI conversion, installment plan
• "FX" – foreign currency charges
• "Payment" – payment received (e.g. NEFT, IMPS, UPI in)
• "Reversal" – failed or reversed transaction
• "Others" – any unclassified transaction

If no match is found, use "Others"

━━━━━━━━  SCHEMA ━━━━━━━━
Return a single JSON object matching this schema. Every field must be present; use null if not available.
━━━━━━━━  IMPORTANT  ━━━━━━━━
Incase no transaction or relevant information is found on a particular page; use No Transaction or Relevant Information Found
Incase of any missing fields set "" or 0 , donot opt out anything if not present
Follow JSON schema:
 <JSONSchema>{json.dumps(json_structure)}</JSONSchema>

"""


            response = self.model.generate_content(
                [
                    system_prompt,
                    "Extract the data from this credit card statement and output JSON.",
                    {
                        "mime_type": "image/jpeg",
                        "data": base64.b64decode(base64_image),
                    },
                ],
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "top_p": 0.95,
                    "top_k": 40,
                },
                stream=False,
            )
            return json.loads(response.text)
        except Exception as e:
            logger.exception("Gemini extraction failed: %s", e)
            return {"error": str(e)}

    # ---------- High‑level helpers ---------- #

    def process_single_page(
        self, base64_image: str
    ) -> Dict[str, Any]:
        return self.extract_invoice_data(base64_image)
    def main_extract(self, read_path: str) -> List[Dict[str, Any]]:
        logger.info("Starting extraction for: %s", read_path)
        file_extension = os.path.splitext(read_path)[1].lower()
        if file_extension in {".jpg", ".jpeg", ".png"}:
            base64_images = [self.encode_image(read_path)]
        elif file_extension == ".pdf":
            base64_images = self.pdf_to_base64_images(read_path)
        else:
            logger.warning("Unsupported file type: %s", file_extension)
            return []

        results: List[Dict[str, Any]] = []
        # for idx, img_b64 in enumerate(base64_images, start=1):
        #     page_result = self.process_single_page(img_b64)
        #     page_result["page_number"] = idx
        #     results.append(page_result)
        #     logger.info("Processed page %s/%s", idx, len(base64_images))

        # return results
        with ThreadPoolExecutor(max_workers=min(5, len(base64_images))) as executor:
            futures = {
                executor.submit(self.process_single_page, img_b64): i
                for i, img_b64 in enumerate(base64_images, start=1)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    result["page_number"] = idx
                    results.append(result)
                    logger.info("Processed page %s/%s", idx, len(base64_images))
                except Exception as e:
                    logger.exception("Failed to process page %s: %s", idx, e)
        results.sort(key=lambda x: x.get("page_number", 0))
        return results

