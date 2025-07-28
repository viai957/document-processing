import requests
import json 
from flask import jsonify
from datetime import datetime
import logging
from common.logger import setup_logger
import random
from dotenv import load_dotenv
import os
from dateutil.relativedelta import relativedelta

load_dotenv()
BASE_URL="https://sandbox.veri5digital.com"
BASE_PATH="/video-id-kyc/api/1.0/"

LOG_LEVEL =os.getenv("LOG_LEVEL")
PLATFORM_LOG_NAME = "veri5_service.log"

class Veri5ServiceAPI :
    logger = setup_logger('veri5_gateway', PLATFORM_LOG_NAME, LOG_LEVEL)

    @staticmethod
    def get_timestamp():
        timestamp= datetime.now().strftime("%H%M%S")
        random_digits = str(random.randint(1000, 9999))  # Generate a 4-digit random number
        return timestamp + random_digits

    @staticmethod
    def common_headers():
        """ Generate common headers for API requests. """
        current_time = Veri5ServiceAPI.get_timestamp()
        headers = {
                    "client_code": "TEST3740",
                    "sub_client_code": "TEST3740",
                    "channel_code": "ANDROID_SDK",
                    "channel_version": "3.1.7",
                    "stan": current_time,
                    "client_ip": "",
                    "transmission_datetime": current_time,
                    "operation_mode": "SELF",
                    "run_mode": "DEFAULT",
                    "actor_type": "DEFAULT",
                    "user_handle_type": "EMAIL",
                    "user_handle_value": "abc@gmail.com",
                    "location": "",
                    "function_code": "DEFAULT",
                    "function_sub_code": "DEFAULT"
                }
        return headers

    @staticmethod
    def send_request(endpoint, headers, request_body):
        full_url = f"{BASE_URL}{BASE_PATH}{endpoint}"   
        Veri5ServiceAPI.logger.debug("Full URL: "+str(full_url))

        try:
            response = requests.post(full_url, headers=headers, json=request_body,verify=False)
        except Exception as e:
            Veri5ServiceAPI.logger.debug("Error while calling external API: "+str(e))
            return None

        response_content= response.json()
        

        Veri5ServiceAPI.logger.info("Calling external API: "+full_url+"  with header "+str(headers)+" with request body: "+str(request_body))
        Veri5ServiceAPI.logger.info("Response from external API: "+str(response_content))
        response.raise_for_status()  # will raise an HTTPError for bad responses
        return response.json()
    

    @staticmethod
    def bget_timestamp():
        timestamp= datetime.now().strftime("%H%M%S")
        random_digits = str(random.randint(1000, 9999))  # Generate a 4-digit random number
        return timestamp + random_digits
       

    @staticmethod
    def extractDocInfo(document_type, document_side, extraction_type, document_front_image="", document_back_image=""):
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json",
        }
        Veri5ServiceAPI.logger.info("In extractDocInfo")
        request_headers = Veri5ServiceAPI.common_headers()

        request_body = {
            "headers": request_headers,
            "request":{
                "request_id":"test1250_2119",
                "hash":"344210b1f5c4fc16389469d142f2035c29d0d6969066b2d7c9eece124de5e8ec",
                "api_key":"BYPASS",
                "purpose":"EDUCATION",
                "document_type": document_type,
                "document_side": document_side,
                "extraction_type": extraction_type,
            }
        }
        if document_front_image!="":
            Veri5ServiceAPI.logger.info("Document front image is not empty")
            request_body["request"]["document_front_image"] = document_front_image
        if document_back_image!="":
            Veri5ServiceAPI.logger.info("Document back image is not empty")
            request_body["request"]["document_back_image"] = document_back_image

        Veri5ServiceAPI.logger.info("Request headers for extractDocInfo: "+str(headers))
        Veri5ServiceAPI.logger.info("Request body for extractDocInfo: "+str(request_body))
        response = Veri5ServiceAPI.send_request("docInfoExtract", headers, request_body)
        return response
        
    @staticmethod
    def extract_bill_data(file_obj, file_type="pdf", doc_type="invoice", api_key="OPOfwhk3z44keJ7Lqrsn9L2gr5TCTzpF-5A2C9F899D464"):
        """
        Sends a multipart/form-data POST request to the Veri5Digital extract_bill_data endpoint.
        Args:
            file_obj: File-like object to upload (e.g., from Flask request.files['file']).
            file_type (str): Type of the file (default: 'pdf').
            doc_type (str): Document type (default: 'invoice').
            api_key (str): API key for authentication.
        Returns:
            dict: JSON response from the API.
        """
        url = "https://qa.veri5digital.com/api/v1/extract_bill_data"
        headers = {
            "accept": "application/json"
        }
        data = {
            "type": file_type,
            "doc_type": doc_type,
            "api_key": api_key
        }
        files = {"file": (file_obj.filename, file_obj.stream, file_obj.mimetype)}
        Veri5ServiceAPI.logger.info(f"Sending file {file_obj.filename} to {url}")
        try:
            response = requests.post(url, headers=headers, data=data, files=files, verify=False)
            response.raise_for_status()
            Veri5ServiceAPI.logger.info(f"Response: {response.text}")
            return response.json()
        except Exception as e:
            Veri5ServiceAPI.logger.error(f"Error in extract_bill_data: {e}")
            return None
