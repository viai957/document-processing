from common.logger import setup_logger
from .db import get_db
from trustt_gpt_service.services import *
from flask import Blueprint, jsonify,request
import os
from trustt_gpt_service import services
from dotenv import load_dotenv
import tempfile


load_dotenv()
LOG_LEVEL =os.getenv("LOG_LEVEL","DEBUG")
TGPT_LOG_NAME = os.getenv("TGPT_LOG_NAME")

logger = setup_logger('trustt_gpt_service', TGPT_LOG_NAME, level=LOG_LEVEL)

# Initialize a Blueprint for this service
trustt_gpt_service = Blueprint('trustt_gpt_service', __name__)


@trustt_gpt_service.route('/extractJsonResponseFromCC', methods=['POST'])
def extract_data():
    try:
        files = request.files.getlist('files') 
        systemPrompt = request.form.get('systemPrompt','')
        
        all_results = []
        
        for file in files:
            
            if file.filename:
                temp_file_path = os.path.join('', file.filename)
            else:
                temp_file_path = os.path.join('', 'temp_file')
            print(f"Processing file: {file.filename}")
            file.save(temp_file_path)

            try:
                
                ocr_instance = Improved_PerformOCR()
                if file.filename:
                    file_extension = os.path.splitext(file.filename)[1].lower()
                else:
                    file_extension = ''
                
                if file_extension in ['.jpg', '.jpeg', '.png']:
                    
                    result = ocr_instance.main_extract(temp_file_path, systemPrompt)
                
                elif file_extension == '.pdf':
                    
                    result = ocr_instance.main_extract(temp_file_path, systemPrompt)
                else:
                    
                    document_result = {
                        "document_name": file.filename,
                        "result": {"error": f"Unsupported file type: {file_extension}"}
                    }
                    all_results.append(document_result)
                    continue

                document_result = {
                    "document_name": file.filename,
                    "result": result
                }
                all_results.append(document_result)
                
            except Exception as e:

                document_result = {
                    "document_name": file.filename,
                    "result": {"error": f"Failed to process document: {str(e)}"}
                }
                all_results.append(document_result)
            
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # with open(f"{file.filename}.json", 'w', encoding='utf-8') as f:
        #     json.dump(all_results, f, ensure_ascii=False, indent=4)
        return jsonify(all_results)

    except Exception as e:
        print(f"Error in extract_data route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/bsaBasic', methods=['POST'])
def bsabasic():
    try:

        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.performBasicAnalysis(request)

    except Exception as e:
        print(f"Error in bsabasic route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/updateTxn', methods=['POST'])
def updateTxn():
    try:

        bsaAnalysis = BankStatementAnalysis(get_db())
        return bsaAnalysis.updateTxn(request)

    except Exception as e:
        print(f"Error in updateTxn route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/extractBankStatement', methods=['POST'])
def extract_bank_statement():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.extractBankStatement(request)
    except Exception as e:
        print(f"Error in extract_bank_statement route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/getTxn', methods=['POST'])
def fetch_txn_json():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.getTxn(request)
    except Exception as e:
        print(f"Error in fetch_txn_json route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/previewTxn', methods=['POST'])
def previewTxn():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.previewTxn(request)
    except Exception as e:
        print(f"Error in fetch_updates route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/updateCustomerDetails', methods=['POST'])
def updateCustomerDetails():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.updateCustomerDetails(request)
    except Exception as e:
        print(f"Error in updateCustomerDetails route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/updateBsaHumanVerification', methods=['POST'])
def updateBsaHumanVerification():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.updateBsaHumanVerification(request)
    except Exception as e:
        print(f"Error in updateBsaHumanVerification route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/getDocAnalysis', methods=['POST'])
def getDocAnalysis():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.getDocAnalysis(request)
    except Exception as e:
        print(f"Error in getDocAnalysis route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/invoiceAnalysis', methods=['POST'])
def invoiceAnalysis():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.invoiceAnalysis(request)
    except Exception as e:
        print(f"Error in invoiceAnalysis route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/bsaBasicGemini', methods=['POST'])
def bsaBasicGemini():
    try:
        bsaAnalysis = BankStatementAnalysis(get_db())
        # Get the form data from the request
        return bsaAnalysis.performBasicAnalysisGemini(request)
    except Exception as e:
        print(f"Error in bsaBasicGemini route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500
    
@trustt_gpt_service.route('/ccAnalysis', methods=['POST'])
def ccAnalysis():
    try:
        ccAnalysis = CreditCardAnalysis(get_db())
        # Get the form data from the request
        return ccAnalysis.performCCAnalysisGemini(request)
    except Exception as e:
        print(f"Error in ccAnalysis route: {e} at line number {str(e.__traceback__.tb_lineno) if e.__traceback__ else 'unknown'}")
        return jsonify({"error": str(e)}), 500