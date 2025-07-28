from trustt_gpt_service import db
import json
from psycopg2.extras import Json 
import logging
from dotenv import load_dotenv
import os
load_dotenv()

class SessionRepository:

    def __init__(self,db):
        self.db = db
    
    def insertWorkItem(self, client_code, work_type, work_id, assignee_id, assignee_name, create_on, created_by):
        query = '''INSERT INTO work_item (client_code, work_type, work_id, assignee_id, assignee_name, create_on, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id'''
        return db.run_query(query, (client_code, work_type, work_id, assignee_id, assignee_name, create_on, created_by), commit=True)
    
    def insertWorkDocInfo(self, work_item_id, doc_type, doc_id, doc_file_name, doc_type_source):
        query = '''INSERT INTO work_doc_info (work_item_id, doc_type, doc_id, doc_file_name, doc_type_source)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id'''
        return db.run_query(query, (work_item_id, doc_type, doc_id, doc_file_name, doc_type_source), commit=True)
    
    def insertBankStmt(self, work_doc_info_id, stmt_id, bank_name, ac_holder_name, ac_num, ifsc_code, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified):
        query = '''INSERT INTO doc_bank_stmt (work_doc_info_id, stmt_id, bank_name, ac_holder_name, ac_num, ifsc_code, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id'''
        return db.run_query(query, (work_doc_info_id, stmt_id, bank_name, ac_holder_name, ac_num, ifsc_code, total_pages, dms_id, Json(verification_metadata), Json(extracted_data), Json(txn_json), analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified), commit=True)

    def fetchTxnJson(self, stmt_id):
        query = '''SELECT txn_json, verification_metadata FROM doc_bank_stmt WHERE stmt_id = %s'''
        result = db.run_query(query, (stmt_id,), one=True)
        return result
    
    def updateBankStmt(self, stmt_id, ac_holder_name, ac_num, ifsc_code, verification_metadata, txn_json, analysis_status, txn_accuracy_percentage, update_on, updated_by):
        query = '''UPDATE doc_bank_stmt
                   SET ac_holder_name = %s, ac_num = %s, ifsc_code = %s, verification_metadata = %s, txn_json = %s, analysis_status = %s, txn_accuracy_percentage = %s, updated_on = %s, updated_by = %s
                   WHERE stmt_id = %s'''
        return db.run_query(query, (ac_holder_name, ac_num, ifsc_code, Json(verification_metadata), Json(txn_json), analysis_status, txn_accuracy_percentage, update_on, updated_by, stmt_id), commit=True)
    
    def fetchBankStmtById(self, stmt_id):
        query = '''SELECT txn_json, verification_metadata, analysis_status FROM doc_bank_stmt WHERE stmt_id = %s'''
        result = db.run_query(query, (stmt_id,), one=True)
        return result
    
    def fetchInitialIdByClientCode(self, client_code):
        query = '''SELECT id FROM work_item WHERE client_code = %s ORDER BY create_on ASC LIMIT 1'''
        result = db.run_query(query, (client_code,), one=True)
        return result

    def updateHumanVerification(self, stmt_id, is_human_verified, verified_by, verified_time):
        query = '''UPDATE doc_bank_stmt
                   SET is_human_verified = %s, verified_by = %s, verified_time = %s
                   WHERE stmt_id = %s'''
        return db.run_query(query, (is_human_verified, verified_by, verified_time, stmt_id), commit=True)
    
    def insertIdentityDoc(self, work_doc_info_id, stmt_id, extracted_data, created_on, created_by, is_deleted, total_pages):
        query = '''INSERT INTO doc_poi_poa (work_doc_info_id, stmt_id, extracted_data, created_on, created_by, is_deleted, total_pages)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id'''
        return db.run_query(query, (work_doc_info_id, stmt_id, Json(extracted_data), created_on, created_by, is_deleted, total_pages), commit=True)
    
    def insertCCStmt(self, work_doc_info_id, cc_stmt_id, bank_name, ac_holder_name, ac_num, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified):
        query = '''INSERT INTO doc_cc_stmt (work_doc_info_id, cc_stmt_id, bank_name, ac_holder_name, ac_num, total_pages, dms_id, verification_metadata, extracted_data, txn_json, analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id'''
        return db.run_query(query, (work_doc_info_id, cc_stmt_id, bank_name, ac_holder_name, ac_num, total_pages, dms_id, Json(verification_metadata), Json(extracted_data), Json(txn_json), analysis_status, txn_accuracy_percentage, create_on, created_by, is_human_verified), commit=True)