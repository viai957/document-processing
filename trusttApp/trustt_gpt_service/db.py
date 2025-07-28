import psycopg2
from psycopg2 import pool, OperationalError
from flask import  g
import logging
import sys
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
POSTGRESQL = {
        'database': os.getenv("POSTGRES_DATABASE"),
        'user': os.getenv("POSTGRES_USER"),
        'password': os.getenv("POSTGRES_PASSWORD"),
        'host': os.getenv("POSTGRES_HOST"),
        'port': os.getenv("POSTGRES_PORT")
    }
DB_SCHEMA = os.getenv("DB_SCHEMA")
MAX_CONNECTIONS = os.getenv("MAX_POOL_SIZE")



class Database:
    _connection_pool = None

    @staticmethod
    def init_db(app):
        db_config = POSTGRESQL
        try:
            Database._connection_pool = pool.SimpleConnectionPool(1, MAX_CONNECTIONS, **db_config)
            
            logger.info("Database connection pool initialized successfully.")
            # Register teardown so each request returns the connection to the pool
            @app.teardown_appcontext
            def teardown_db(exception):
                logger.info("Database connection teardown.:exception:"+str(exception))
                close_db(exception)
        except Exception as e:
            logger.exception("Failed to create database connection pool.")

    @staticmethod
    def get_connection():
        try:
            return Database._connection_pool.getconn()
        except Exception as e:
            logger.exception("Failed to get connection from pool.")
            raise

    @staticmethod
    def put_connection(connection):
        try:
            Database._connection_pool.putconn(connection)
        except Exception as e:
            logger.exception("Failed to return connection to pool.")

    @staticmethod
    def close_all_connections():
        try:
            Database._connection_pool.closeall()
            logger.info("All database connections closed.")
        except Exception as e:
            logger.exception("Failed to close all database connections.")

def get_db():
    if 'db_conn' not in g:
        g.db_conn = Database.get_connection()
        logger.info("Database connection acquired.")
    return g.db_conn

def close_db(e=None):
    db_conn = g.pop('db_conn', None)
    if db_conn:
        Database.put_connection(db_conn)

def close_connection(exception):
    close_db(exception)
    if exception:
        logger.error("Error during request teardown: %s", exception)

def run_query(query, args=None, one=False, many=False, commit=False):
    conn = get_db()
    result = None
    if query.find("tgpt_dms")>=0:
        logger.debug("Executing query: %s", query)
    else:
        logger.debug("Executing query: %s", query + "  ::  Args:"+str(args)) 
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SET search_path TO "{DB_SCHEMA}" """)  
            if many:
                cur.executemany(query, args)
                rows_affected = cur.rowcount
            else:
                cur.execute(query, args) 
            
            if cur.description:
                result = cur.fetchone() if one else cur.fetchall()

            if commit:
                conn.commit()
                return result if result is not None else cur.rowcount
            else:
                return result
            # if commit:
            #     conn.commit()
            #     affected = cur.rowcount if many else cur.lastrowid
            #     return affected
            # if cur.description:
            #     result = cur.fetchone() if one else cur.fetchall()
            #     logger.info("Query executed successfully.")
    except OperationalError as e:
        logger.error("Operational error during query execution: %s", e)
        conn.rollback()
    except Exception as e:
        logger.error("Unexpected error during query execution: %s", e)
        conn.rollback()
        # if result is None:
        #     logger.info("No result found."+"  ;Query:"+query)  
        #     return result
    finally:
        if result is not None:
            try:
                logger.info("Result obtained successfully")
            except UnicodeEncodeError as log_error:
                logger.debug("UnicodeEncodeError")
            
            return result

def query_without_params(query):
    return run_query(query)

def insert_data(query, data):
    return run_query(query, args=data, commit=True)

def insert_data_many(query, data_list):
    return run_query(query, args=data_list, many=True, commit=True)

def select_data(query, value):
    return run_query(query, args=(value,), one=True)

def fetch_all_data(query, data):
    return run_query(query, args=(data,))
