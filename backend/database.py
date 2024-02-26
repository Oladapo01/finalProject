from flask import jsonify
import logging
from psycopg2 import pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DBPool:
    _instance = None

    @staticmethod
    def get_instance():
        if DBPool._instance == None:
            DBPool._instance = pool.ThreadedConnectionPool(
                minconn=1, maxconn=10,
                user="project",
                password="project",
                host="postgresql",
                port="5432",
                database="final_project"
            )
        return DBPool._instance

    def create_table(self):
        conn = None
        try:
            conn = DBPool.get_instance().getconn()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS languages (
                    id SERIAL PRIMARY KEY,
                    english TEXT,
                    latin TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    date_of_birth DATE NOT NULL,
                    gender TEXT,
                    interests TEXT,
                    goals TEXT, 
                    preferred_learning_style TEXT,
                    language_proficiency TEXT,
                    accessibility_needs TEXT,
                    last_login TIMESTAMP,
                    account_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    account_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    profile_picture TEXT,
                    settings JSONB,
                    progress_tracking JSONB,
                    feedback_history JSONB,
                    privacy_settings JSONB,
                    verification_token VARCHAR(255),
                    token_expiration TIMESTAMP,
                    verified BOOLEAN NOT NULL DEFAULT FALSE
                    )
                """)
            conn.commit()
            logger.info("Database tables created successfully")
            return jsonify({"message": "Database created successfully"})
        except Exception as e:
            logger.error("Database creation failed: %s", e)
            return jsonify({"message": "Database creation failed"})
        finally:
            if conn:
                conn.close()
                logger.info("Connection closed")
