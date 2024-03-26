import datetime
from flask import jsonify
import logging
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import os
import jwt
import json


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv('JWT_SECRET')
DB_NAME  = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT')
DB_HOST = os.getenv('DB_HOST')

class DBPool:
    _instance = None

    @staticmethod
    def get_instance():
        if DBPool._instance == None:
            DBPool._instance = pool.ThreadedConnectionPool(
                minconn=1, maxconn=10,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME
            )
        return DBPool._instance

    def create_table(self):
        with DBPool.get_instance().getconn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user')")
                table_exists = cursor.fetchone()[0]
                if not table_exists:
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
                        interests JSON[],
                        language_proficiency TEXT,
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
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS goals (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        goals JSONB
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS preference_learning_style (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        learning_style JSONB
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS accessibility_needs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        accessibility_need JSONB
                        )
                    """)
                    conn.commit()
                    logger.info("Database tables created successfully")
                    return jsonify({"message": "Database created successfully"})
                else:
                    logger.error("Database creation failed: %s", exc_info=True)
                    return jsonify({"message": "Database creation failed"})




def create_user(username, hashed_password, email, first_name, last_name, date_of_birth, gender, interests, goals, preferred_learning_style, language_proficiency, accessibility_needs, last_login, account_created, account_updated, profile_picture, settings, progress_tracking, feedback_history, privacy_settings, verification_token):
    token_expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=3)
    
    try:
        with DBPool.get_instance().getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (
                    username, password, email, first_name, last_name, date_of_birth, gender, interests, goals, preferred_learning_style, language_proficiency, accessibility_needs, last_login, account_created, account_updated, profile_picture, settings, progress_tracking, feedback_history, privacy_settings, verification_token, token_expiration, verified
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (username, hashed_password, email, first_name, last_name, date_of_birth, gender, json.dumps(interests), goals, preferred_learning_style, language_proficiency, accessibility_needs, last_login, account_created, account_updated, profile_picture, json.dumps(settings), json.dumps(progress_tracking), json.dumps(feedback_history), json.dumps(privacy_settings), verification_token, token_expiration, False))

                user_id = cur.fetchone()[0]
                conn.commit()
                logger.info("User created successfully")
                return jsonify({"message": "User created successfully", "user_id": user_id}), 200

    except psycopg2.errors.UniqueViolation as e:
        logger.error('Duplicate email or username', exc_info=True)
        return jsonify({"error": "Duplicate email or username"}, 409)

    except psycopg2.Error as e:
        logger.error(f'Failed to insert user: {e}', exc_info=True)
        return jsonify({"error": "Failed to insert user"}, 500)

    except Exception as e:
        logger.error(f'Failed to insert user: {e}', exc_info=True)
        return jsonify({"error": "Database error"}, 500)


def generate_verification_token(email):
    try:
        expiration_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)
        token = jwt.encode({'email': email, 'exp': expiration_time}, JWT_SECRET, algorithm='HS256')
        # Ensuring that the token is a string
        verification_token = token.decode('utf-8') if isinstance(token, bytes) else token
        logger.info('Verification token generated: %s', verification_token)
        return verification_token
    except Exception as e:
        logger.error(f'Failed to generate verification token: {e}', exc_info=True)
        return None

def get_logger():
    return logger