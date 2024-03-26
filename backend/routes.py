from database import *
from flask import make_response, request, jsonify
import bcrypt
from email_sender import sendVerificationEmail





def user_routes(app, mail):
    @app.route('/signup', methods=['POST', 'OPTIONS'])
    def signup():
        if request.method == 'OPTIONS':
            response = make_response(jsonify({'status': 'success'}), 200)
            response.headers['Access-Control-Allow-Origin'] = 'http://localhost:4200'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'POST'
            return _build_cors_preflight_response()
        try:
            data = request.get_json()
            logger.info(data)

            hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
            verification_token = generate_verification_token(data['email'])

            create_user(
                username=data['username'],
                hashed_password=hashed_password,
                email=data['email'],
                first_name=data['first_name'],
                last_name=data['last_name'],
                date_of_birth=data['date_of_birth'],
                gender=data['gender'],
                interests=data['interests'],
                goals=data['goals'],
                preferred_learning_style=data['preferred_learning_style'],
                language_proficiency=data['language_proficiency'],
                accessibility_needs=data['accessibility_needs'],
                last_login=data['last_login'],
                account_created=data['account_created'],
                account_updated=data['account_updated'],
                profile_picture=data['profile_picture'],
                settings=data['settings'],
                progress_tracking=data['progress_tracking'],
                feedback_history=data['feedback_history'],
                privacy_settings=data['privacy_settings'],
                verification_token=verification_token
            )            
            """email_sent = sendVerificationEmail(data['email'], verification_token, mail, app)
            if email_sent:
                return jsonify({"message": "User created successfully. Please check your email for verification"})
            else:
                return jsonify({"message": "Failed to send verification email"})"""

        except Exception as e:
            logger.error(f'Failed to create user: {e}', exc_info=True)
            return jsonify({"error": "Failed to create user"}), 500





    def _build_cors_preflight_response():
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:4200")
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response