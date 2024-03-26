from flask import Flask, make_response, request, jsonify
from database import *
from flask_cors import CORS
from routes import user_routes
from flask_mail import Mail, Message




app = Flask(__name__)
CORS(app)
db_pool = DBPool()
mail = Mail(app)

user_routes(app, mail)
# Create tables when the application starts
with app.app_context():
    db_pool.create_table()



@app.route('/hello', methods= ['GET'])
def hello_world():
    return jsonify(message="Hello, World!")


# Route to get all languages
@app.route("/get_languages", methods=["GET"])
def get_languages():
    try:
        conn = db_pool.get_instance().getconn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM languages")
            result = cursor.fetchall()
        return jsonify({"languages": result})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn:
            db_pool.get_instance().putconn(conn)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
