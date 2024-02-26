from flask import Flask, request, jsonify
import pymysql.cursors
from flask_cors import CORS
from database import DBPool



app = Flask(__name__)
db_pool = DBPool()

# Create tables when the application starts
db_pool.create_table()



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
