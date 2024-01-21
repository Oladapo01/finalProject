from flask import Flask, request, jsonify
import pymysql.cursors



app = Flask(__name__)

def get_db_connection():
    return pymysql.connect(host='mysql',
                           user='devops',
                           password='devops',
                           database='devops',
                           cursorclass=pymysql.cursors.DictCursor)


# Route to add a language entry
@app.route("/add_language", methods=["POST"])
def add_language():
    data = request.get_json()
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO `languages` (`english`, `taino`, `french`, `latin`, `spanish`, `gaelic`) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (data['english'], data['taino'], data['french'], data['latin'], data['spanish'], data['gaelic']))
        connection.commit()
    finally:
        connection.close()
    return jsonify({"message": "Language added successfully"})

# Route to get all languages
@app.route("/get_languages", methods=["GET"])
def get_languages():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM `languages`")
            result = cursor.fetchall()
        return jsonify({"languages": result})
    finally:
        connection.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
