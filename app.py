from flask import Flask, request, jsonify
# ... other dependencies


app = Flask(__name__)

@app.route("/data", methods=["GET"])
def get_data():
    # Fetch or process data
    data = {"message": "Hello, World!"}  # Replace with actual data logic
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)