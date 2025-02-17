from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("fake_news_model.pkl")

@app.route("/detect", methods=["POST"])
def detect():
    text = request.json["text"]
    prediction = model.predict([text])[0]
    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
