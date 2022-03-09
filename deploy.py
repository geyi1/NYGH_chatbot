# https://www.youtube.com/watch?v=a37BL0stIuM&t=1547s&ab_channel=PythonEngineer
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
# from train import model, lbl_encoder, data, max_len, tokenizer
from predict import predict_response as find_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = find_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
