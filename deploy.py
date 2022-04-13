from flask import Flask, render_template, request, jsonify
from predict import predict_response as find_response

'''
this is the python file to run to launch the chatbot on localhost
'''

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
