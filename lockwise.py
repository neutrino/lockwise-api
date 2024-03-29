# coding = utf8
from flask import Flask, request, render_template, jsonify
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
import numpy as np
import re

app = Flask(__name__)

@app.route('/',  methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    input_text = request.form.get('content')
    if input_text == "":
      message = "It can not process empty text."
      return render_template('index.html', message = message)

    result = predict(input_text)
    if result:
      message = "This may have some sensitive data."
    else:
      message = "It didn't detect any sensitive information here."
    return render_template('index.html', message = message)
  else:
    return render_template('index.html')


@app.route('/check',  methods=['POST'])
def check():
  try:
    data = request.get_json()
    app.logger.debug(data)
    # build_data
    text_content = []
    if 'subject' in data:
      text_content.append("\nsubject\n")
      text_content.append(data["subject"])
    if 'body' in data:
      text_content.append("body\n")
      text_content.append(data["body"])
    # convert to a string
    input_text = ''.join(text_content)

    response = { }

    if input_text == "":
      response["error"] =  "Invalid Data!"
      return jsonify(data)

    # process the data
    result = predict(input_text)
    if result:
      response["message"] = "This may have some sensitive data."
    else:
      response["message"] = "It didn't detect any sensitive information here."
    response["status"] = 'ok'
    return jsonify(response)

  except:
    response = {
      'message' : 'Sorry, something went wrong.',
      'status' : 'error'
    }
    return jsonify(message), 400


def predict(input_text):
  testString = [input_text]
  modesSaved=joblib.load('processed/modes.sav')
  vectorizer = modesSaved[0]
  lsa = modesSaved[1]
  classifier = modesSaved[2]

  testMatrix = vectorizer.transform(testString) # string into tfidf matrix
  X_test = lsa.transform(testMatrix) # dimensionality reduction
  X_test = Normalizer(copy=False).transform(X_test) # normalization
  Y_predict = classifier.predict(X_test)


  testMatrix = vectorizer.transform(testString) # string into tfidf matrix
  X_test = lsa.transform(testMatrix) # dimensionality reduction
  X_test = Normalizer(copy=False).transform(X_test) # normalization
  Y_predict = classifier.predict(X_test) # predict with KNN method in our case

  print(Y_predict[0])

  if Y_predict[0] == 0.0:
    return True
  else:
    return False

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
