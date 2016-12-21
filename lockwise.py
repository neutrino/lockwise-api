# coding = utf8
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

@app.route('/',  methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    message = "TODO!"
    return render_template('index.html', message = message)
  else:
    return render_template('index.html')


@app.route('/check',  methods=['POST'])
def check():
  try:
    data = request.get_json()
    app.logger.debug(data)
    response = {
      'percentage' : 75,
      'status' : 'ok'
    }
    return jsonify(data)
  except:
    response = {
      'message' : 'Sorry, something went wrong.',
      'status' : 'error'
    }
    return jsonify(message), 400
    # return jsonify()


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
