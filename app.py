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
    return jsonify(data)
  except:
    message = {'error': 'Sorry, something went wrong.'}
    return jsonify(message), 400
    # return jsonify()



if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
