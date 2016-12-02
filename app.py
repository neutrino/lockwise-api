# coding = utf8
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/',  methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    message = "TODO!"
    return render_template('index.html', message=message)
  else:
    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
