#main.py
from bokeh.embed import server_document
from bokeh.util.string import encode_utf8
from flask import Flask, request, render_template
import subprocess

app = Flask(__name__)

subprocess.call("bokeh serve --allow-websocket-origin=* dashboard.py")

@app.route('/')
def index():
    script = server_document("http://localhost:5006/dashboard")
    return render_template('index.html', plot_script=script)

if __name__ == '__main__':
    app.run(port='80')
