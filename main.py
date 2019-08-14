#main.py
from bokeh.embed import server_document
from flask import Flask, request, render_template
import subprocess

app = Flask(__name__)

bokeh_process = subprocess.Popen(
    ['bokeh', 'serve','--allow-websocket-origin=*','dashboard.py'], stdout=subprocess.PIPE)

@app.route('/')
def index():
    script = server_document("http://localhost:5006/dashboard")
    print(f"Script: {script}")
    return render_template('index.html', plot_script=script)

if __name__ == '__main__':
    app.run(port='8080')
