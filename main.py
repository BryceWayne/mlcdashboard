#main.py
from bokeh.embed import server_document
from flask import Flask
import subprocess

script = server_document("https://demo.bokeh.org/sliders")
app = Flask(__name__)

bokeh_process = subprocess.Popen(
    ['bokeh', 'serve','--allow-websocket-origin=*','dashboard.py'], stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route('/')
def index():
    script = server_document("http://localhost:5006/")
    return render_template_string('index.html', bokeh_script=script)

if __name__ == '__main__':
    app.run(port='8080')
