#main.py
from bokeh.embed import server_document
<<<<<<< HEAD
from bokeh.util.string import encode_utf8
=======
from flask import Flask, request, render_template
>>>>>>> 98fec8ca0a2bc5bfa2ce1c0a1b40298346a1e45b
import subprocess

app = Flask(__name__)

<<<<<<< HEAD
subprocess.call("bokeh serve --allow-websocket-origin=* dashboard.py")
=======
bokeh_process = subprocess.Popen(
    ['bokeh', 'serve','--allow-websocket-origin=*', 'dashboard.py'], stdout=subprocess.PIPE)

>>>>>>> 98fec8ca0a2bc5bfa2ce1c0a1b40298346a1e45b

@app.route('/')
def index():
    script = server_document("http://localhost:5006/dashboard")
    return render_template('index.html', plot_script=script)

if __name__ == '__main__':
    app.run(port='80')
