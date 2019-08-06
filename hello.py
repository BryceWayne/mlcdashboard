from flask import Flask, flash, redirect, render_template, request, session, abort
#from bokeh.embed import autoload_server
from bokeh.embed import server_document
import subprocess

subprocess.call('bokeh serve ./bokeh-sliders.py --allow-websocket-origin=127.0.0.1:5006')
app = Flask(__name__)

@app.route("/")
def hello():
    #script=autoload_server(model=None,app_path="/bokeh-sliders",url="http://localhost:5006")
    script=server_document("http://localhost:5006/bokeh-sliders")
    print(script)
    return render_template('hello.html',bokS=script)

if __name__ == "__main__":
    app.run()
