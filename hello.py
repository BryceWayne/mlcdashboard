from flask import Flask, flash, redirect, render_template, request, session, abort
from bokeh.embed import autoload_server
app = Flask(__name__)

@app.route("/")
def hello():
    script=autoload_server(model=None,app_path="/bokeh-sliders",url="http://localhost:5006")
    return render_template('hello.html',bokS=script)

if __name__ == "__main__":
    app.run()
