from flask import Flask, render_template
from bokeh.embed import server_document
from bokeh.util.string import encode_utf8
import subprocess

app = Flask(__name__)

bokeh_process = subprocess.Popen(["bokeh", 
				  "serve", 
				  "--allow-websocket-origin=*",
				  "dashboard.py"], stdout=subprocess.PIPE)

@app.route('/')
def index():
	# render template
	script = server_document("http://localhost:5006/dashboard")
	html = render_template('index.html', plot_script=script)
	return encode_utf8(html)


if __name__ == '__main__':
    app.run(port='8080')
