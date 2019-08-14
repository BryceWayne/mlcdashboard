from flask import Flask, render_template
from bokeh.embed import server_document
from bokeh.util.string import encode_utf8
import subprocess
import atexit

app = Flask(__name__)

bokeh_process = subprocess.Popen(["bokeh", 
				  "serve", 
				  "--allow-websocket-origin=*",
				  "dashboard.py"], stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route('/')
def index():
	session = pull_session(app_path='/')
    	bokeh_script = autoload_server(None, app_path="/", session_id=session.id)
	# render template
# 	script = server_document("http://localhost:5006/dashboard")
	html = render_template('index.html', plot_script=bokeh_script)
	return encode_utf8(html)


if __name__ == '__main__':
    app.run(port='8080')
