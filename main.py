'''This example demonstrates embedding an autoloaded Bokeh server
into a simple Flask application, and passing arguments to Bokeh.
To view the example, run:
    python flask_server.py
in this directory, and navigate to:
    http://localhost:5000
'''
import atexit
import subprocess

from flask import render_template_string, Flask
from bokeh.embed import server_document

home_html = """
<!DOCTYPE html>
<html lang="en">
  <body>
    <div class="bk-root">
      <h1><a href="/batch/1"> Batch 1 (cos)</a></h1>
      <h1><a href="/batch/2"> Batch 2 (sin)</a></h1>
      <h1><a href="/batch/3"> Batch 3 (tan)</a></h1>
    </div>
  </body>
</html>
"""

app_html = """
<!DOCTYPE html>
<html lang="en">
  <body>
    {{ bokeh_script|safe }}
  </body>
</html>
"""

app = Flask(__name__)

bokeh_process = subprocess.Popen(
    ['python', '-m', 'bokeh', 'serve', '--allow-websocket-origin=localhost:8080', 'bokeh1.py'], stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route('/')
def home():
    return render_template_string(home_html)

@app.route('/batch/<int:batchid>')
def visualization(batchid):
    bokeh_script = server_document(url='http://localhost:8080/bokeh_server', arguments=dict(batchid=batchid))
    return render_template_string(app_html, bokeh_script=bokeh_script)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
