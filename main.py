from flask import Flask, render_template
from bokeh.embed import server_document

app = Flask(__name__)


@app.route('/')
def index():
	# render template
	script = server_document("http://localhost:5006/dashboard")
	html = render_template('index.html', plot_script=script)
	return encode_utf8(html)


if __name__ == '__main__':
    app.run(port='8080')
