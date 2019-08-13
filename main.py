from flask import Flask, render_template
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.io import show, output_file, curdoc
from bokeh.embed import server_document
from bokeh.models import ColumnDataSource, CustomJS, Slider, Range1d
from bokeh.models.widgets import Slider, TextInput, Tabs, Panel, Button, DataTable
from bokeh.models.widgets import NumberFormatter, TableColumn, Dropdown, RadioButtonGroup, Select
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
	# render template
	script = server_document("http://localhost:5006/dashboard")
	html = render_template('index.html', plot_script=script)
	return encode_utf8(html)


if __name__ == '__main__':
    app.run(port='8080')
