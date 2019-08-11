from flask import Flask, render_template
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

app = Flask(__name__)


@app.route('/')
def index():
	fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
	counts = [5, 3, 4, 2, 4, 6]

	source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

	p = figure(x_range=fruits, plot_height=600, plot_width=1000, toolbar_location=None, title="Fruit Counts")
	p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
	       line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

	p.xgrid.grid_line_color = None
	p.y_range.start = 0
	p.y_range.end = 9
	p.legend.orientation = "horizontal"
	p.legend.location = "top_center"

	# grab the static resources
	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()

	# render template
	script, div = components(p)
	html = render_template(
	    'index.html',
	    plot_script=script,
	    plot_div=div,
	    js_resources=js_resources,
	    css_resources=css_resources,
	)
	return encode_utf8(html)


if __name__ == '__main__':
    app.run(port='8080')
