from flask import Flask, render_template
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column

app = Flask(__name__)


@app.route('/')
def index():
	"""Atempt 1"""
	# fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
	# counts = [5, 3, 4, 2, 4, 6]

	# source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

	# p = figure(x_range=fruits, plot_height=600, plot_width=1000, toolbar_location=None, title="Fruit Counts")
	# p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
	#        line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

	# p.xgrid.grid_line_color = None
	# p.y_range.start = 0
	# p.y_range.end = 9
	# p.legend.orientation = "horizontal"
	# p.legend.location = "top_center"

	"""Attempt 2"""
	x = np.linspace(0, 10, 500)
	y = np.sin(x)

	source = ColumnDataSource(data=dict(x=x, y=y))

	p = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

	p.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

	amp_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
	freq_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
	phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
	offset_slider = Slider(start=-5, end=5, value=0, step=.1, title="Offset")

	callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
		                code="""
								const data = source.data;
								const A = amp.value;
								const k = freq.value;
								const phi = phase.value;
								const B = offset.value;
								const x = data['x']
								const y = data['y']
								for (var i = 0; i < x.length; i++) {
								    y[i] = B + A*Math.sin(k*x[i]+phi);
								}
								source.change.emit();
								""")

	amp_slider.js_on_change('value', callback)
	freq_slider.js_on_change('value', callback)
	phase_slider.js_on_change('value', callback)
	offset_slider.js_on_change('value', callback)

	layout = row(
	    plot,
	    column(amp_slider, freq_slider, phase_slider, offset_slider),
	)
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
