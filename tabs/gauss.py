#GAUSS
import numpy as np
from bokeh.io import curdoc, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, TextInput, Tabs, Panel, Button, DataTable, Div, CheckboxGroup
from bokeh.models.widgets import NumberFormatter, TableColumn, Dropdown, RadioButtonGroup, Select
from bokeh.plotting import figure
from pprint import pprint
import pandas as pd

N = 1000
sigma = 1
pi = np.pi
phi = 1.618
ratio = 0
mu = 0
x1 = np.linspace(mu - 6 * sigma, mu + 6 * sigma, N)
y1 = 1/(sigma*np.sqrt(2*pi))*np.exp(-0.5*((x1-mu)/sigma)**2)
source1 = ColumnDataSource(data=dict(x=x1, y=y1))
u1 = np.linspace(mu - 6 * sigma, mu + 6 * sigma, N)
v1 = 1/(sigma*np.sqrt(2*pi))*np.exp(-0.5*((u1-mu)/sigma)**2)
source1_reference = ColumnDataSource(data=dict(x=u1, y=v1))

plot1 = figure(plot_height=600, plot_width=int(phi*600), title="Oh my Gauss",
              tools="save", x_range=[x1.min(), x1.max()], y_range=[0, phi*y1.max()])
plot1.line('x', 'y', source=source1_reference, line_width=3*1/phi, line_alpha=0.6, line_color="gray", line_dash='dashed', legend_label='Reference')
plot1.line('x', 'y', source=source1, line_width=3, legend_label="Your Gauss")

div1 = Div(text="""<p style="border:3px; border-style:solid; border-color:#FF0000; padding: 1em;">
                    Oh My Gauss is designed to let you explore what various kinds of normal distributions look like. 
                    Try changing the Standard Deviation or Average to see how this affects the plot.</p>""",
                    width=300, height=130)
title1 = TextInput(title="Plot Title", value='Oh my Gauss')
sigma = TextInput(title="Standard Deviation", value="1.0")
mu = TextInput(title="Average", value="0.0")
recompute1 = Button(label="Recompute", button_type="success")
reset1 = Button(label="Reset", button_type="success")
checkbox1 = CheckboxGroup(labels=["Reference"], active=[1, 0])

