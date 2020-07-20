import numpy as np
from bokeh.io import curdoc, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, TextInput, Tabs, Panel, Button, DataTable, Div, CheckboxGroup
from bokeh.models.widgets import NumberFormatter, TableColumn, Dropdown, RadioButtonGroup, Select
from bokeh.plotting import figure
from bokeh.models import CustomJS, HoverTool, NumeralTickFormatter
from pprint import pprint
import pandas as pd
import requests
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
from pprint import pprint

"""
DEFAULTS
"""
plt.style.use('default')
PHI = 1.618
w = 12*60*60*1000 # half day in ms
"""
SETUP DATA
"""
def get_data(market='Tezos'):
    z = datetime.datetime.today()
    z.strftime("%x")
    temp = str(z).split('-')
    current_day = temp[0]+temp[1]+temp[2].split(" ")[0]
    web = requests.get(f"https://coinmarketcap.com/currencies/{market.lower()}/historical-data/?start=20130428&end=" + current_day)
    dfs = pd.read_html(web.text)
    data = dfs[2]
    data = data.iloc[::-1]
    # print(data)
    data['Date'] = pd.to_datetime(data['Date'])
    LENGTH = 30
    window1, window2 = LENGTH, 7*LENGTH
    data[f'{window1} Day MA'] = data['Close**'].rolling(window=window1).mean()
    data[f'{window1} Week MA'] = data['Close**'].rolling(window=window2).mean()
    data['Risk'] = data[f'{window1} Day MA']/data[f'{window1} Week MA']
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data[['Risk']])
    data['Risk'] = np_scaled
    return data

df = get_data()
source = ColumnDataSource(df)

"""
SETUP PLOTS
"""
intro = Select(title="Cryptocurrency", value="Tezos",
               options=['Bitcoin', 'Ethereum', 'Litecoin', 'Verge', 'Chainlink', 'Tezos'])

inc = df['Close**'] > df['Open*']
dec = df['Open*'] > df['Close**']
price = figure(plot_height=600, plot_width=int(PHI*600), title="Tezos", tools="crosshair,pan,reset,save,wheel_zoom", x_axis_type="datetime")
price.line(x='Date', y='Close**', line_width=1, line_alpha=0.6, source=source)
price.xaxis.major_label_orientation = np.pi/4
price.grid.grid_line_alpha=0.3
price.segment(df['Date'], df['High'], df['Date'], df['Low'], color="black")
price.vbar(df['Date'][inc], w, df['Open*'][inc], df['Close**'][inc], fill_color="#D5E1DD", line_color="black")
price.vbar(df['Date'][dec], w, df['Open*'][dec], df['Close**'][dec], fill_color="#F2583E", line_color="black")

ma = figure(plot_height=600, plot_width=int(PHI*600), title="Moving Averages", tools="crosshair,pan,reset,save,wheel_zoom", x_axis_type="datetime")
ma.xaxis.major_label_orientation = np.pi/4
ma.grid.grid_line_alpha=0.3
ma.line(x='Date', y="30 Day MA", line_width=1, line_alpha=1, source=source, line_color='red', legend_label='30 Day MA')
ma.line(x='Date', y="30 Week MA", line_width=1.618, line_alpha=0.6, source=source, line_color='green', legend_label='30 Week MA')

risk = figure(plot_height=600, plot_width=int(PHI*600), title="Risk", tools="crosshair,pan,reset,save,wheel_zoom", x_axis_type="datetime")
risk.xaxis.major_label_orientation = np.pi/4
risk.grid.grid_line_alpha=0.3
risk.line(x='Date', y="Risk", line_width=1, line_alpha=1, source=source, line_color='red', legend_label='Risk')
risk.line(x=source.data['Date'], y=0.3, source=source, line_width=1, line_alpha=1, line_color='red', legend_label='Risk')

"""
Setting up widgets
"""

"""
Set up callbacks
"""
def callback(attr, old, new):
    # print(attr, old, new)
    df = get_data(intro.value)
    # print("Got data")
    source.data = df.to_dict('list')
    # print("Updated Data.")
    price.title.text = intro.value

intro.on_change('value', callback)

# Set up layouts and add to document

tab0 = row(intro)
tab0 = Panel(child=tab0, title="Crypto Selection")
tab1 = row(price, width=int(PHI*400))
tab1 = Panel(child=tab1, title="Price")
tab2 = row(ma, width=int(PHI*400))
tab2 = Panel(child=tab2, title="Moving Averages")
tab3 = row(risk, width=int(PHI*400))
tab3 = Panel(child=tab3, title="Risk")
tabs = Tabs(tabs=[tab0, tab1, tab2, tab3])

curdoc().title = "CMC Dashboard"
curdoc().theme = 'caliber'
curdoc().add_root(tabs)
