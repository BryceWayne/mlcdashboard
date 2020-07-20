import numpy as np
from bokeh.io import curdoc, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, TextInput, Tabs, Panel, Button, DataTable, Div, CheckboxGroup
from bokeh.models.widgets import NumberFormatter, TableColumn, Dropdown, RadioButtonGroup, Select
from bokeh.plotting import figure
<<<<<<< HEAD
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
=======
from pprint import pprint
import pandas as pd


"""
SETUP DATA
"""
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

x2 = np.random.randint(low=1, high=11, size=100)
unique2, counts2 = np.unique(x2, return_counts=True)
source2 = ColumnDataSource(data=dict(x=unique2, y=counts2))

# x3 = np.random
source3 = ColumnDataSource(data=dict(x=[], y=[]))
ratio_report3 = ColumnDataSource(data=dict(w=[0], x=[0], y=[0], z=[0], a=[0]))
running_averages = ColumnDataSource(data=dict(x=[], y=[], z=[]))

num_dice = 2
x4 = np.random.randint(low=1, high=7, size=1000)
for _ in range(num_dice-1):
    x4 += np.random.randint(low=1, high=7, size=1000)
unique4, counts4 = np.unique(x4, return_counts=True)
source4 = ColumnDataSource(data=dict(label=unique4, value=counts4, x=unique4, y=counts4, z=counts4/counts4.sum()))

source5 = ColumnDataSource(data=dict(x=[], y=[]))
source5_games = ColumnDataSource(data=dict(a=[], b=[], c=[], x=[], y=[]))
>>>>>>> 552c2a6edbbb082e5265ba40ce482531ae7fcf6d

"""
SETUP PLOTS
"""
<<<<<<< HEAD
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
=======
plot1 = figure(plot_height=600, plot_width=int(phi*600), title="Oh my Gauss",
              tools="save", x_range=[x1.min(), x1.max()], y_range=[0, phi*y1.max()])
plot1.line('x', 'y', source=source1_reference, line_width=3*1/phi, line_alpha=0.6, line_color="gray", line_dash='dashed', legend_label='Reference')
plot1.line('x', 'y', source=source1, line_width=3, legend_label="Your Gauss")

plot2 = figure(plot_height=700, plot_width=int(phi*700), title="Block Party",
              tools="save", x_range=[0, 11], y_range=[0, phi*counts2.max()])
plot2.vbar(x='x', top='y', source=source2, width=1/phi)

plot3 = figure(plot_height=500, plot_width=500, title="Scatter!", tools="", x_range=[-1, 1], y_range=[-1, 1], background_fill_color='#4169e1')
plot3.circle(x=0, y=0, fill_alpha=1, fill_color='#89cff0', radius=1)
plot3.scatter(x='x', y='y', source=source3, radius=0.0075, fill_color='#FF0000', fill_alpha=0.8, line_color=None)
plot3_below = figure(plot_height=500, plot_width=800, title="Running Average", tools="save", x_range=[0, 10], y_range=[3.0, 3.3])
plot3_below.circle('x', 'y', source=running_averages, size=10, fill_color='#FF0000', alpha=0.2, legend_label="Sample")
plot3_below.line('x', 'z', source=running_averages, line_width=2, line_dash='dashed', legend_label="Running Average")

plot4 = figure(plot_height=700, plot_width=int(phi*700), title="Dice Party",
              tools="save", x_range=[1, 13], y_range=[0, phi*max(counts4)])
plot4.vbar(x='label', top='value', source=source4, width=1/phi)

plot5 = figure(plot_height=500, plot_width=int(650*phi), title="Monte Carlo Casino",
              tools="save", background_fill_color='#FFFFFF', y_range=[0, 1000], x_range=[0, 1000])
plot5.line('x', 'y', source=source5, line_width=1, line_color="navy", legend_label="Bankroll")
plot5_sub = figure(plot_height=int(500/phi), plot_width=int(650*phi), title="Game Data",
              tools="save", background_fill_color='#FFFFFF', y_range=[0, 5000], x_range=[1, 2])
plot5_sub.line('x', 'y', source=source5_games, line_width=1, line_color="navy", legend_label="Average Max Bankroll")
plot5_sub.circle('x', 'b', source=source5_games, size=10, fill_color='#FF0000', alpha=0.2, legend_label="Sample")

"""
SETUP WIDGETS
"""
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

div2 = Div(text="""<p style="border:3px; border-style:solid; border-color:#FF0000; padding: 1em;">
                    Block Party demonstrates what occurs when there is a uniform distribution of events. 
                    Try changing the number of samples. 
                    You can hit the reset button to start over.</p>""",
                    width=300, height=125)
title2 = TextInput(title="Plot Title", value='Block Party')
num_sample = TextInput(title='Number of Samples', value='10')
sample = Button(label="Sample", button_type="success")
reset2 = Button(label="Reset", button_type="success")
columns2 = [TableColumn(field="x", title="Event", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="y", title="Count", formatter=NumberFormatter(text_align='center'))]
data_table2 = DataTable(source=source2,
                        columns=columns2,
                        index_position=None,
                        fit_columns=True,
                        width=275,
                        height=280,
                        selectable=False)
tot_sample2 = TextInput(title='Total Number of Samples', value=f"{counts2.sum()}")

title3 = TextInput(title="Plot Title", value='Scatter!')
num_sample3 = TextInput(title='Number of Samples', value='1000')
sample3 = Button(label="Sample", button_type="success")
reset3 = Button(label="Reset", button_type="success")
output3 = TextInput(title="Ratio", value=str(ratio))
div3 = Div(text="""<p style="border:3px; border-style:solid; border-color:#FF0000; padding: 0.5em;">
                    Scatter! is a game that we can play, similar to Buffon's Needle, that allows us to approximate &#960. 
                    This is done by randomly throwing darts at a dart board. We obtain our approximation by computing 4*In/Total.</p>""",
                    width=300, height=125)
columns3 = [TableColumn(field="w", title="#", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="x", title="In", formatter=NumberFormatter(text_align='center')), 
            TableColumn(field="y", title="Out", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="z", title="Total", formatter=NumberFormatter(text_align='center'))]
data_table3 = DataTable(source=ratio_report3,
                        columns=columns3,
                        index_position=None,
                        fit_columns=True,
                        width=275,
                        height=50,
                        selectable=False)

title4 = TextInput(title="Dice Party", value='Dice Party')
num_dice = TextInput(title="Number of Dice", value="2")
num_sides = TextInput(title="Number of Sides:", value="6")
roll4 = Button(label='Roll', button_type='success')
type_selection4 = Select(title="Select Frequency Type:", value="Totals", options=["Totals", "Cummulative Frequency", "Relative Frequency"])
reset4 = Button(label="Reset", button_type="success")
columns4 = [TableColumn(field="x", title="Roll", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="y", title="Count", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="z", title="Rel. Freq.", formatter=NumberFormatter(text_align='center', format='0.00 %'))]
data_table4 = DataTable(source=source4,
                        columns=columns4,
                        index_position=None,
                        fit_columns=True,
                        width=275,
                        height=280,
                        selectable=False)

div5 = Div(text="""<p style="border:3px; border-style:solid; border-color:#FF0000; padding: 0.5em;">
                    The Monte Carlo Casino (MCC) is a virtual casino. 
                    The purpose to MCC is to help you understand the long-term outcome of various betting strategies.</p>""",
                    width=300, height=100)
title5 = TextInput(title="Enter a title",
                   value='Monte Carlo Casino')
menu5 = [("Martingale", "Strategy 1"), ("Modified Martingale", "Strategy 2"), ("Complex", "Strategy 3")]
dropdown5 = Dropdown(label="Pick Your Strategy",
                     button_type="warning",
                     value='Strategy 1',
                     menu=menu5)
table_min = TextInput(title="Table Min ($)",
                      value='10')
table_max = TextInput(title="Table Max ($)",
                      value='500')
bankroll = TextInput(title="Bankroll",
                     value='100')
roll5 = Button(label="Roll", button_type="success")
reset5 = Button(label="Reset",
                button_type="primary")
columns5 = [TableColumn(field="a", title="Games Played", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="b", title="Max Bankroll", formatter=NumberFormatter(text_align='center')),
            TableColumn(field="c", title="Min Bankroll", formatter=NumberFormatter(text_align='center'))]
data_table5 =  DataTable(source=source5_games,
                        columns=columns5,
                        index_position=None,
                        fit_columns=True,
                        width=275,
                        height=plot5.plot_height,
                        selectable=False)

>>>>>>> 552c2a6edbbb082e5265ba40ce482531ae7fcf6d

"""
Set up callbacks
"""
<<<<<<< HEAD
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
=======
def update_title(attrname, old, new):
    plot1.title.text = title1.value
    plot2.title.text = title2.value
    plot3.title.text = title3.value
    plot4.title.text = title4.value
    plot5.title.text = title5.value


for t in [title1, title2, title3, title4, title5]:
    t.on_change('value', update_title)


def recompute_window_1():
    global source1
    Mu = mu.value
    Sigma = sigma.value
    try:
        sigma.value = str(float(Sigma))
        Sigma = float(Sigma)
        sigma.title = "Standard Deviation:"
    except:
        Sigma = 1.
        sigma.title = "Standard Deviation: ~Error~ (Please enter in a number)"
    try:
        mu.value = str(float(Mu))
        Mu = float(Mu)
        mu.title = "Average:"
    except:
        Mu = 0.
        mu.title = "Average: ~Error~ (Please enter in a number)"
    x1 = np.linspace(Mu - 6 * Sigma, Mu + 6 * Sigma, N)
    y1 = 1 / (Sigma * np.sqrt(2 * pi)) * np.exp(-0.5 * ((x1 - Mu) / Sigma) ** 2)
    plot1.x_range.start = min([x1.min(), u1.min()])
    plot1.x_range.end = max([x1.max(), u1.max()])
    plot1.y_range.end = phi*max([y1.max(), v1.max()])
    source1.data = dict(x=x1, y=y1)


def reset_window_1():
    global source1
    Sigma = 1.
    Mu = 0.
    x1 = np.linspace(Mu - 6 * Sigma, Mu + 6 * Sigma, N)
    y1 = 1 / (Sigma * np.sqrt(2 * pi)) * np.exp(-0.5 * ((x1 - Mu) / Sigma) ** 2)
    u1 = np.linspace(Mu - 6 * Sigma, Mu + 6 * Sigma, N)
    v1 = 1/(Sigma*np.sqrt(2*pi))*np.exp(-0.5*((u1-Mu)/Sigma)**2)
    plot1.title.text = "Oh My Gauss"
    plot1.x_range.start = x1.min()
    plot1.x_range.end = x1.max()
    plot1.y_range.start = 0.
    plot1.y_range.end = phi * y1.max()
    sigma.value = str(1.0)
    mu.value = str(0.0)
    sigma.title = "Standard Deviation:"
    mu.title = "Average:"
    checkbox1.active = [1, 0]
    source1.data = dict(x=x1, y=y1)
    source1_reference.data = dict(x=u1, y=v1)


recompute1.on_click(recompute_window_1)
reset1.on_click(reset_window_1)


def update_checkbox1(new):
    global source1, source1_reference
    val = checkbox1.active
    if 0 in val:
        u1 = np.linspace(0 - 6 * 1, 0 + 6 * 1, N)
        v1 = 1/(np.sqrt(2*pi))*np.exp(-0.5*(u1-0)**2)
        source1_reference.data = dict(x=u1, y=v1)
    else:
        u1, v1 = [], []
        source1_reference.data = dict(x=u1, y=v1)


checkbox1.on_click(update_checkbox1)

def update_window_2(attr, new, old):
    global source2
    # Sample = sample.button_type
    N = num_sample.value
    try:
        N = float(N)
        N = np.round(N, 0)
        if N > 0:
            N = int(N)
        else:
            N = int(abs(N))
        num_sample.title = 'Number of Samples'
        num_sample.value = str(N)
    except:
        N = 10
        N = int(N)
        num_sample.title = 'Number of Samples: (Please enter a positive integer)'
        num_sample.value = str(N)
    numbers = np.random.randint(low=1, high=11, size=N)
    count = source2.data['y']
    for x in numbers:
        count[x - 1] += 1
    unique, counts = source2.data['x'], count
    plot2.y_range.end = phi*counts.max()
    tot_sample2.value = f"{counts.sum()}"
    source2.data = dict(x=unique, y=counts)


num_sample.on_change('value', update_window_2)

def update_window_2_click():
    global source2
    # Sample = sample.button_type
    N = num_sample.value
    try:
        N = float(N)
        N = np.round(N, 0)
        if N > 0:
            N = int(N)
        else:
            N = int(abs(N))
        num_sample.title = 'Number of Samples'
        num_sample.value = str(N)
    except:
        N = 10
        N = int(N)
        num_sample.title = 'Number of Samples: (Please enter a positive integer)'
        num_sample.value = str(N)
    numbers = np.random.randint(low=1, high=11, size=N)
    count = source2.data['y']
    for x in numbers:
        count[x - 1] += 1
    unique, counts = source2.data['x'], count
    plot2.y_range.end = phi*counts.max()
    tot_sample2.value = f"{counts.sum()}"
    source2.data = dict(x=unique, y=counts)


sample.on_click(update_window_2_click)

def reset_window_2():
    x2 = np.random.randint(low=1, high=11, size=100)
    unique, counts = np.unique(x2, return_counts=True)
    plot2.y_range.end = phi * counts.max()
    num_sample.value = str(10)
    tot_sample2.value = f"{counts.sum()}"
    source2.data = dict(x=unique, y=counts)


reset2.on_click(reset_window_2)


def update_window_3():
    global source3, ratio_report3, running_averages
    # Sample = sample.button_type
    N = num_sample3.value
    try:
        N = float(N)
        N = np.round(N, 0)
        if N > 0:
            N = int(N)
        else:
            N = int(abs(N))
        num_sample.title = 'Number of Samples'
        num_sample.value = str(N)

        x3 = source3.data['x']
        x3_temp = np.random.uniform(low=-1, high=1, size=N)
        x3 = np.concatenate((x3, x3_temp))
        y3 = source3.data['y']
        y3_temp = np.random.uniform(low=-1, high=1, size=N)
        y3 = np.concatenate((y3, y3_temp))

        r = np.sqrt(x3_temp**2 + y3_temp**2)
        len_r = len(r)
        in_r = len([i for i in r if i <= 1])
        out_r = len([i for i in r if i > 1])
        
        I = ratio_report3.data['w']
        I[0] += 1
        IN = ratio_report3.data['x']
        IN[0] += in_r
        OUT = ratio_report3.data['y']
        OUT[0] += out_r
        TOTAL = ratio_report3.data['z']
        TOTAL[0] += int(abs(N))
        ratio = 4*IN[0]/TOTAL[0]
        output3.value = str(np.round(ratio, 8))

        source3.data = dict(x=x3_temp, y=y3_temp)
        ratio_report3.data = dict(w=I, x=IN, y=OUT, z=TOTAL)
    except:
        N = 1000
        N = int(N)
        num_sample.title = 'Number of Samples: (Please enter a positive integer)'
        num_sample.value = str(N)

    L = range(1, len(running_averages.data['y']) + 2)
    averages = running_averages.data['y'] + [4*in_r/len_r]
    running_average = running_averages.data['z'] + [ratio]
    if len(L) > plot3_below.x_range.end:
        plot3_below.x_range.end = len(L)
    running_averages.data = dict(x=L, y=averages, z=running_average)

sample3.on_click(update_window_3)


def reset_window_3():
    global source3, ratio_report3, running_averages
    # print("Reset")
    source3.data = dict(x=[], y=[])
    ratio_report3.data = dict(w=[0], x=[0], y=[0], z=[0], a=[0])
    running_averages.data = dict(x=[], y=[], z=[])
    plot3_below.x_range.end = 10
    output3.value = '0'
    num_sample3.value = '1000'


reset3.on_click(reset_window_3)


def update_window_4(attr, new, old):
    global source4, type_selection4, num_dice
    if num_dice.value is None:
        num_dice.value = '2'
    type_selection4.value = 'Totals'
    dice = int(num_dice.value)
    x4 = np.random.randint(low=1, high=7, size=1000)
    for _ in range(dice-1):
        x4 += np.random.randint(low=1, high=7, size=1000)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    L = range(dice, sides*dice+1)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    x4 = L
    y4 = np.zeros_like(L)
    for _ in range(1000):
        temp = np.random.randint(low=1, high=sides+1, size=dice).sum()
        y4[temp-dice] += 1

    source4.data['label'] = source4.data['x'] = x4
    source4.data['y'] = y4
    source4.data['z'] = source4.data['y'] / sum(source4.data['y'])
    if type_selection4.value == 'Totals':
        plot4.y_range.end = phi * max(y4)
        source4.data['value'] = source4.data['y']
    elif type_selection4.value == 'Cummulative Frequency':
        plot4.y_range.end = 1
        source4.data['value'] = np.cumsum(source4.data['y'] / sum(source4.data['y']))
    elif type_selection4.value == 'Relative Frequency':
        plot4.y_range.end = phi * max(source4.data['y'] / sum(source4.data['y']))
        source4.data['value'] = source4.data['y'] / sum(source4.data['y'])


num_dice.on_change('value', update_window_4)


def update_type_4(attr, new, old):
    global source4, type_selection4, num_dice, num_sides
    if num_dice.value is None:
        num_dice.value = '2'
    dice = int(num_dice.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    L = range(dice, sides*dice+1)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    x4 = L
    y4 = source4.data['y']
    source4.data['label'] = source4.data['x'] = x4
    source4.data['y'] = y4
    source4.data['z'] = source4.data['y'] / sum(source4.data['y'])
    if type_selection4.value == 'Totals':
        plot4.y_range.end = phi * max(y4)
        source4.data['value'] = source4.data['y']
    elif type_selection4.value == 'Cummulative Frequency':
        plot4.y_range.end = 1
        source4.data['value'] = np.cumsum(source4.data['y'] / sum(source4.data['y']))
    elif type_selection4.value == 'Relative Frequency':
        plot4.y_range.end = phi * max(source4.data['y'] / sum(source4.data['y']))
        source4.data['value'] = source4.data['y'] / sum(source4.data['y'])


type_selection4.on_change('value', update_type_4)


def update_roll4():
    global source4, num_dice, type_selection4, num_sides
    if num_dice.value is None:
        num_dice.value = '2'
    dice = int(num_dice.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    L = range(dice, sides*dice+1)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    x4 = L
    y4 = source4.data['y']
    for _ in range(1000):
        temp = np.random.randint(low=1, high=sides+1, size=dice).sum()
        y4[temp-dice] += 1
    source4.data['label'] = source4.data['x'] = x4
    source4.data['y'] = y4
    source4.data['z'] = source4.data['y'] /sum(source4.data['y'])
    if type_selection4.value == 'Totals':
        plot4.y_range.end = phi * max(y4)
        source4.data['value'] = source4.data['y']
    elif type_selection4.value == 'Cummulative Frequency':
        plot4.y_range.end = 1
        source4.data['value'] = np.cumsum(source4.data['y'] / sum(source4.data['y']))
    elif type_selection4.value == 'Relative Frequency':
        plot4.y_range.end = phi * max(source4.data['y'] / sum(source4.data['y']))
        source4.data['value'] = source4.data['y'] / sum(source4.data['y'])


roll4.on_click(update_roll4)


def change_num_sides_4(attr, old, new):
    global source4, type_selection4, num_dice, num_sides
    if num_dice.value is None:
        num_dice.value = '2'
    dice = int(num_dice.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides * dice + 1
    L = range(dice, sides * dice + 1)
    x4 = L
    y4 = np.zeros_like(L)
    for _ in range(1000):
        temp = np.random.randint(low=1, high=sides + 1, size=dice)
        temp = temp.sum()
        y4[temp - dice] += 1
    source4.data['label'] = source4.data['x'] = x4
    source4.data['y'] = y4
    source4.data['z'] = source4.data['y'] / sum(source4.data['y'])
    if type_selection4.value == 'Totals':
        plot4.y_range.end = phi * max(y4)
        source4.data['value'] = source4.data['y']
    elif type_selection4.value == 'Cummulative Frequency':
        plot4.y_range.end = 1
        source4.data['value'] = np.cumsum(source4.data['y'] / sum(source4.data['y']))
    elif type_selection4.value == 'Relative Frequency':
        plot4.y_range.end = phi * max(source4.data['y'] / sum(source4.data['y']))
        source4.data['value'] = source4.data['y'] / sum(source4.data['y'])


num_sides.on_change('value', change_num_sides_4)


def reset_window_4():
    global source4, num_dice, type_selection4, num_sides
    num_dice.value = '2'
    dice = 2
    num_sides.value = '6'
    sides = 6
    type_selection4.value = 'Totals'
    x4 = np.random.randint(low=1, high=sides + 1, size=1000)
    for _ in range(dice-1):
        x4 += np.random.randint(low=1, high=sides + 1, size=1000)
    x4, y4 = np.unique(x4, return_counts=True)
    z4 = y4/y4.sum()
    source4.data = dict(label=x4, value=y4, x=x4, y=y4, z=z4)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    plot4.y_range.start = 0
    plot4.y_range.end = phi*max(y4)


reset4.on_click(reset_window_4)


def update_window_5(attr, new, old):
    source5_games.data = dict(a=[], b=[], c=[], x=[], y=[])
    if dropdown5.value is None:
        dropdown5.value = 'Strategy 1'
    if int(table_max.value) <= int(table_min.value) or int(bankroll.value) <= 0:
        reset_window_5()
    play5()


for w in [table_min, table_max, bankroll, dropdown5]:
    w.on_change('value', update_window_5)


def play5():
    global source5, source5_games, table_min, table_max, bankroll
    def init_game(GAMES):
        return {"game": len(GAMES), "bankroll": int(bankroll.value), "P": 0, "NP": 0, "pass_bet": 0, "no_pass_bet": 0, "roll": 0}
    GAMES = [init_game([])]
    games_played = source5_games.data['a']
    maximum_bankroll = source5_games.data['b']
    minimum_bankroll = source5_games.data['c']
    avg_max_bankroll = source5_games.data['y']

    def win(g, b, pb, npb, r, GAMES):
        g['game'] = len(GAMES)
        g['roll'] = r
        g['P'] = 1
        g['NP'] = 0
        g['pass_bet'] = pb
        g['no_pass_bet'] = npb
        g['bankroll'] = b + pb - npb
        return g

    def loss(g, b, pb, npb, r, GAMES):
        g['game'] = len(GAMES)
        g['roll'] = r
        g['P'] = 0
        g['NP'] = 1
        g['pass_bet'] = pb
        g['no_pass_bet'] = npb
        g['bankroll'] = b - pb + npb
        return g

    def roll_dice():
        return np.random.randint(low=1, high=7, size=2).sum()

    if dropdown5.value is None:
        dropdown5.value = 'Strategy 1'
    if dropdown5.value == 'Strategy 1':
        game, bank, pass_bet, no_pass_bet, minimum_bet = init_game(GAMES), int(bankroll.value), int(table_min.value), 0, int(table_min.value)
    elif dropdown5.value == 'Strategy 2':
        game, bank, pass_bet, no_pass_bet, minimum_bet = init_game(GAMES), int(bankroll.value), int(table_min.value), 0, int(table_min.value)
    elif dropdown5.value == 'Strategy 3':
        game, bank, pass_bet, no_pass_bet, minimum_bet = init_game(GAMES), int(bankroll.value), int(table_min.value), int(table_min.value), 3*int(table_min.value)
    while 0 < minimum_bet <= bank:
        roll = roll_dice()
        if roll in (2, 3, 12):
            game = loss(game, bank, pass_bet, no_pass_bet, roll, GAMES)
        elif roll in (7, 11):
            game = win(game, bank, pass_bet, no_pass_bet, roll, GAMES)
        else:
            point = roll
            roll = roll_dice()
            while roll not in (point, 7):
                roll = roll_dice()
            if roll == 7:
                game = loss(game, bank, pass_bet, no_pass_bet, point, GAMES)
            elif roll == point:
                game = win(game, bank, pass_bet, no_pass_bet, point, GAMES)

        bank = game['bankroll']
        if dropdown5.value == 'Strategy 1':
            if game['NP'] == 1:
                if 2*pass_bet < bank and 2*pass_bet < int(table_max.value):
                    pass_bet = 2*pass_bet
                elif pass_bet < bank and pass_bet < int(table_max.value):
                    pass
                elif pass_bet > bank:
                    while pass_bet > bank:
                        pass_bet = pass_bet/2
                else:
                    pass_bet = int(table_min.value)
            elif game['P'] == 1:
                pass_bet = int(table_min.value)
        elif dropdown5.value == 'Strategy 2':
            if game['NP'] == 1:
                if 2*pass_bet + int(table_min.value) < bank and 2*pass_bet + int(table_min.value) < int(table_max.value):
                    pass_bet = 2*pass_bet + int(table_min.value)
                elif pass_bet < bank and pass_bet < int(table_max.value):
                    pass
                elif pass_bet > bank:
                    while pass_bet > bank:
                        if pass_bet % 2 == 0:
                            pass_bet = pass_bet/2
                        else:
                            pass_bet = (pass_bet - 5)/2
                else:
                    pass_bet = int(table_min.value)
            elif game['P'] == 1:
                pass_bet = int(table_min.value)
        elif dropdown5.value == 'Strategy 3':
            if game['NP'] == 1:
                if 2*pass_bet + int(table_min.value) < bank and 2*pass_bet + int(table_min.value) < int(table_max.value):
                    pass_bet = 2*pass_bet + int(table_min.value)
                elif pass_bet < bank and pass_bet < int(table_max.value):
                    pass
                elif pass_bet > bank:
                    while pass_bet > bank:
                        if pass_bet % 2 == 0:
                            pass_bet = pass_bet/2
                        else:
                            pass_bet = (pass_bet - 5)/2
                else:
                    pass_bet = int(table_min.value)
                no_pass_bet = int(table_min.value)
            elif game['P'] == 1:
                if 2*no_pass_bet + int(table_min.value) < bank and 2*no_pass_bet + int(table_min.value) < int(table_max.value):
                    no_pass_bet = 2*no_pass_bet + int(table_min.value)
                elif no_pass_bet < bank and no_pass_bet < int(table_max.value):
                    pass
                elif no_pass_bet > bank:
                    while no_pass_bet > bank:
                        if no_pass_bet % 2 == 0:
                            no_pass_bet = no_pass_bet/2
                        else:
                            no_pass_bet = (no_pass_bet - 5)/2
                else:
                    no_pass_bet = int(table_min.value)
                pass_bet = int(table_min.value)

        # pprint(game)
        GAMES.append(game)
        game = init_game(GAMES)

    df = pd.DataFrame.from_records(GAMES).fillna(0)
    plot5.x_range.start, plot5.x_range.end = 0, df['game'].max()
    plot5.y_range.start, plot5.y_range.end = 0, df['bankroll'].max()
    source5.data = dict(x=df['game'].tolist(), y=df['bankroll'].tolist())
    games_played.append(df['game'].max())
    minimum_bankroll.append(df['bankroll'].min())
    maximum_bankroll.append(df['bankroll'].max())
    num_games_played = range(1, len(games_played)+1)
    avg_max_bankroll.append(np.mean(maximum_bankroll))
    source5_games.data = dict(a=games_played, b=maximum_bankroll, c=minimum_bankroll, x=num_games_played, y=avg_max_bankroll)
    plot5_sub.x_range.start, plot5_sub.x_range.end = 1, num_games_played[-1] + 1
    plot5_sub.y_range.start, plot5_sub.y_range.end = 0, phi*max(maximum_bankroll)


roll5.on_click(play5)


def reset_window_5():
    global source5, dropdown5, table_min, table_max, bankroll
    dropdown5.value = 'Strategy 1'
    plot5.title.text = 'Monte Carlo Casino'
    table_max.value = '500'
    table_min.value = '10'
    bankroll.value = '100'
    title5.value = 'Monte Carlo Casino'
    source5.data = dict(x=[], y=[])
    source5_games.data = dict(a=[], b=[], c=[], x=[], y=[])
    plot5.x_range.start, plot5.x_range.end = 0, 1000
    plot5.y_range.start, plot5.y_range.end = 0, 1000
    plot5_sub.x_range.start, plot5_sub.x_range.end = 1, 2
    plot5_sub.y_range.start, plot5_sub.y_range.end = 0, phi*int(bankroll.value)


reset5.on_click(reset_window_5)


# Set up layouts and add to document
inputs1 = column(div1, title1, sigma, mu, recompute1, reset1, checkbox1)
inputs2 = column(div2, title2, num_sample, sample, reset2, data_table2, tot_sample2)
inputs3 = column(div3, title3, num_sample3, sample3, reset3, data_table3, output3)
inputs4 = column(title4, num_dice, type_selection4, num_sides, roll4, reset4, data_table4)
inputs5 = column(div5, title5, dropdown5, table_max, table_min, bankroll, roll5, reset5)
tab1 = row(inputs1, plot1, width=int(phi*400))
tab2 = row(inputs2, plot2, width=int(phi*400))
tab3 = row(inputs3, plot3, plot3_below, width=int(phi*400))
tab4 = row(inputs4, plot4, width=int(phi*400))
plot5_column = column(plot5, plot5_sub)
tab5 = row(inputs5, plot5_column, data_table5, width=int(phi*400))
tab1 = Panel(child=tab1, title="Like a Gauss")
tab2 = Panel(child=tab2, title="Block Party")
tab3 = Panel(child=tab3, title="Scatter!")
tab4 = Panel(child=tab4, title="Roll")
tab5 = Panel(child=tab5, title="Monte Carlo Casino")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

curdoc().title = "MCC MLC Dashboard"
>>>>>>> 552c2a6edbbb082e5265ba40ce482531ae7fcf6d
curdoc().theme = 'caliber'
curdoc().add_root(tabs)
