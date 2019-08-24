import numpy as np
from bokeh.io import curdoc, output_file
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, TextInput, Tabs, Panel, Button, DataTable, Div, CheckboxGroup
from bokeh.models.widgets import NumberFormatter, TableColumn, Dropdown, RadioButtonGroup, Select
from bokeh.plotting import figure
from pprint import pprint


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

x4 = np.random.randint(low=1, high=7, size=100)
unique4, counts4 = np.unique(x4, return_counts=True)
source4 = ColumnDataSource(data=dict(label=unique4, value=counts4, x=unique4, y=counts4, z=counts4/counts4.sum()))

source5 = ColumnDataSource(data=dict(x=[], y=[]))

"""
SETUP PLOTS
"""
plot1 = figure(plot_height=600, plot_width=int(phi*600), title="Oh my Gauss",
              tools="save", x_range=[x1.min(), x1.max()], y_range=[0, phi*y1.max()])
plot1.line('x', 'y', source=source1_reference, line_width=3*1/phi, line_alpha=0.6, line_color="gray", line_dash='dashed', legend='Reference')
plot1.line('x', 'y', source=source1, line_width=3, legend="Your Gauss")

plot2 = figure(plot_height=700, plot_width=int(phi*700), title="Block Party",
              tools="save", x_range=[0, 11], y_range=[0, phi*counts2.max()])
plot2.vbar(x='x', top='y', source=source2, width=1/phi)

plot3 = figure(plot_height=500, plot_width=500, title="Scatter!", tools="", x_range=[-1, 1], y_range=[-1, 1], background_fill_color='#4169e1')
plot3.circle(x=0, y=0, fill_alpha=1, fill_color='#89cff0', radius=1)
plot3.scatter(x='x', y='y', source=source3, radius=0.0075, fill_color='#FF0000', fill_alpha=0.8, line_color=None)
plot3_below = figure(plot_height=500, plot_width=800, title="Running Average", tools="save", x_range=[0, 10], y_range=[3.0, 3.3])
plot3_below.circle('x', 'y', source=running_averages, size=10, fill_color='#FF0000', alpha=0.2, legend="Sample")
plot3_below.line('x', 'z', source=running_averages, line_width=2, line_dash='dashed', legend="Running Average")

plot4 = figure(plot_height=700, plot_width=int(phi*700), title="Dice Party",
              tools="save", x_range=[0, 7], y_range=[0, phi*max(counts4)])
plot4.vbar(x='label', top='value', source=source4, width=0.8)

plot5 = figure(plot_height=750, plot_width=750, title="Wheel Party",
              tools="save", x_range=[-1, 1], y_range=[-1, 1], background_fill_color='#FFFFFF')

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
menu4 = [(str(x), str(x)) for x in range(1, 7)]
dropdown4 = Dropdown(label="Number of Dice", button_type="warning", menu=menu4)
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

title5 = TextInput(title="Enter a title",
                   value='Wheel Party')
menu5 = [("Strategy" + str(x), "Strategy" + str(x)) for x in range(1, 3)]
dropdown5 = Dropdown(label="Pick Your Strategy",
                     button_type="warning",
                     menu=menu5)
table_min = TextInput(title="Table Min ($)",
                      value='10')
table_max = TextInput(title="Table Max ($)",
                      value='500')
bankroll = TextInput(title="Bankroll",
                     value='1000')
reset5 = Button(label="Reset",
                button_type="success")


"""
Set up callbacks
"""
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
    global source1_reference
    val = checkbox1.active
    if 0 in val:
        u1 = np.linspace(0 - 6 * 1, 0 + 6 * 1, N)
        v1 = 1/(np.sqrt(2*pi))*np.exp(-0.5*(u1-0)**2)
        source1_reference.data = dict(x=u1, y=v1)
    else:
        source1_reference.data = dict(x=[], y=[])


checkbox1.on_click(update_checkbox1)


def update_window_2():
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


sample.on_click(update_window_2)


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

        r = np.sqrt(x3**2 + y3**2)
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
        TOTAL[0] += len_r 
        
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
    global source4, type_selection4, dropdown4
    # print('dropdown4 val:', dropdown4.value, "type:", type(dropdown4.value))
    if dropdown4.value is None:
        dropdown4.value = '1'
    type_selection4.value = 'Totals'
    dice = int(dropdown4.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    L = range(dice, sides*dice+1)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    x4 = L
    y4 = np.zeros_like(L)
    for _ in range(10000):
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


dropdown4.on_change('value', update_window_4)


def update_type_4(attr, new, old):
    global source4, type_selection4, dropdown4, num_sides
    # print('dropdown4 val:', dropdown4.value, "type:", type(dropdown4.value))
    if dropdown4.value is None:
        dropdown4.value = '1'
    dice = int(dropdown4.value)
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
    global source4, dropdown4, type_selection4, num_sides
    # print('dropdown4 val:', dropdown4.value, "type:", type(dropdown4.value))
    if dropdown4.value is None:
        dropdown4.value = '1'
    dice = int(dropdown4.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    L = range(dice, sides*dice+1)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides*dice + 1
    x4 = L
    y4 = source4.data['y']
    for _ in range(10000):
        temp = np.random.randint(low=1, high=sides+1, size=dice).sum()
        # print("Temp roll:", temp)
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
    global source4, type_selection4, dropdown4, num_sides
    if dropdown4.value is None:
        dropdown4.value = '1'
    dice = int(dropdown4.value)
    sides = abs(int(num_sides.value))
    num_sides.value = str(sides)
    plot4.x_range.start = dice - 1
    plot4.x_range.end = sides * dice + 1
    L = range(dice, sides * dice + 1)
    x4 = L
    y4 = np.zeros_like(L)
    for _ in range(10000):
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
    global source4, dropdown4, type_selection4, num_sides
    dropdown4.value = '1'
    num_sides.value = '6'
    type_selection4.value = 'Totals'
    x4 = np.random.randint(low=1, high=7, size=100)
    x4, y4 = np.unique(x4, return_counts=True)
    z4 = y4/y4.sum()
    source4.data = dict(label=x4, value=y4, x=x4, y=y4, z=z4)
    plot4.x_range.start = 0
    plot4.x_range.end = 7
    plot4.y_range.start = 0
    plot4.y_range.end = phi*max(y4)


reset4.on_click(reset_window_4)


def update_window_5(attr, new, old):
    if dropdown5.value is None:
        dropdown5.value = 'Strategy 1'


for w in [dropdown5, table_min, table_max, bankroll]:
    w.on_change('value', update_window_5)


def reset_window_5():
    global source5, dropdown5
    dropdown5.value = 'Strategy 1'
    plot5.title.text = 'Wheel Party'
    title5.value = 'Wheel Party'


reset5.on_click(reset_window_5)


# Set up layouts and add to document
inputs1 = column(div1, title1, sigma, mu, recompute1, reset1, checkbox1)
inputs2 = column(div2, title2, num_sample, sample, reset2, data_table2, tot_sample2)
inputs3 = column(div3, title3, num_sample3, sample3, reset3, data_table3, output3)
inputs4 = column(title4, dropdown4, type_selection4, num_sides, roll4, reset4, data_table4)
inputs5 = column(title5, dropdown5, reset5)
tab1 = row(inputs1, plot1, width=int(phi*400))
tab2 = row(inputs2, plot2, width=int(phi*400))
tab3 = row(inputs3, plot3, plot3_below, width=int(phi*400))
tab4 = row(inputs4, plot4, width=int(phi*400))
tab5 = row(inputs5, plot5, width=int(phi*400))
tab1 = Panel(child=tab1, title="Like a Gauss")
tab2 = Panel(child=tab2, title="Block Party")
tab3 = Panel(child=tab3, title="Scatter!")
tab4 = Panel(child=tab4, title="Dice Party")
tab5 = Panel(child=tab5, title="Wheel Party")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

curdoc().title = "MCC MLC Dashboard"
curdoc().theme = 'light_minimal'
curdoc().add_root(tabs)
