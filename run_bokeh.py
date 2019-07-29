from numpy.random import random
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, widgetbox
from bokeh.models import Button, ColumnDataSource
from bokeh.server.server import Server

"""
create and run a demo bokeh app on a cloud server
"""

def run(doc):

    fig = figure(title='random data', width=400, height=200, tools='pan,box_zoom,reset,save')

    source = ColumnDataSource(data={'x': [], 'y': []})
    fig.line('x', 'y', source=source)

    def click(n=100):
        source.data = {'x': range(n), 'y': random(n)}

    button = Button(label='update', button_type='success')
    button.on_click(click)

    layout = column(widgetbox(button), fig)
    doc.add_root(layout)
    click()

# configure and run bokeh server
kws = {'port': 5100, 'prefix': '/bokeh', 'allow_websocket_origin': ['165.227.26.215']}
server = Server(run, **kws)
server.start()
if __name__ == '__main__':
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()
