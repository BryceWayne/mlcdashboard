from flask import Flask, render_template

from bokeh.embed import server_document, components
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Select
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop
from bokeh.palettes import inferno
from bokeh.transform import factor_cmap

# from bokeh.embed import components

from bokeh.io import curdoc

import pymongo 
import pandas as pd
import numpy as np

app = Flask(__name__)


def modify_doc(doc):

    query_dict = {'game_pk':531036}
    fields_return_dict = { "_id": 0, "pfx_x": 1, "pfx_z": 1, "pitch_name":1, "plate_x":1, "plate_z": 1,
    "release_pos_x":1, "release_pos_z":1, "pitcher":1}

    client = pymongo.MongoClient('localhost', 27017)
    db = client['MLB'] #select database
    db_collection = db['pitch_level'] #select the collection within the databse
    test_df = pd.DataFrame(list(db_collection.find(query_dict,fields_return_dict))) #convert entire collection to pandas DataFrame
    client.close()

    test_df = test_df.dropna()

    name_array = test_df.pitcher.values.tolist()


    def unique(list1):

        # intilize a null list 
        unique_list = [] 
          
        # traverse for all elements 
        for x in list1: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 

        return unique_list

    name_list = unique(name_array)

    fields_return_dict = { "_id": 0, "MLBID": 1, "MLBNAME": 1}

    client = pymongo.MongoClient('localhost', 27017)
    db = client['MLB'] #select database
    db_collection = db['PLAYER_ID_MAP'] #select the collection within the databse
    name_df = pd.DataFrame(list(db_collection.find({"MLBID": {'$in': name_list}},fields_return_dict))) #convert entire collection to pandas DataFrame
    client.close()

    name_df.rename(columns={'MLBID':'pitcher'}, inplace=True)

    df_merge = pd.merge(test_df, name_df, on='pitcher')

    source = ColumnDataSource(df_merge)

    plot1 = figure(plot_height=400, sizing_mode='scale_width')

    color_factors = [str(x) for x in sorted(df_merge.pitch_name.unique())]

    color_map = factor_cmap('pitch_name',
        palette=inferno(len(df_merge.pitch_name.unique())),
        factors=color_factors)

    r = plot1.circle(x='pfx_x', y='pfx_z',
        source=source,
        size=10,
        color=color_map,
        legend="pitch_name")

    plot1.title.text = 'Pitch Movement'
    plot1.xaxis.axis_label = 'Horizontal Movement (feet)'
    plot1.yaxis.axis_label = 'Vertical Movement (feet)'

    hover = HoverTool()
    hover.tooltips=[
    ('Horizontal Movement', '@pfx_x'),
    ('Vertical Movement', '@pfx_z'),
    ('Pitch Name', '@pitch_name'),
    ('Pitcher', '@MLBNAME')
    ]

    plot1.add_tools(hover)

    plot1.legend.location = "top_left"

    pitch = df_merge['MLBNAME'].unique()
    pitchers = pitch.tolist()
    p = [str(i) for i in pitchers]
    p.append("All")

    select = Select(title="Option:", value="All", options=p)

    def pitcher_select(attr, old, new):
        if select.value=="All":
            df_filter = df_merge.copy()
        else:
            df_filter = df_merge[df_merge['MLBNAME']==select.value]
        source1 = ColumnDataSource(df_filter)
        r.data_source.data = source1.data
        r2.data_source.data = source1.data
        r3.data_source.data = source1.data

    select.on_change("value", pitcher_select)

    plot2 = figure(plot_height=400, sizing_mode='scale_width')

    r2 = plot2.circle(x='plate_x', y='plate_z',
        source=source,
        size=10,
        color=color_map,
        legend="pitch_name")

    plot2.title.text = 'Pitch Location As It Crosses Home Plate'
    plot2.xaxis.axis_label = 'Horizontal position (feet)'
    plot2.yaxis.axis_label = 'Vertical position (feet)'

    hover2 = HoverTool()
    hover2.tooltips=[
    ('Horizontal Position', '@plate_x'),
    ('Vertical Position', '@plate_z'),
    ('Pitch Name', '@pitch_name'),
    ('Pitcher', '@MLBNAME')
    ]

    plot2.add_tools(hover2)

    plot2.legend.location = "top_left"


    plot3 = figure(plot_height=400, sizing_mode='scale_width')

    r3 = plot3.circle(x='release_pos_x', y='release_pos_z',
        source=source,
        size=10,
        color=color_map,
        legend="pitch_name")

    plot3.title.text = 'Release Location of Pitch From Catchers Perspective'
    plot3.xaxis.axis_label = 'Horizontal position (feet)'
    plot3.yaxis.axis_label = 'Vertical position (feet)'

    hover3 = HoverTool()
    hover3.tooltips=[
    ('Horizontal Position', '@release_pos_x'),
    ('Vertical Position', '@release_pos_z'),
    ('Pitch Name', '@pitch_name'),
    ('Pitcher', '@MLBNAME')
    ]

    plot3.add_tools(hover3)

    plot3.legend.location = "top_left"
    
    grid = gridplot([[plot1, plot2], [None, plot3]])
    column1 = column(select, grid, sizing_mode = 'scale_width')

    doc.add_root(column1)

    doc.theme = Theme(filename="theme.yaml")


@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return render_template("embed.html", script=script, template="Flask", relative_urls=False)


def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
    server.start()
    server.io_loop.start()

from threading import Thread
Thread(target=bk_worker).start()

if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(port=8000)
