#main.py
import pickle, os, time
from database import database
from pprint import pprint
import numpy as np
import pandas as pd
from bokeh.embed import server_document
from flask import Flask, request, render_template
import subprocess
from pycoingecko import CoinGeckoAPI


# app = Flask(__name__)

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select, ColumnDataSource, TableColumn, DataTable
from bokeh.models.widgets import Tabs, PreText, Panel

# stats = PreText(text='', width=500)
cg = CoinGeckoAPI()
db = database('test')
intro = Select(title="Cryptocurrency", value="All",
               options=['All'])


# @app.route('/')
# def index():
#     script = server_document("http://localhost:5006/dashboard")
#     print(f"Script: {script}")
#     return render_template('index.html', plot_script=script)

# if __name__ == '__main__':
#     app.run(port='8080')




# print("Ping:", cg.ping())


# db.create()
# coins_list = cg.get_coins_list()
# db.ids(coins_list)
# coins_list = cg.get_coins_markets(vs_currency='usd')
# db.markets(coins_list)
# print(db.data['ids'])
# db.save()
# print(cg.get_coins_markets(vs_currency='usd'))
# print(cg.get_price(ids='verge', vs_currencies='usd'))
# print("Supported vs. Curr.:", cg.get_supported_vs_currencies())
# print("Exchanges List:", cg.get_exchanges_list())

# pprint(db.data['markets'])
def clean_entry(coin):
	data, columns = {}, []
	data['market_cap_rank'] = [coin['market_cap_rank']]
	columns.append(TableColumn(field='market_cap_rank', title='market_cap_rank'))
	data['symbol'] = [coin['symbol'].upper()]
	columns.append(TableColumn(field='symbol', title='Ticker'))
	data['name'] = [coin['name']]
	columns.append(TableColumn(field='name', title='Name'))
	data['current_price'] = [coin['current_price']]
	columns.append(TableColumn(field='current_price', title='Current Price ($)'))
	data['ath'] = [coin['ath']]
	columns.append(TableColumn(field='ath', title='All Time High'))
	data['atl'] = [coin['atl']]
	columns.append(TableColumn(field='atl', title='All Time Low'))
	return data, columns


coins_list = cg.get_coins_markets(vs_currency='usd')
coin = coins_list[0]
pprint(coin)
data, columns = clean_entry(coin)
df = []
for market in coins_list:
	df.append(market)
df = pd.DataFrame(df)
df = df[['market_cap_rank', 'ath', 'atl', 'name', 'symbol', 'current_price']]
df.sort_values('market_cap_rank')
df['symbol'] = df['symbol'].str.upper()
source = ColumnDataSource(data=df)
data_table = DataTable(source=source, columns=columns, width=800, height=800, index_position=None, fit_columns=True)
"""
Set up callbacks
"""
def select_crypto(attr, old, new):
	coin = intro.value
	if coin == 'All':
		df = []
		for market in db.data['markets']:
			df.append(market['info'])
		df = pd.DataFrame(df)
		df = df[['market_cap_rank', 'ath', 'atl', 'name', 'symbol', 'current_price']]
		df.sort_values('market_cap_rank')
		df['symbol'] = df['symbol'].str.upper()
	else:
		coin = db.coin_info(coin)
		data, _ = clean_entry(coin)
		df = pd.DataFrame(data)
	source.data = df


intro.on_change('value', select_crypto)
INTRO = row(intro, data_table)
INTRO = Panel(child=INTRO, title="Market Info")
tabs = Tabs(tabs=[INTRO])
curdoc().title = "CMC Dashboard"
curdoc().theme = 'caliber'
curdoc().add_root(tabs)
