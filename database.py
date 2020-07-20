#database.py
import pickle, os
from time import strftime
from pprint import pprint


class database:
	def __init__(self, name):
		self.name = name
		cwd = os.getcwd()
		path = os.path.join(cwd,'data')
		try:
			with open(f'{path}/{self.name}.pkl', 'rb') as f:
				data = pickle.load(f)
			print('Data loaded')
		except:
			data = {}
			if os.path.isdir(path) == False:
				os.makedirs(path)
			with open(f'{path}/{self.name}.pkl', 'wb') as f:
				pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
			print('Data created')
			data['History'] = {}
		self.data = data
	def save(self):
		cwd = os.getcwd()
		path = os.path.join(cwd,'data')
		with open(f'{path}/{self.name}.pkl', 'wb') as f:
			pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
		print("Data saved\n")
	def update(self, timestamp, data):
		date = timestamp.date()
		timestamp = str(timestamp.time()).split('.')[0]
		timestamp = timestamp.split(':')[0] + ':' + timestamp.split(':')[1]
		if date in self.data.keys():
			if timestamp in self.data[date].keys():
				# print("Data exists")
				pass
			else:
				self.data[date][timestamp] = data
				# print('New timestamp inserted into database')
		elif date not in self.data.keys():
			self.data[date] = {timestamp: data}
			# print("New date inserted into database")
		# pprint(self.data[date].keys())
	def history(self, timestamp, data):
		date = timestamp.date()
		timestamp = str(timestamp.time()).split('.')[0]
		timestamp = timestamp.split(':')[0] + ':' + timestamp.split(':')[1] 
		# print("Date:", date, " Time:", timestamp)
		candles = {}
		for datum in data:
			candle = {'Name':datum['name'],
					  'Price':datum['current_price'],
					  'Rank': datum['market_cap_rank'],
					  'Circulating Supply': datum['circulating_supply'],
					  'Total Supply': datum['total_supply'],
					  'Total Volume': datum['total_volume'],
					  'Market Cap':datum['market_cap']}
			candles[datum['symbol'].upper()] = candle
		try:
			self.data['History'][date][timestamp] = candles
			# print('Inserted new history candles.')
		except:
			self.data['History'][date] = {timestamp:candles}
			# print('Created new history date data')
		# pprint(self.data['History'][date])

if __name__ == '__main__':
	from pycoingecko import CoinGeckoAPI
	from datetime import datetime
	import time
	cg = CoinGeckoAPI()
	db = database('test')
	while True:
		print(cg.ping())
		if cg.ping() is not None:
			coins_list = cg.get_coins_markets(vs_currency='usd')
			now = datetime.now()
			db.update(now, coins_list)
			db.history(now, coins_list)
			db.save()
		time.sleep(60)
