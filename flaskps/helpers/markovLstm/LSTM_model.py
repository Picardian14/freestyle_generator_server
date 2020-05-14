import keras
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers.core import Dense

import os
import numpy as np

from flaskps.helpers.markovLstm.Lyrics_Rhyme_gen import split_lyrics_file, rhymeindex, build_dataset
text_file = './flaskps/static/Freestyle/merge_with_wos.txt'
artist = "freestyle"
def create_network(depth, weight_path):
	model = Sequential()
	model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))
	for i in range(depth):
		model.add(LSTM(8, return_sequences=True))
	model.add(LSTM(2, return_sequences=True))
	model.summary()
	model.compile(optimizer='rmsprop',
              loss='mse')	
	model.load_weights(weight_path)
	print("loading saved network: " + weight_path) 
	return model

def train(x_data, y_data, model):
	model.fit(np.array(x_data), np.array(y_data),
			  batch_size=2,
			  epochs=5,
			  verbose=1)
	model.save_weights(artist + ".rap")

"""Function to train the model"""

def train_model(model, text_model, depth):
	bars = split_lyrics_file(text_file)
	rhyme_list = rhymeindex(bars)
	x_data, y_data = build_dataset(bars, rhyme_list)
	train(x_data, y_data, model)