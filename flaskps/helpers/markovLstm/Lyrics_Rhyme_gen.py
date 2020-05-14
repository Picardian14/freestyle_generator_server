import pronouncing
import markovify
import re
import random
import numpy as np
import os


maxsyllables = 8
artist = "freestyle"
rap_file = "temporary_poem.txt"


def markov(text_file):
    ######
	read = open(text_file, "r", encoding='utf-8').read()
	text_model = markovify.NewlineText(read)
	return text_model

"""Determine number of syllables in line"""

def syllables(line):
	count = 0
	for word in line.split(" "):
		vowels = 'aeiouy'
# 		word = word.lower().strip("!@#$%^&*()_+-={}[];:,.<>/?")
		word = word.lower().strip(".:;?!")
		if word == '':
			continue
		if word[0] in vowels:
			count +=1
		for index in range(1,len(word)):
			if word[index] in vowels and word[index-1] not in vowels:
				count +=1
		if word.endswith('e'):
			count -= 1
		if word.endswith('le'):
			count+=1
		if count == 0:
			count +=1
	return count / maxsyllables

"""Make index of words that rhyme with your word"""

def rhymeindex(lyrics, weight_path):
	if "rhymes" in os.listdir(weight_path):
		print ("loading saved rhymes from local static")
		return open(f"{weight_path}/rhymes", "r",encoding='utf-8').read().split("\n")
	else:
		rhyme_master_list = []
		print ("Building list of rhymes:")
		for i in lyrics:
			word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()
			rhymeslist = pronouncing.rhymes(word)
			rhymeslistends = []      
			for i in rhymeslist:
				rhymeslistends.append(i[-2:])
			try:
				rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
			except Exception:
				rhymescheme = word[-2:]
			rhyme_master_list.append(rhymescheme)
		rhyme_master_list = list(set(rhyme_master_list))
		reverselist = [x[::-1] for x in rhyme_master_list]
		reverselist = sorted(reverselist)
		rhymelist = [x[::-1] for x in reverselist]
		print("List of Sorted 2-Letter Rhyme Ends:")
		print(rhymelist)
		f = open(str(artist) + ".rhymes", "w", encoding='utf-8')
		f.write("\n".join(rhymelist))
		f.close()
		return rhymelist

"""Make index of rhymes that you use"""

def rhyme(line, rhyme_list):
	word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
	rhymeslist = pronouncing.rhymes(word)
	rhymeslistends = []
	for i in rhymeslist:
		rhymeslistends.append(i[-2:])
	try:
		rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
	except Exception:
		rhymescheme = word[-2:]
	try:
		float_rhyme = rhyme_list.index(rhymescheme)
		float_rhyme = float_rhyme / float(len(rhyme_list))
		return float_rhyme
	except Exception:
		float_rhyme = None
		return float_rhyme

"""Separate each line of the input txt"""

def split_lyrics_file(text_file):
	text = open(text_file, encoding='utf-8').read()
	text = text.split("\n")
	while "" in text:
		text.remove("")
	return text

"""Generate lyrics"""

from tqdm import tqdm
from math import ceil

def get_last_word(bar):
	last_word = bar.split(" ")[-1]
	if last_word[-1] in "!.?,":
		last_word = last_word[:-1]
	return last_word

def generate_lyrics(text_model, text_file, wanted_word):
	bars = []
	last_words = []
	lyriclength = len(open(text_file,encoding='utf-8').read().split("\n"))
	markov_model = markov(text_file)
 
	rhymes_amount = 50 #lyriclength / 9
	pbar = tqdm(total=ceil(rhymes_amount),position=0,leave=True)
	while len(bars) < rhymes_amount:
		if (len(bars) % 32) == 0:
			bar = markov_model.make_sentence_with_start(wanted_word, strict=False, tries=100)
		else:
			bar = markov_model.make_sentence(max_overlap_ratio = .49, tries=100)
		if type(bar) != type(None) and syllables(bar) < 1:
			last_word = get_last_word(bar)
			if bar not in bars and last_words.count(last_word) < 3:
				bars.append(bar)
				last_words.append(last_word)
				pbar.update()
	pbar.close()
	return bars

"""Build dataset"""

def build_dataset(lines, rhyme_list):
	dataset = []
	line_list = []
	for line in lines:
		line_list = [line, syllables(line), rhyme(line, rhyme_list)]
		dataset.append(line_list)
	x_data = []
	y_data = []
	for i in range(len(dataset) - 3):
		line1 = dataset[i    ][1:]
		line2 = dataset[i + 1][1:]
		line3 = dataset[i + 2][1:]
		line4 = dataset[i + 3][1:]
		x = [line1[0], line1[1], line2[0], line2[1]]
		x = np.array(x)
		x = x.reshape(2,2)
		x_data.append(x)
		y = [line3[0], line3[1], line4[0], line4[1]]
		y = np.array(y)
		y = y.reshape(2,2)
		y_data.append(y)
	x_data = np.array(x_data)
	y_data = np.array(y_data)
	return x_data, y_data

"""Compose verse"""

def compose_rap(lines, rhyme_list, lyrics_file, model):
	rap_vectors = []
	human_lyrics = split_lyrics_file(lyrics_file)
	initial_index = random.choice(range(len(human_lyrics) - 1))
	initial_lines = human_lyrics[initial_index:initial_index + 2]
	starting_input = []
	for line in initial_lines:
		starting_input.append([syllables(line), rhyme(line, rhyme_list)])
	starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(1, 2, 2))
	rap_vectors.append(starting_vectors)
	for i in range(50):
		rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).flatten().reshape(1, 2, 2)))
	return rap_vectors

"""Compose verse (part 2)"""

#Auxiliary methods

def last_word_compare(rap, line2):
	penalty = 0
	for line1 in rap:
		word1 = line1.split(" ")[-1]
		word2 = line2.split(" ")[-1]
		while word1[-1] in "?!,. ":
			word1 = word1[:-1]
		while word2[-1] in "?!,. ":
			word2 = word2[:-1]
		if word1 == word2:
			penalty += 0.2
	return penalty

def calculate_score(vector_half, syllables, rhyme, penalty, rhyme_list):
	desired_syllables = vector_half[0]
	desired_rhyme = vector_half[1]
	desired_syllables = desired_syllables * maxsyllables
	desired_rhyme = desired_rhyme * len(rhyme_list)	
	score = 1.0 - abs(float(desired_syllables) - float(syllables)) + abs(float(desired_rhyme) - float(rhyme)) - penalty
	return score



def vectors_into_song(vectors, generated_lyrics, rhyme_list, wanted_word):
	print()	
	print("Escribiendo un verso con:", wanted_word)
	print()
	dataset = []
	for line in generated_lyrics:
		line_list = [line, syllables(line), rhyme(line, rhyme_list)]
		dataset.append(line_list)
	rap = []
	vector_halves = []
	for vector in vectors:
		vector_halves.append(list(vector[0][0])) 
		vector_halves.append(list(vector[0][1]))
	for vector in vector_halves:
		scorelist = []
		for item in dataset:
			line = item[0]
			if len(rap) != 0:
				penalty = last_word_compare(rap, line)
			else:
				penalty = 0
			total_score = calculate_score(vector, item[1], item[2], penalty, rhyme_list)
			score_entry = [line, total_score]
			scorelist.append(score_entry)
		fixed_score_list = [0]
		for score in scorelist:
			fixed_score_list.append(float(score[1]))
		max_score = max(fixed_score_list)
		for item in scorelist:
			if item[1] == max_score:
				rap.append(item[0])
				#print (str(item[0]))
				for i in dataset:
					if item[0] == i[0]:
						dataset.remove(i)
						break
				break
	f = open(rap_file, "w", encoding='utf-8')
	bar_index=0        
	final_rap = []    
	while bar_index < len(rap):
		first = rap[bar_index].split(" ")[0]
		if first == wanted_word:
			for bar_index in range(bar_index, bar_index+4):				
				print(rap[bar_index])                
				final_rap.append(rap[bar_index])
				f.write(rap[bar_index])
				f.write("\n")
			f.write("\n")	
			print(final_rap)
			return final_rap
		else:
			bar_index+=1
	return final_rap
