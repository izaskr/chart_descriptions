#!/usr/bin/python3

"""
Label the bar chart summaries using the information provided in the json that contains plotting information from FigureQA.

Parse each summary, check if names of bars occur, label them (e.g. x_axis_label_highest_value).
Similarly, their absolute heights (e.g. y_axis_least_value_val) and relations between them (e.g. y_axis_inferred_value_mul_v1=highest_v2=least).

Write the result into a file: 
unlabeled tokens - one token per line,
labeled single tokens - token tab label
labeled multi-token units - tokens separated with whitespace tab label

These files will be manually reviewed by Iza and Emeka.  
"""

import json
import numpy as np
import argparse
import sys
from itertools import combinations
#import mord
from sklearn.linear_model import LinearRegression
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH, NORM
import num2word
import re
from nltk.tokenize import word_tokenize
from learn_labels import get_discourse_tokens

parser = argparse.ArgumentParser()


parser.add_argument("-write", required=True, help="write y or n to decide if the labeled data should be written into a file")
#parser.add_argument("-summaries", required=True, help="file with chart summaries; unlabeled and unpreprocessed")

args = vars(parser.parse_args())

#summary_file = args["summaries"]
write_yn = args["write"]
write_file = False
if write_yn in {"y", "yes"}: write_file = True
if write_yn not in {"y","n", "yes", "no"}: sys.exit("Use 'y' or 'n' as value for -write")
if write_yn in {"n", "no"}: sys.exit("Parser currently supports only writing mode")

version_dir = "/home/iza/chart_descriptions/data_batch3/"
description_files = ["batch1/akef_inc_closing_stock_prices_1.txt",
"batch1/average_time_spent_on_social_media_1.txt",
"batch1/fatal_injuries_at_pula_steel_factory_1.txt",
"batch1/gender_pay_gap_2.txt",
"batch1/how_young_people_spend_their_evenings_1.txt",
"batch1/minority_representation_in_libya_parliament_1.txt",
"batch1/what_causes_obesity_2.txt",
"batch1/women_representation_in_different_uni_departments_2.txt",
"batch2/akef_inc_closing_stock_prices_2.txt",
"batch2/average_time_spent_on_social_media_2.txt",
"batch2/fatal_injuries_at_pula_steel_factory_2.txt",
"batch2/median_salary_of_women_2.txt",
"batch2/minority_representation_in_libya_parliament_2.txt",
"batch2/money_spent_on_HE_2.txt",
"batch2/what_students_study_at_lagos_uni.txt",
"batch2/women_representation_in_different_sectors_2.txt"]


# descriptions_files_json is a dict, where the keys are files names (as given by Rudy) with descriptions, and the values are tuples - its first element is the name of the json file with raw plot data, its second element is the title of the plot (it's unique for each plot)


# TODO new descriptions_files_json indices correspond to image indices assigned by FigureQA
anno_paths = ("run2_jsons/","/annotations.json")
#jsons = ["run2_jsons/train1/annotations.json", "run2_jsons/val1/annotations.json", "run2_jsons/val2/annotations.json"]
jsons = {"train1":"run2_jsons/train1/annotations.json", "val1":"run2_jsons/val1/annotations.json", "val2":"run2_jsons/val2/annotations.json"}

descriptions_files_json = {"batch1/akef_inc_closing_stock_prices_1.txt":("train1", 2),
"batch1/average_time_spent_on_social_media_1.txt":("train1", 3),
"batch1/fatal_injuries_at_pula_steel_factory_1.txt":("train1",4),
"batch1/gender_pay_gap_2.txt":("train1",9),
"batch1/how_young_people_spend_their_evenings_1.txt":("train1",6),
"batch1/minority_representation_in_libya_parliament_1.txt":("train1",5),
"batch1/what_causes_obesity_2.txt":("train1",8),
"batch1/women_representation_in_different_uni_departments_2.txt":("train1",0),
"batch2/akef_inc_closing_stock_prices_2.txt":("val2",2),
"batch2/average_time_spent_on_social_media_2.txt":("val2",1),
"batch2/fatal_injuries_at_pula_steel_factory_2.txt":("val1", 1),
"batch2/median_salary_of_women_2.txt":("val2",0),
"batch2/minority_representation_in_libya_parliament_2.txt":("train1",7),
"batch2/money_spent_on_HE_2.txt":("val1",2),
"batch2/what_students_study_at_lagos_uni.txt":("val1",0),
"batch2/women_representation_in_different_sectors_2.txt":("train1",1)}

description_files = ["new_cameras1.png.txt",
"new_cameras2_g3.png.txt",
"new_citySA1_g2.png.txt",
"new_citySA2_g3.png.txt",
"new_glaciers1_g1.png.txt",
"new_glaciers2_g3.png.txt",
"new_minority1_g3.png.txt",
"new_minority2_g2.png.txt",
"new_moneyHE1_g1.png.txt",
"new_paygap1_g2.png.txt",
"new_paygap2_g3.png.txt",
"new_paygap3_g1.png.txt",
"new_quiz1_g2.png.txt",
"new_quiz2_g1.png.txt",
"new_socmedia1_g3.png.txt",
"new_socmedia2_g2.png.txt",
"new_stock_price1_g2.png.txt",
"new_womendep1_g2.png.txt",
"new_womendep2_g1.png.txt",
"new_womendep3_g3.png.txt",
"new_womensec1_g1.png.txt"]

map_file = "/home/iza/chart_descriptions/data_batch3/README.txt"



# v["image_index"]

# just the tokenizer, rule for ,000 be treated as a single token, not split
nlp = English()  # just the language with no model

#nlp.add_pipe(tokenizer)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

tokenizer = Tokenizer(nlp.vocab)
case_000 = [{ORTH: ",000"} ]#, {ORTH: "BB", NORM: ", 000"}]
tokenizer.add_special_case(",000", case_000)
case_ds = [{ORTH: "°C"}]
tokenizer.add_special_case("°C", case_ds)


nlp_en_core = spacy.load("en_core_web_sm")
# note that all_stopwords includes also spelled numerals
all_stopwords = nlp_en_core.Defaults.stop_words

def get_info_from_map(mp_file="/home/iza/chart_descriptions/data_batch3/README.txt"):
	# open the map file, return a list of tuples (filename, index_in_annotations_json)
	maps = []
	with open(mp_file, "r") as f:
		for line in f:
			if not line.startswith("#") and len(line.split()) > 1:
				line = line.split(" : ")
				#print(line)
				maps.append((line[0], int(line[1])))
	return maps



def get_x_y(filename):
	"""
	Function for extracting only the information needed to analyze the data, return a dict for every plot,
	including the values of x axis, y axis, plot type, plot title

	:param filename: name of annotations.json file, which follows a certain format
	:type filename: str
	"""
	with open(filename) as f:
		data = json.load(f) # a list of dictionaries

	xy = {}
	
	for i,k in enumerate(data):
		x = k["models"][0]["x"]
		y = k["models"][0]["y"]
		title = k["general_figure_info"]["title"]["text"]
		# x_order_info = k["models"][0]["x_order_info"] TODO add to json files
		x_order_info = []
		plot_type = k["type"] # "vbar_categorical"
		y_axis_unit_name = k["general_figure_info"]["y_axis"]["label"]["text"] # label of y from the plot
		x_axis_label_name = k["general_figure_info"]["x_axis"]["label"]["text"]
		x_major_ticks = k["general_figure_info"]["x_axis"]["major_ticks"]["values"][:-(len(x))] # why double
		len_yt = int(len(k["general_figure_info"]["y_axis"]["major_ticks"]["values"]) / 2)
		y_major_ticks = k["general_figure_info"]["y_axis"]["major_ticks"]["values"][:-len_yt]
		image_index = k["image_index"] # int

		# xy[i+1] before
		xy[i] = {"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, "x_axis_label_name": x_axis_label_name, "x_major_ticks":x_major_ticks,"y_major_ticks":y_major_ticks, "image_index":image_index}

	return xy



def get_stat_info(data):
	"""
	Calculate simple statistical information about the data for each plot
	Return keys and values to be added to the according dictionary belonging of each plot

	:param data: information about each plot individually
	:type data: dict
	"""
	
	# what labels in the annotated descriptions can be mapped to directly from the calculations?
	x, y = data["x"], data["y"]

	y_max_idx = np.argmax(y)
	x_max, y_max = x[y_max_idx], y[y_max_idx]

	y_min_idx = np.argmin(y)
	x_min, y_min = x[y_min_idx], y[y_min_idx]
	
	# mean
	mean = np.mean(data["y"])

	# sorted axes, according to descending y values
	y_sorted, x_sorted = (list(t) for t in zip(*sorted(zip(y,x),reverse=True)))
	#print(x_sorted,"\n",y_sorted)
	labels = {}

	# labels for ordered x and y
	anno_size_x = ["<x_axis_label_highest_value>", "<x_axis_label_Scnd_highest_value>", "<x_axis_label_3rd_highest_value>", "<x_axis_label_4th_highest_value>","<x_axis_label_5th_highest_value>","<x_axis_label_least_value>"]

	# <y_axis_4th_highest_val>
	anno_size_y = ["<y_axis_highest_value_val>", "<y_axis_Scnd_highest_val>", "<y_axis_3rd_highest_val>", "<y_axis_4th_highest_val>","<y_axis_5th_highest_val>", "<y_axis_least_value_val>"]

	n = len(x) - 2 # -1 for max, -1 for min
	anno_size_x2 = anno_size_x[:n+1]
	anno_size_x2.append(anno_size_x[-1])
	anno_size_y2 = anno_size_y[:n+1]
	anno_size_y2.append(anno_size_y[-1])

	label_name_pairs_x = {e[0]:e[1] for e in zip(anno_size_x2,x_sorted)}
	label_value_pairs_y = {e[0]:e[1] for e in zip(anno_size_y2,y_sorted)}

	# label <x_axis_label_count> is the number of categories on the x axis
	label_count = len(x)
	misc = {"<x_axis_label_count>":label_count}
	
	print(data["title"])

	multi = {}
	addi = {}

	for e,v in label_value_pairs_y.items():
		for e2, v2 in label_value_pairs_y.items():
			if e != e2:
				n1, n2 = e.split("_")[2], e2.split("_")[2]
				#print(n1,n2)
				#print(v, v2)
				lm = "<y_axis_inferred_value_mul_v1="+n1+"_v2="+n2+">" # label multiplication
				vm = round(v / v2,2) # value multiplication
				multi[lm] = vm
				la = "<y_axis_inferred_value_add_v1="+n1+"_v2="+n2+">" # label addition
				va = round(abs(v - v2),2) # only positive!
				addi[la] = va
				
	#print("\t",multi)
	#print("\t",addi)

	results = {"label_name_pairs_x":label_name_pairs_x, "label_value_pairs_y":label_value_pairs_y, "multi":multi, "addi": addi}
	results_basic = {** label_name_pairs_x, ** label_value_pairs_y}
	results_cal = {**multi, ** addi}
	results_basic["<y_axis>"] = data["y_axis_unit_name"]
	#print(len(results_all), sum([len(v) for k,v in results.items()]))

	#return data["title"], data["x_order_info"],differences, label_name_pairs_x,label_value_pairs_y, misc
	#return results
	other_info = {"title": data["title"], "<x_axis_label_count>":label_count, "y_axis_name":data["y_axis_unit_name"], "x_axis_name": data["x_axis_label_name"]}
	return results_basic, results_cal, other_info

def discourse_labels():
	""" dlabel_word is a dict, discourse labels as keys, vocab set as value
	word_dlabel is a dict, word as key, dict counter of labels and their counts as value
	"""
	#xml_file = "corpora_v02/chart_summaries_b01.xml"
	#dlabel_word, word_dlabel = get_discourse_tokens(xml_file)

	# b1b2 includes data from the first and second collection batch
	xml_file_b1b2 = "corpora_v02/all_descriptions/chart_summaries_b01_toktest2.xml"
	dlabel_word, word_dlabel = get_discourse_tokens(xml_file_b1b2)

	return dlabel_word, word_dlabel


def find_closest(target, numbers, numbers_labels):
	""" in a list of numbers, it finds the number closest to the target number and returns the number's index"""
	exact_approx = {"<y_axis_highest_value_val>":"<y_axis_inferred_highest_value_approx>", "<y_axis_Scnd_highest_val>":"<y_axis_inferred_Scnd_highest_value_approx>", "<y_axis_3rd_highest_val>":"<y_axis_inferred_3rd_highest_value_approx>", "<y_axis_4th_highest_val>":"<y_axis_inferred_4th_highest_value_approx>",
"<y_axis_5th_highest_val>":"<y_axis_inferred_5th_highest_value_approx>", "<y_axis_least_value_val>":"<y_axis_inferred_least_value_approx>"}
	closest, smallest_diff, ind = -1, 1000, -3
	assigned_label = None
	for k, x in enumerate(numbers):
		try:
			x2=float(x)
			if abs(x2 - target) <= smallest_diff:
				closest, smallest_diff = x, abs(x2 - target)
				assigned_label = numbers_labels[k]
				#ind = numbers.index(closest)
				#print("\t", target, x2, smallest_diff, numbers_labels[k]) #, numbers)
		except ValueError:
			continue
	#assigned_label = numbers_labels[ind]
	#print("\t",numbers)
	if smallest_diff == 0 or assigned_label not in exact_approx: # exact value
		return assigned_label
	return exact_approx[assigned_label] # not exact, return label for approximation: only for bar heights, not * +


def in_title_labels(target, chart_text_dict):
	""" target is a token (str), chart_text_dict is a dict: label as key, str as value """
	# consider the labels: t=[0|1]_a=[x|y|b]
	assigned_lbl = None
	#import pdb; pdb.set_trace()
	target = target.lower()
	barcount_word = num2word.word(chart_text_dict["<x_axis_label_count>"]).lower()
	if target == barcount_word:
		assigned_lbl = "<x_axis_label_count>"
		return assigned_lbl

	if target in all_stopwords:
		return assigned_lbl

	target_lemma = [t.lemma_ for t in nlp_en_core(target)][0] # list of 1
	title_lemmas = [t.lemma_ for t in nlp_en_core(chart_text_dict["title"].lower())]
	y_lemmas = [t.lemma_ for t in nlp_en_core(chart_text_dict["y_axis_name"].lower())]
	x_lemmas = [t.lemma_ for t in nlp_en_core(chart_text_dict["x_axis_name"].lower())]

	if target_lemma in title_lemmas:
		if target_lemma in y_lemmas and target_lemma not in x_lemmas:
			assigned_lbl = "<t=1_a=y>"
		elif target_lemma in x_lemmas and target_lemma not in y_lemmas:
			assigned_lbl = "<t=1_a=x>"
		else:
			assigned_lbl = "<t=1_a=b>"

	elif target_lemma not in title_lemmas and target_lemma in y_lemmas:
		assigned_lbl = "<t=0_a=y>"
	elif target_lemma not in title_lemmas and target_lemma in x_lemmas:
		assigned_lbl = "<t=0_a=x>"

	else:

		# this might be because of the way lemmatization is done; some examples
		# glacier --> lemma "glaci"; glaciers --> lemma "glaciers"
		# America --> lemma "american"
		if len(target) > 2:
			crop = target[:-1]
			#print(" \t ---", target, crop, chart_text_dict["title"].lower(), "\n")
			if crop in chart_text_dict["title"].lower() and crop not in {chart_text_dict["y_axis_name"].lower(), chart_text_dict["x_axis_name"].lower()}:
				assigned_lbl = "<t=1_a=b>"
			elif crop in chart_text_dict["y_axis_name"].lower():
				assigned_lbl = "<t=0_a=y>"
			elif crop in chart_text_dict["x_axis_name"].lower():
				assigned_lbl = "<t=0_a=x>"
		else:
			if target not in {",", ".", "!", "?" , "(", ")", "-"} and not assigned_lbl:
				print("not in title, not a stopword, not a punct., and not labeled : ", target)
		#import pdb; pdb.set_trace()
	#if target in {} #
	return assigned_lbl

# TODO: some bar names tagged at first, then not anymore
# city is a also left untagged (x axis name)
# appears in title or not: t=0_a=x/y/b
# "and" don't label it with range  DONE ?
# closes value: why preferring _inferred_add/mul? DONE


def post_check(labeled_summary, basic_cal_as_text, basic_cal_as_values, info_other, units_set, magnitude_set):
	"""
	labeled summary : list of tuples (unigram, label) : label is either None or a label (str)
	info_basic : dict : basic plot info (bars and their heights)
	info_cal : dict : info about multiplication and addition
	info_other : dict : info about the x and y axis labels, plot title, number of bars
	"""
	new_lab_sum = []
	# first check unigrams
	for (unigram, label) in labeled_summary:
		if len(unigram) <= 1 or label: # tokens with 1 character (punctuation, a, i) or labeled tokens
			new_lab_sum.append((unigram, label))
			continue
		elif unigram[-1].lower() in {"c", "m"}: # cases: 23c, 36C, 100m
			if unigram[:-2].isdigit():
				uni1, uni2 = unigram[:-2], unigram[-1]
				lbl1 = find_closest(float(uni1), basic_cal_as_text, basic_cal_as_values)
				lbl2 = "<y_axis_inferred_label>"
				new_lab_sum += [(uni1, lbl1), (uni2, lbl2)]
				continue
		elif len(unigram) >= 9 and (unigram.endswith("million") or unigram.endswith("millions")):
			if unigram.endswith("million") and unigram[:-len("million")].isdigit():
				uni1, uni2 = unigram[:-len("million")], "million"
				lbl1 = find_closest(float(uni1), basic_cal_as_text, basic_cal_as_values)
				lbl2 = "<y_magnitude>"
				new_lab_sum += [(uni1, lbl1), (uni2, lbl2)]
				continue
			if unigram.endswith("millions") and unigram[:-len("millions")].isdigit():
				uni1, uni2 = unigram[:-len("millions")], "millions"
				lbl1 = find_closest(float(uni1), basic_cal_as_text, basic_cal_as_values)
				lbl2 = "<y_magnitude>"
				new_lab_sum += [(uni1, lbl1), (uni2, lbl2)]
				continue
		if "\\n" in unigram and not label: # cases america.\\n\\nthe
			# find indices of \\n
			ind = [(m.start(0), m.end(0)) for m in re.finditer("\\\\n", unigram)]
			unigrams_labels = []
			if len(ind) == 0:
				print("no matches despite separator in string?")
			else:
				s1, e1 = ind[0][0], ind[-1][-1]
				uni1 = unigram[s1].split() # list of tokens till first separator; might be a single token
				uni2 = unigram[e1:] # after the last separator; might be an empty list
				u_label = None
				if uni1:
					for u in uni1:
						# check if u is in bar names, heights or anything entity-like, see below

						if u in units_set:
							u_label = "<y_axis_inferred_label>"
							unigrams_labels.append((u, u_label))
							continue

						try:
							float(u)
							# closest match, append, continue
							# numbers, target, numbers_labels
							u_label = find_closest(float(u), basic_cal_as_text, basic_cal_as_values)
							unigrams_labels.append((u, u_label))

						except ValueError:
							# the token is a word, not a number; check if the token matches any word in basic_cal_text

							if u.lower() in basic_cal_as_text:
								tokindex = basic_cal_as_text.index(u.lower())
								u_label = basic_cal_as_values[tokindex]
								unigrams_labels.append((u, u_label))
								
							continue


def read_summaries(summary_file, info_basic, info_cal, info_other):
	"""
	summary_file is a path to the file with summaries from a single plot, one summary per line 
	summary_file is a txt file containing summaries belonging to one chart
	info_basic is a dict with basic plot info, such as bar1: height1, bar2: height2
	info_cal is a a dict with calculated info, such as mul_bar1_bar2: k, so multiplication and addition
	"""
	#print(info_basic, info_cal)
	syns = {"monday":"mon", "tuesday":"tue", "wednesday":"wed", "thursday":"thu", "friday":"fri", "genetics":"genetic", "%": "percent", "$": "dollars", "uk":"u.k.", "u.k.":"uk"}
	units = {"%", "$", "£", "pounds", "pound", "sterling", "dollar", "dollars", "percent", "degrees","degree" ,"c", "°c", "°", "celsius"}
	magnitude = {"thousand", "thousands", "k", "million", "millions", "m"}

	basic_text = [str(s).lower() for s in list(info_basic.values())]
	basic_raw = [str(s).lower() for s in list(info_basic.values())]
	basic_round = []
	for s in info_basic.values():
		if type(s) in {float,int}:
			basic_round.append(str(round(s)))
		else:
			basic_round.append(str(s).lower())

	# relative values
	cal_text = [str(s).lower() for s in list(info_cal.values())]
	cal_raw = [str(s).lower() for s in list(info_cal.values())]
	cal_round = []
	for s in info_cal.values():
		if type(s) in {float,int}:
			cal_round.append(str(round(s)))
		else:
			cal_round.append(str(s).lower())
	#print(basic_text, basic_raw, basic_round) # _text not really used

	# join the two lists, starting with the calculated so that the last closest value will probably be from basic
	basic_cal_text = cal_text + basic_text
	basic_cal_values = list(info_cal.keys()) + list(info_basic.keys())
	#print(basic_cal_text, "\n", basic_cal_values)

	summaries_final = []  # list of labeled summaries for a single plot
	with open(version_dir+summary_file, "r", encoding="utf8") as f:
		for line in f:
			if len(line) < 6:
				continue
			
			line2 = line[3:-4]
			linesplit = line2.split()
			if linesplit == []:
				continue
			cn, cl = 0, 0
			#tline = word_tokenize(line)

			doc = nlp_en_core(line2)
			tokens = [t.text for t in doc]
			nchunk = [] # noun chunk text, start index, end index, label

			labeled_chunk_dict = {}
			labeled_chunk_ind = set()
			for chunk in doc.noun_chunks:
				a_label = None
				for label, text in info_basic.items():
					text2 = str(text).lower()
					chunk2 = chunk.text.lower()
					if chunk2 == text2:
						a_label = label
						break
					if chunk2 in syns:
						if syns[chunk2] == text2:
							a_label = label
							break
				if a_label:
					cl +=1
					nchunk.append((chunk.text, chunk.start, chunk.end, a_label))
					labeled_chunk_ind = labeled_chunk_ind.union(set(np.arange(chunk.start, chunk.end)))
					labeled_chunk_dict[tuple(np.arange(chunk.start, chunk.end))] = (chunk.text, a_label)


			labeled_token_i = set()
			ltoken = []
			ltoken_dict = {}
			if False:
				"nothing, fix this"
			else:
				for i, t in enumerate(tokens):


					if t in units:
						t_label = "<y_axis_inferred_label>"
						labeled_token_i.add(i)
						ltoken_dict[i] = (t,t_label)
						continue

					try:
						float(t)
						# closest match, append, continue
						# numbers, target, numbers_labels

						t_label = find_closest(float(t), basic_cal_text, basic_cal_values)
						labeled_token_i.add(i)
						#input("next token")

						ltoken_dict[i] = (t, t_label)

					except ValueError:
						# the token is a word, not a number; check if the token matches any word in basic_val_text

						if t.lower() in basic_cal_text:
							tokindex = basic_cal_text.index(t.lower())
							labeled_token_i.add(i)
							ltoken_dict[i] = (t, basic_cal_values[tokindex])

						continue


			labeled_summary = [] # tuple (token/s, label)
			checked_j = set()

			for j in range(len(tokens)):
				zz = j
				#if zz == 90: # Lima (in a noun chunk)
					#import pdb; pdb.set_trace()

				if j in checked_j: continue
				if j in labeled_chunk_ind and j not in labeled_token_i:
					#import pdb; pdb.set_trace()
					match = [k for k,tup in labeled_chunk_dict.items() if k[0] == j]

					for p1 in match:
						if len(p1) == 1: checked_j.add(p1[0])
						if len(p1) > 1:
							checked_j = checked_j.union(set(np.arange(p1[0],p1[1]+1)))
					if match == []:
						print("debug", labeled_chunk_ind, checked_j,j, tokens[j])
						input()

					labeled_summary.append(labeled_chunk_dict[match[0]])

					continue

				if j in labeled_token_i:
					labeled_summary.append(ltoken_dict[j])
					checked_j.add(j)
					continue

				elif j not in checked_j:

					t = tokens[j]
					t_label = None
					t1, t2 = None, None
					t2_label = None

					if t.endswith("000"):
						t2_label = "<y_magnitude>"

						if t.endswith(",000") == False: # 30000
							t = tokens[j][:-3]
							t2 = tokens[j][-3:]

						if t.endswith(",000") and t.endswith("000,000") == False:
							t = tokens[j][:-4] # the new numerical token to be checked for labels 
							t2 = tokens[j][-4:]

						if tokens[j].endswith(",000,000"):
							t = tokens[j][:-8]
							t2 = tokens[j][-8:]

						if tokens[j].endswith("000000"):
							t = tokens[j][:-6]
							t2 = tokens[j][-6:]
							

					if t in magnitude: # when magnitude is expressed verbally (k or thousand)
						t_label = "<y_magnitude>"

					if t in units:
						t_label = "<y_axis_inferred_label>"


					if t.lower() in word_dlabel:
						current = word_dlabel[t.lower()]
						t_label = max(current, key=current.get) # use the most frequent label
						t_label = "<" + t_label + ">"


					# for cases where t was split: 23,000 into 23 and ,000 - check if match
					if t2 or "." in t: # t2 is either None or in ,000 000 ,000,000 000000
							# or covering cases like 13.5
						try:
							float(t)
							# closest match, append, continue
							# numbers, target, numbers_labels
							t_label = find_closest(float(t), basic_cal_text, basic_cal_values)
							#print(t, t_label)

						except ValueError:
							#continue
							pass

					if t_label:
						labeled_summary.append((t, t_label))

					if t2 and t2_label:
						labeled_summary.append((t2, t2_label))

					
					#if "," in t and len(t) > 1: print(t) # 4 cases: 
					#22,500 26,500 19,750 23,000.\\n\\nThe 2000,2005,2010

					#if t in {"13.5", "8.8", "8.5"}:
						#print("\t ---", tokens[j], t, t_label, t2_label)

					elif t_label == None:

						# check if the token appears in the title, or the axis labels
						# new_lbl can be either None or a label
						new_lbl = in_title_labels(tokens[j], info_other)

						labeled_summary.append((tokens[j], new_lbl))

			# check the parsed and labeled summary for frequent errors and fix tem
			post_check(labeled_summary, basic_cal_text, basic_cal_values, info_other, units, magnitude)

			import pdb; pdb.set_trace()
			summaries_final.append(labeled_summary)
	return summaries_final



def get_data_dict(dd,ii):
	for b,v in dd.items():
		if v["image_index"] == ii:
			return v



def get_according_fnames(dfs, splitt):
	fnames = []
	for fname, (splitname, image) in dfs.items():
		if splitname == splitt:
			fnames.append((fname, splitname, image))
	return fnames


if __name__ == "__main__":

	# load the annotations.json of batch 3
	anno_json_fpath = "/home/iza/chart_descriptions/data_batch3/annotations.json"
	with open(anno_json_fpath, "r") as jsf:
		all_chart_data = json.load(jsf)

	extracted = get_x_y(anno_json_fpath) # dict; key: index, value, dict of chart info
	#import pdb; pdb.set_trace()
	# tuples [(summary_file_name, json_ID) ... ]
	maps = get_info_from_map() 


	dlabel_word, word_dlabel = discourse_labels()
	new_pn = "/home/iza/chart_descriptions/data_batch3/auto_labeled"
	for (summary_fname, json_ID) in maps:
		c = 0
		current_basic, current_cal, current_other = get_stat_info(extracted[json_ID])
		auto_labeled = read_summaries(summary_fname, current_basic, current_cal, current_other)
		#import pdb; pdb.set_trace()
	

	for data_split,path_json in jsons.items():
		c = 0
		new_pn = "/home/iza/chart_descriptions/data_batch3/auto_labeled"

		# "batch1/akef_inc_closing_stock_prices_1.txt":("train1", 2) are the items in descriptions_files_json
		#get_according_filenames = lambda dfs, splitt: [(fname, splitname, image) for fname, (splitname,image) in dfs.items() if splitname == splitt] # TODO won't work of because incorrect syntax (lamda doesn't like if in this way)
		current_triplets = get_according_fnames(descriptions_files_json, data_split)
		extracted = get_x_y(path_json)

		for (filename, splitname, i_image) in current_triplets:
			v = get_data_dict(extracted, i_image)
			current_basic, current_cal = get_stat_info(v)
			auto_labeled = read_summaries(filename, current_basic, current_cal)
			new_fn = new_pn + filename[7:]
			newdoc = open(new_fn, "w")
			for single in auto_labeled:
				newdoc.write("<start_of_description>")
				newdoc.write("\n")
				for tup in single:
					if tup[1] == None:
						newdoc.write(tup[0])
					else:
						newdoc.write(tup[0] + "\t" + tup[1])
					newdoc.write("\n")
				newdoc.write("<end_of_description>")
				newdoc.write("\n")
				newdoc.write("\n")
			newdoc.close()
			#input("next file")
		

# 1.4 k more labels assigned, bigger recall, precision?
