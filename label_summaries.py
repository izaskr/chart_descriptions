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
import mord
from sklearn.linear_model import LinearRegression
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH, NORM
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()


parser.add_argument("-write", required=True, help="write y or n to decide if the labeled data should be written into a file")
#parser.add_argument("-summaries", required=True, help="file with chart summaries; unlabeled and unpreprocessed")

args = vars(parser.parse_args())

#summary_file = args["summaries"]
write_yn = args["write"]
write_file = False
if write_yn in {"y", "yes"}: write_file = True
if write_yn not in {"y","n", "yes", "no"}: sys.exit("Use 'y' or 'n' as value for -write")


version_dir = "corpora_v02/run2_chart_summaries/"
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
jsons = ["run2_jsons/train1/annotations.json", "run2_jsons/val1/annotations.json", "run2_jsons/val2/annotations.json"]
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


# v["image_index"]

# just the tokenizer, rule for ,000 be treated as a single token, not split
nlp = English()  # just the language with no model

#nlp.add_pipe(tokenizer)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

tokenizer = Tokenizer(nlp.vocab)
case_000 = [{ORTH: ",000"} ]#, {ORTH: "BB", NORM: ", 000"}]
tokenizer.add_special_case(",000", case_000)


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

		xy[i+1] = {"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, "x_axis_label_name": x_axis_label_name, "x_major_ticks":x_major_ticks,"y_major_ticks":y_major_ticks, "image_index":image_index}


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
	#print("\t",label_name_pairs_x)
	#print("\t",label_value_pairs_y)


	# relations
	# mul y_axis_inferred_value_mul_v1=highest_v2=least , y_axis_inferred_value_mul_v1=highest_v2=Scnd
	# add y_axis_inferred_value_add_v1=highest_v2=least

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

	"""

	slope = None
	# linear regression: x is ratio
	if data["x_order_info"]["x_is_ordered"] and data["x_order_info"]["x_is_ratio"]:
		x_reshape = np.asarray([int(c) for c in x]).reshape(-1,1)
		y_order_as_x = np.asarray(data["x_order_info"]["y_order_as_x"])
		lr = LinearRegression().fit(x_reshape,y_order_as_x)
		slope = lr.coef_[0]
		

	# ordered logistic regression: x is ordinal
	if data["x_order_info"]["x_is_ordered"] and data["x_order_info"]["x_is_ratio"] == False:
		x_dummy_coding = np.asarray([i for i in range(len(x))]).reshape(-1,1)
		y_order_as_x = np.asarray(data["x_order_info"]["y_order_as_x"])
		olr = mord.OrdinalRidge(alpha=0.001,fit_intercept=True,normalize=False, copy_X=True,max_iter=None,tol=0.001,solver="auto")
		olr.fit(x_dummy_coding,y_order_as_x)
		slope = olr.coef_[0]
		#print(slope)

	"""
		

	"""
	if show_info:
		print(data["title"])
		print("x categories \t", x)
		print("y values \t", y)
		print("x major ticks", data["x_major_ticks"])
		print("y major ticks", data["y_major_ticks"])
		print("y axis unit \t", data["y_axis_unit_name"])
		print("x axis name \t", data["x_axis_label_name"])
		print("differences ADD", differences["plus"])
		print("differences MUL", differences["times"])
		print("category count", misc["<x_axis_label_count>"])
		print("x_order_info", data["x_order_info"])
		if slope:
			print("slope of regression line", slope)
	"""
	#return data["title"], data["x_order_info"],differences, label_name_pairs_x,label_value_pairs_y, misc
	#return results
	return results

"""
for a in jsons:
	extracted = get_x_y(a)
	for b,v in extracted.items():
		get_stat_info(v)
		input("Press Enter to show the next plot")
"""

# for every json in jsons
# 	have a list of corresponding txt files
#	for every file
#		go line by line and preprocess the description (tokenize, segment)
#		labeling based on string matching
#		write into file, mark the beginning and end of summary


def tokenize_label(s, mapping):
	""" s is a chart summary as a string, mapping is a dictionary as returned by get_stat_info """
	return None


def read_summaries(summary_file, plot_info):
	""" summary_file is a path to the file with summaries from a single plot, one summary per line """
	
	with open(version_dir+summary_file, "r", encoding="utf8") as f:
		for line in f:
			tline = word_tokenize(line)
			#tline2 = tokenizer(line)
			#if tline: print(tline)
			#tline3 = [t.text for t in tline2]
			#if tline3: print(tline3)
			#print(tline)
			for token in tline:
				if token in {"%", "$", "£", "pounds", "dollars", "dollar", "pound"}:
					print(token)
			# use Spacy's dep parsing to get chunks and check if any match the x labels

			


for data_split,path_json in jsons.items():
	for filename, (ds, i_image) in descriptions_files_json.items():
		#print(filename, ds)
		if data_split == ds:
			extracted = get_x_y(path_json)
			for b, v in extracted.items():
				if v["image_index"] == i_image:
					current_data = get_stat_info(v)
					read_summaries(filename, current_data)
					
			
	
	

"""

if __name__ == "__main__":

	data = descriptions_files_json[annotations][0] + "_annotations3.json"

	data_ext = get_x_y(data)

	title = descriptions_files_json[annotations][1]

	d = {}
	for i, d in data_ext.items():
		if d["title"] == title:
			corpus = d
	#print(corpus)
	results = get_stat_info(corpus)

"""

