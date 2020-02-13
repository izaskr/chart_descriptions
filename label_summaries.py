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


# v["image_index"]

# just the tokenizer, rule for ,000 be treated as a single token, not split
nlp = English()  # just the language with no model

#nlp.add_pipe(tokenizer)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

tokenizer = Tokenizer(nlp.vocab)
case_000 = [{ORTH: ",000"} ]#, {ORTH: "BB", NORM: ", 000"}]
tokenizer.add_special_case(",000", case_000)


nlp_en_core = spacy.load("en_core_web_sm")



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
	return results_basic, results_cal

def discourse_labels():
	""" dlabel_word is a dict, discourse labels as keys, vocab set as value
	word_dlabel is a dict, word as key, dict counter of labels and their counts as value
	"""
	xml_file = "corpora_v02/chart_summaries_b01.xml"
	dlabel_word, word_dlabel = get_discourse_tokens(xml_file)
	return dlabel_word, word_dlabel

def find_closest(numbers, target, numbers_labels):
	""" in a list of numbers, it finds the number closest to the target number and returns the number's index"""
	closest, smallest_diff, ind = -1, 1000, -3
	for x in numbers:
		try:
			x2=float(x)
			if abs(x2 - target) < smallest_diff:
				closest, smallest_diff = x, abs(x2 - target)
				ind = numbers.index(closest)
		except ValueError:
			continue
	#return closest, ind
	return numbers_labels[ind]



def read_summaries(summary_file, info_basic, info_cal):
	""" summary_file is a path to the file with summaries from a single plot, one summary per line """
	#print(info_basic, info_cal)
	syns = {"monday":"mon", "tuesday":"tue", "wednesday":"wed", "thursday":"thu", "friday":"fri", "genetics":"genetic", "%": "percent", "$": "dollars", "uk":"u.k.", "u.k.":"uk"}
	units = {"%", "$", "£", "pounds", "pound", "dollar", "dollars", "percent"}
	magnitude = {"thousand", "thousands", "k"}

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

	summaries_final = []  # list of labeled summaries for a single plot
	with open(version_dir+summary_file, "r", encoding="utf8") as f:
		for line in f:
			if len(line) < 6:
				continue
			
			line2 = line[3:-4]
			linesplit = line2.split()
			if linesplit == []:
				continue
			cn,cl = 0,0
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
				for i,t in enumerate(tokens):

					if t.lower() in basic_text:
						t_label = list(info_basic.keys())[basic_text.index(t.lower())]
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						continue

					if t.lower() in basic_raw:
						t_label = list(info_basic.keys())[basic_raw.index(t.lower())]
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						continue

					if t.lower() in basic_round:
						t_label = list(info_basic.keys())[basic_round.index(t.lower())]
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						continue


					if t.lower() in cal_raw:
						t_label = list(info_cal.keys())[cal_raw.index(t.lower())]
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						continue

					if t.lower() in cal_round:
						t_label = list(info_cal.keys())[cal_round.index(t.lower())]
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						continue

					if t in units:
						t_label = "<y_axis_inferred_label>"
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)
						#print("---------------------UNIT")	
						continue

					try:
						int(t)
						# closest match, append, continue
						# numbers, target, numbers_labels
						t_label=find_closest(basic_text, int(t), list(info_basic.keys()) )
						labeled_token_i.add(i)
						ltoken.append((t, i, t_label))
						ltoken_dict[i] = (t,t_label)

					except ValueError:
						continue

			
			labeled_summary = [] # tuple (token/s, label)
			checked_j = set()

			for j in range(len(tokens)):

				if j in checked_j: continue
				if j in labeled_chunk_ind:

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

					if t.lower() in cal_raw:
						t_label = list(info_cal.keys())[cal_raw.index(t.lower())]


					if t.lower() in cal_round:
						t_label = list(info_cal.keys())[cal_round.index(t.lower())]



					if t.lower() in basic_raw:
						t_label = list(info_basic.keys())[basic_raw.index(t.lower())]


					if t.lower() in basic_round:
						t_label = list(info_basic.keys())[basic_round.index(t.lower())]
						

					if t.lower() in basic_text:
						t_label = list(info_basic.keys())[basic_text.index(t.lower())]


					if t.lower() in word_dlabel:
						current = word_dlabel[t.lower()]
						t_label = max(current, key=current.get) # use the most frequent label
						t_label = "<" + t_label + ">"


					# for cases where t was split: 23,000 into 23 and ,000 - check if match
					if t2: # t2 is either None or in ,000 000 ,000,000 000000
						try:
							int(t)
							# closest match, append, continue
							# numbers, target, numbers_labels
							t_label=find_closest(basic_text, int(t), list(info_basic.keys()) )
							
							#labeled_token_i.add(i)
							#ltoken.append((t, i, t_label))
							#ltoken_dict[i] = (t,t_label)

						except ValueError:
							continue

					if t_label:
						labeled_summary.append((t, t_label))

					if t2 and t2_label:
						labeled_summary.append((t2, t2_label))
						print("\t", t, t_label, t2, t2_label)


					else: labeled_summary.append((tokens[j], None))
					#continue
			#for unit in labeled_summary:
			#	print("\t",unit)
			#print("eod")
			summaries_final.append(labeled_summary)
	return summaries_final


def write_file(labeled_summaries, fname):
	"""
	labeled summaries
	"""
	return None

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
	dlabel_word, word_dlabel = discourse_labels()

	for data_split,path_json in jsons.items():
		c = 0
		new_pn = "corpora_v02/run2_chart_summaries/auto_labeled01/"

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
		

# 1.4 k more labels assigned, bigger recall, precision?
