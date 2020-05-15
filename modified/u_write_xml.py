
"""
Rewrite annotated vertical description corpora into XML following the InScript schema
"""
import time
import json
import numpy as np
import argparse
from itertools import combinations
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH, NORM
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
from collections import defaultdict

from initialize_dicts import *


def is_digit(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_closest(num, l):
    """
    Takes a number and a list, returns a comma seperated string of 
    the two closes elements of the list to the number.
    """
    ret = [None]
    msf = float('inf') #min so far
    s = set(l)
    for i in l: 
        if abs(i-num) <= msf: 
            msf = abs(i-num)
            ret[0] = i
    
    for i in l: 
        if i == ret[0]:
            continue
        if abs(i-num) == msf: 
            msf = abs(i-num)
            ret.append(i)
    
    if len(ret) == 1: 
        ret.append(ret[0])
	    
    ret.sort()    
    ret = [str(i) for i in ret]
    
    return ",".join(ret)



parser = argparse.ArgumentParser()
#parser.add_argument("-data", required=True)
#parser.add_argument("-labeled_file", required=True, help="file with annotated description")
parser.add_argument("-labeled_file", required=False, help="file with annotated description")
#parser.add_argument("-img_id", required=True, help="int denoting graph id")
#parser.add_argument("-csv", required=True, help="csv file from Prolific")
args = vars(parser.parse_args())



nlp_web = spacy.load("en_core_web_sm") # used only for tokenization

nlp = English()  # just the language with no model

#nlp.add_pipe(tokenizer)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

# just the tokenizer, rule for ,000 be treated as a single token, not split
tokenizer = Tokenizer(nlp.vocab)
case_000 = [{ORTH: ",000"} ]#, {ORTH: "BB", NORM: ", 000"}]
tokenizer.add_special_case(",000", case_000)


version_dir = "../corpora_v02/all descriptions/" #assumes all description txt would be here, and the output also saved here  




def get_x_y(filename):
	"""
	Function for extracting only the information needed to analyze the data, return a dict for every plot,
	including the values of x axis, y axis, plot type, plot title

	:param filename: name of annotations.json file, which follows a certain format
	:type filename: str
	"""
	with open(filename) as f:
		data = json.load(f)

	xy = {}
	
	for i,k in enumerate(data):
		x = k["models"][0]["x"]
		y = k["models"][0]["y"]
		title = k["general_figure_info"]["title"]["text"]
		x_order_info = k["models"][0]["x_order_info"]
		plot_type = k["type"] # "vbar_categorical"
		y_axis_unit_name = k["general_figure_info"]["y_axis"]["label"]["text"] # label of y from the plot
		x_axis_label_name = k["general_figure_info"]["x_axis"]["label"]["text"]

		y_axis_major_ticks = k["general_figure_info"]["y_axis"]["major_ticks"]["values"] 
		y_axis_major_ticks = sorted(list(set(y_axis_major_ticks)))
		

		
		y_axis_minor_ticks = k["general_figure_info"]["y_axis"]["minor_ticks"]["values"] 
		y_axis_minor_ticks = sorted(list(set(y_axis_minor_ticks)))

		
		xy[i+1] = {
			"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, 
			"x_axis_label_name": x_axis_label_name, "y_axis_major_ticks": y_axis_major_ticks, "y_axis_minor_ticks": y_axis_minor_ticks}

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
	

	times = {"label": "<y_axis_inferred_value_mul>","pairs_int":{}, "pairs_float":{}, "pairs_float_reverse":{} }
	# To interpret this dict: (c1, c2): 5 -> c1 has a value about 5 times larger than c2
	# OR: c2 has a value 5 times smaller than c1
	
	plus = {"label": "<y_axis_inferred_value_add>","pairs":{}}
	# To interpret this dict: (c1, c2): 13 -> c1 has a value that is 13 [unit] larger than c2
	# OR: c2 has a value 13 [unit] smaller than c1
	x_combin, y_combin = list(combinations(x_sorted,2)), list(combinations(y_sorted,2))
	for i,pair in enumerate(x_combin):
		k = y_combin[i][0] // y_combin[i][1] # floor division, result rounded down int

		k2 = round(y_combin[i][0] / y_combin[i][1], 2) # float division, round to 2 decimals
		k2_reverse = k2 ** (-1)
		pair_reverse = tuple((pair[1],pair[0]))

		times["pairs_int"][pair] = k
		times["pairs_float"][pair] = k2
		times["pairs_float_reverse"][pair_reverse] = round(k2_reverse,2)		

		d = y_combin[i][0] - y_combin[i][1] # difference
		plus["pairs"][pair] = d
	
	differences = {"plus":plus, "times":times}

	results = {"differences": differences, "label_name_pairs_x": label_name_pairs_x, "label_value_pairs_y": label_value_pairs_y, "misc": misc,"y_axis_unit_name": data["y_axis_unit_name"],"x_axis_label_name": data["x_axis_label_name"]}
	#print(data["title"], x, y)
	#return data["title"], data["x_order_info"],differences, label_name_pairs_x,label_value_pairs_y, misc
	return results





def collect_parse(annotations, calculated):
	""" 
	annotations is a file with several annotated plot descriptions
	calculated is a dict returned by get_stat_info
	"""

	differences = calculated["differences"]
	label_name_pairs_x = calculated["label_name_pairs_x"]
	label_value_pairs_y = calculated["label_value_pairs_y"]
	label_count = calculated["misc"]
	y_axis_unit_name = calculated["y_axis_unit_name"]
	x_axis_label_name = calculated["x_axis_label_name"]


	previous_line_empty = True
	#TODO: check encoding
	# with open(version_dir + annotations, "r", encoding="ISO-8859-1") as f:
	with open(version_dir + annotations, "r", encoding="ISO-8859-1") as f:
		end_desc = False
		desc_tokens = [] # list of strings
		desc_labels = [] # list of tuples 
		i_token = 0 # track of tokens within one description
		for_xml = {}
		i_desc = 1
		for line in f:
			line2 = line.split()
			line3 = " ".join(line2)

			if line2 == []: # an empty line between descriptions
				end_desc = True
				previous_line_empty = True

			if len(line2) == 1:
				if line2[-1][0] not in {'<', '"'}:
					#print("no label", line2)
					i_token += 1
					desc_tokens.append(line2[0])

				# if line2[-1][0]  == "<":
				# 	print(line2[-1], line2[0], "\n\n")
			

			if len(line2) > 1 and line2[-1].startswith("<"):
				previous_line_empty = False
				label = line2[-1]

				if len(line2[:-1]) == 1: # one token, one label
					desc_tokens.append(line2[0])
					i_token += 1
					desc_labels.append((label, i_token, i_token, line2[0]))
					
				else: # several tokens, one label
					i_token += 1

					temp = " ".join(line2[:-1])
					#temp_doc = nlp_web(temp) # old?
					temp_doc = tokenizer(temp)
					temp_tokens = [t.text for t in temp_doc]
					if temp_tokens == [",","000"]: # TOKENIZER EXCEPTION

						temp_tokens = [",000"]
					temp_join = " ".join(temp_tokens)
					i_end = i_token + len(temp_tokens) - 1
					desc_tokens += temp_tokens
					desc_labels.append((label, i_token, i_end, temp_join))
					i_token = i_end




			if len(line2) == 1 and line2[0] == "<end_of_description>" and previous_line_empty == False:
				if desc_tokens:

					desc = " ".join(desc_tokens)

					#doc = nlp(desc) # TODO this is where ,000 is split into , and 000 old?
					doc = tokenizer(desc)

					doc_tokens = [t.text for t in doc]

					#print(desc_tokens,"\n" , "\n", doc_tokens, "\n \n", desc_labels, "\n",desc, "\n", len(desc_tokens), len(doc))

					for_xml[i_desc] = {"desc_tokens":desc_tokens,"doc_tokens":doc_tokens,"desc_labels":desc_labels}
					i_desc += 1
					
				desc_tokens, desc_labels, i_token = [], [], 0
				
				
	return for_xml




def prettify(elem):
	"""Return a pretty-printed XML string for the Element.
	"""
	#TODO: check encoding	
	# rough_string = ET.tostring(elem, 'utf-8')
	rough_string = ET.tostring(elem, 'ISO-8859-1')
	reparsed = parseString(rough_string)
	return reparsed.toprettyxml(indent="\t")


def make_xml_tree():
	""" d is a dictionary, indices as keys, dict as values with keys desc_tokens, doc_tokens, desc_labels """
	global my_cnt
	main_root = ET.Element("summaries")

	for tri in zip(description_files_order, topic_image_id):

		# print("::::::::::::::::::::::::::::::::::::::::::")
		# print(tri)
		# print("::::::::::::::::::::::::::::::::::::::::::")
		annotations = tri[0]
		chart_topic, chart_id, chart_x_type = tri[1][0], tri[1][1], tri[1][2]

		

		data = descriptions_files_json[annotations][0] + "_annotations3.json"
		data_ext = get_x_y("../"+data)
		# print("::::::::::::::::::::::::::::::::::::::::::")
		# print(data_ext)
		# print("::::::::::::::::::::::::::::::::::::::::::")
		title = descriptions_files_json[annotations][1]
		# print("::::::::::::::::::::::::::::::::::::::::::")
		# print(annotations, "       ", chart_topic, "     ", title)
		# print("::::::::::::::::::::::::::::::::::::::::::")
		d = {}
		for i, d in data_ext.items():
			if d["title"] == title: 
				corpus = d
		# print(corpus['title'], "       ", title)
		results = get_stat_info(corpus)
		for_xml = collect_parse(annotations, results)

		# print("line 310 **********")
		# for key in for_xml:
		# 	print('\n', key)
		# 	print(for_xml[key])
		# print("\nline 310 **********")
		desc_ids = list(for_xml.keys())
		# print(desc_ids)

		ex = set()

		topic_root = ET.SubElement(main_root,"topic")
		topic_root.set("topic", chart_topic)
		topic_root.set("topic_id", chart_id) # the name of the bar chart image file
		topic_root.set("scenario", chart_x_type)

		# print("\n\n\n\n\n\n\n")
		for story_id in desc_ids: # iterate over all descriptions of a single chart
			d1 = for_xml[story_id]

			# print("::::::::::::::::::::::::::::::::::::::::::")
			# print(d1)
			# print("::::::::::::::::::::::::::::::::::::::::::")

			str_story_id = str(story_id)
			if story_id < 10:
				str_story_id = "0" + str_story_id

			topic_story_id = chart_id + "-" + str_story_id
			i, j, k = 0, 0, 0

			story = ET.SubElement(topic_root, "story")
		
			story.set("story_id", topic_story_id)

			text = ET.SubElement(story, "text")
			content = ET.SubElement(text, "content")

			content.text = " ".join(d1["desc_tokens"])
			sentences = ET.SubElement(text, "sentences")

			doc = nlp(" ".join(d1["desc_tokens"]))
			label_info = d1["desc_labels"]
	
			boundaries = [] # list of sentence-final-token indices
			sentence_lengths = []
			s_counts = 0
			for sent in doc.sents:
				s_counts += 1
				i += 1 # sentence index, starting with 1
				j = 0 # index for tokens within a single sentence, starting with 1
				add_sent = ET.SubElement(sentences, "sentence")
				add_sent.set("id", topic_story_id + "-" + str(i))

				# ADD FIX: the tokenizer inherent to the sentencizer cannot be adapted for ,000
				tokens = tokenizer(sent.text) # str

				for token in tokens:
					k += 1 # index for tokens within the entire description, starting with 1
					j += 1
					add_token = ET.SubElement(add_sent, "token")
					add_token.set("content", token.text)
					add_token.set("content_fix", token.text)
					add_token.set("id", topic_story_id + "-" + str(i)+"-"+str(j))

				# boundaries basically represent last index of each sentenc in the description 
				boundaries.append(k) # last token index of a sentence, starting with 1
				sentence_lengths.append(j)

	

			annotations = ET.SubElement(story, "annotations")
			events = ET.SubElement(annotations, "events")

			label_bound = {i:[] for i in range(1,len(boundaries)+1)}

			# TODO: start here

			"""
			dictionaries for hedges, approx and relex id maping
			this dictionary is intended to map from quadriple (label, start,end, text_str) to 
			the ID number (which is initially initialized to None)
			"""
			hedge_id_map = defaultdict(list)
			hedge_id_tracker = 1
			approx_id_map = defaultdict(list)
			approx_id_tracker = 1
			relex_id_map = defaultdict(list)
			relex_id_tracker = 1

			for (label, start, end, text_str) in label_info:
				for n, boundary in enumerate(boundaries):
					if end <= boundary:


						label_bound[n+1].append((label, start,end, text_str))


						#TODO: collect labels here
						# case: hedge
						if label.find("y_axis_approx") != -1:
							hedge_id_map[(label, start, end, text_str)] = [hedge_id_tracker, n+1, len(label_bound[n+1])-1]  # [id, sentence, label_bound_sentence_index]
							hedge_id_tracker += 1

						#case: approx
						elif label.find("value_approx") != -1: 
							approx_id_map[(label, start, end, text_str)] = [approx_id_tracker, n+1, len(label_bound[n+1])-1]
							approx_id_tracker += 1

						# case: relex
						elif label.find("_mul_") != -1 or label.find("_add_") != -1: 
							relex_id_map[(label, start, end, text_str)] = [relex_id_tracker, n+1, len(label_bound[n+1])-1]
							relex_id_tracker += 1
						
						break

			
			

			
			# print("LABEL BOUNDS")
			# print(label_info)
			# print(len(label_bound))
			# my_cnt = 0

			# info maps are used to store different informations about the numerical expression
			# currently maps from the unique ID to a list of the quadriples and corresponding annotation ID
			hedge_info_map = defaultdict(list)
			approx_info_map = defaultdict(list)
			relex_info_map = defaultdict(list)
			


			c = 0
			for i_sent, triplets in label_bound.items():
			# i_sent is the sentence index: within a description
			# triplets: a list of tuples (label, start_id, end_id) where id is according to entire desc.
				
				# print("TTTTTTTTTTTTTTTTTTT**************************")
				# print(corpus["title"])	
				# print("i_sent: ", i_sent)
				# print("triplets:",triplets) 
				# print("TTTTTTTTTTTTTTTTTTT**************************")

				if i_sent > 1:
					prev_boundary = boundaries[i_sent-2]
				for n,(label, start, end, text_str) in enumerate(triplets): 					
					
					c +=1		


					if i_sent == 1:
						from_value = str(i_sent)+"-"+str(start)
						to_value = str(i_sent)+"-"+str(end)
					else:
						#print(sentence_lengths, i_sent)
						start2 = start - prev_boundary
						from_value = str(i_sent)+"-"+str(start2)
						end2 = end - prev_boundary
						to_value = str(i_sent)+"-"+str(end2)

						
						if start2 == 0 or start2 == -1:
							# print(i_sent,triplets, prev_boundary, boundaries)
							pass

					# if ''.join(corpus["x"]) not in ex:
					# 	print("\nAAAAAAAA**********")	
					# 	print(corpus["y"])	
					# 	ex.add(''.join(corpus["x"]))
					# 	print("AAAAAAAA****************\n")	

					add_label = ET.SubElement(events, "label")
					add_label.set("from", topic_story_id + "-" + from_value)
					add_label.set("to", topic_story_id + "-" + to_value)

					# if any(x in label[1:-1] for x in ["_value_val", "_value_mul", "_value_add"]):
					# 	print(text_str, label[1:-1])
					# 	my_cnt += 1
					add_label.set("id", topic_story_id + "-" + str(c)) # label id
					add_label.set("name", label[1:-1])
					add_label.set("text", text_str)
					add_label.set("type", "event")


					# populate hedge/approx/relex imaps
					quad = (label, start, end, text_str)
					annotation_id = topic_story_id + "-" + str(c)

					ref_id_map = None
					ref_info_map = None

					if quad in hedge_id_map:
						ref_id_map = hedge_id_map
						ref_info_map = hedge_info_map

					elif quad in approx_id_map:
						ref_id_map = approx_id_map
						ref_info_map = approx_info_map
					
					elif quad in relex_id_map:
						ref_id_map = relex_id_map
						ref_info_map = relex_info_map
					
					else:
						continue
					

					ref_id = ref_id_map[quad][0]
					ref_info_map[ref_id] = [quad, annotation_id]
			
			


			def get_previous(n, lb_sent_key, lb_sent_index):
				lb_sentence = label_bound[lb_sent_key]
				index = lb_sent_index - n
				if index < 0: 
					return None
				return lb_sentence[index]

			def get_next(n, lb_sent_key, lb_sent_index):
				lb_sentence = label_bound[lb_sent_key]
				index = lb_sent_index + n
				if index >= len(lb_sentence):
					return None 
				return lb_sentence[index]

			# build tags for approximation
			def build_approx_tags(parent_tag):
				approximations = ET.SubElement(parent_tag, "approximations")

				for i in range(1, len(approx_id_map)+1):					
					unit, magnitude, hedge_str , format, vg, va, direction = "", "", "", "word", "##REVIEW", None, "##REVIEW"
					major_ts, minor_ts = "##REVIEW", "##REVIEW"

					
					# print(approx_info_map[i+1], "*********************")
					quad = approx_info_map[i][0]  ## get the quadriplpes 
					(curr_label, curr_start, curr_end, curr_text) = quad

					label_bound_sentence_key = approx_id_map[quad][1]
					label_bound_sentence_index = approx_id_map[quad][2]

					prev1 = get_previous(1, label_bound_sentence_key, label_bound_sentence_index)
					prev2 = get_previous(2, label_bound_sentence_key, label_bound_sentence_index)
					prev3 = get_previous(3, label_bound_sentence_key, label_bound_sentence_index)

					next1 = get_next(1, label_bound_sentence_key, label_bound_sentence_index)
					next2 = get_next(2, label_bound_sentence_key, label_bound_sentence_index)
					next3 = get_next(3, label_bound_sentence_key, label_bound_sentence_index)
					

					
					# get magnitude (currently only checks next1)
					if next1 is not None: 
						next1_label, next1_start, next1_end, next1_text = next1
						if next1_label == "<y_magnitude>" and next1_start == curr_end+1:
							magnitude = next1_text

					# TODO: extend unit
					# get magnitude (currently only checks prev1)

					unit_quad = None
					if prev1 and prev1[0] == "<y_axis_inferred_label>" and prev1[2]+1 == curr_start:
						unit_quad = prev1
						# print("prev 1")
					elif next1 and next1[0] == "<y_axis_inferred_label>" and curr_end+1 == next1[1]:
						unit_quad = next1
						# print("next 1")
					elif next2 and next2[0] == "<y_axis_inferred_label>" and curr_end+2 == next2[1]:
						if next1[0] == "<y_magnitude>":
							unit_quad = next2
							# print("next 2")
					if unit_quad:
						unit = unit_quad[3]
						# print(unit, "--------------------------------------")

					# if prev1 is not None: 
					# 	unit_label, unit_start, unit_end, unit_text = prev1
					# 	if unit_label == "<y_axis_inferred_label>" and unit_end+1 == curr_start:
					# 		unit = unit_text
							
					





					# get hedge string (currently only checks previous 2 indexes)
					hedge_quad = None
					if prev1 and prev1[0] == "<y_axis_approx>":
						hedge_quad = prev1
					elif prev2 and prev2[0] == "<y_axis_approx>":
						hedge_quad = prev2
					
					
					if hedge_quad is not None:
						hedge_str = hedge_quad[3]

					
					# get va
					# va_index = None
					# sorted_y_values = sorted(corpus['y'])
					
					# y_ordered_index = {
					# 	"least":0, "Scnd":1, "3rd":2, "4th":3, "5th":4, "6th":5, 
					# 	"highest":len(sorted_y_values)-1
					# 	}

					sorted_y_values = sorted(corpus['y'], reverse=True)  
					y_ordered_index = {
						"highest":0, "Scnd":1, "3rd":2, "4th":3, "5th":4, "6th":5, "least":len(sorted_y_values)-1
						}

					for key in y_ordered_index:
						if curr_label.find("inferred_"+key) != -1:
							va_index = y_ordered_index[key]
							break

					if va_index is not None: 
						va = sorted_y_values[va_index]


					# get format, vg and direction
					if is_digit(curr_text):
						format = "digit"
						vg = float(curr_text)
						if vg.is_integer():
							vg = int(vg)	

						if vg > va: 
							direction = "up"
						elif vg < va:
							direction = "down"


						# TODO: change to va
						# get closest major and minor ticks (only works if vg is digit)
						major_ts = get_closest(vg, corpus['y_axis_major_ticks'])
						minor_ts = get_closest(vg, corpus['y_axis_minor_ticks'])

						# print('\n\n', vg)
						# print(major_ts)
						# print(minor_ts)
						 
					

					add_approx = ET.SubElement(approximations, "approx") 

					
					add_approx.set("id", topic_story_id + "-" + str(i)) # label id
					add_approx.set("anno_id", approx_info_map[i][1])
					add_approx.set("magnitude", magnitude)
					add_approx.set("format", format)
					add_approx.set("vg", str(vg))
					add_approx.set("va", str(va))
					add_approx.set("hedge_str", hedge_str)
					add_approx.set("text", curr_text)
					add_approx.set("direction", direction)
					add_approx.set("unit", unit)
					add_approx.set("nearest_major_ticks", major_ts)
					add_approx.set("nearest_minor_ticks", minor_ts)

			
			def build_relex_tags(parent_tag):

				# TODO: vg, va, format (might not be necessary)
				relative_expressions = ET.SubElement(parent_tag, "relative_expressions")

				for i in range(1, len(relex_id_map)+1):					
					hedge_str, vg, va, v1, v2 = "", "##REVIEW",  "##REVIEW", None, None
					relex_type, magnitude = None, ""
					
					# print(relex_info_map[i+1], "*********************")
					quad = relex_info_map[i][0]  ## get the quadriplpes 
					(curr_label, curr_start, curr_end, curr_text) = quad

					label_bound_sentence_key = relex_id_map[quad][1]
					label_bound_sentence_index = relex_id_map[quad][2]

					prev1 = get_previous(1, label_bound_sentence_key, label_bound_sentence_index)
					prev2 = get_previous(2, label_bound_sentence_key, label_bound_sentence_index)
					prev3 = get_previous(3, label_bound_sentence_key, label_bound_sentence_index)

					next1 = get_next(1, label_bound_sentence_key, label_bound_sentence_index)
					next2 = get_next(2, label_bound_sentence_key, label_bound_sentence_index)
					
					if next1 and next1[1] != curr_end+1:
						next1 = None
						# print("next1" * 8)
					if next2 and (not next1 or  next1[2]+1 != next2[1]):
						next2 = None
						# print("next2" * 8)

					# set relex_type 
					if curr_label.find("_add_") != -1:
						relex_type = "add"
					elif curr_label.find("_mul_") != -1:
						relex_type = "mul"

					# set hedge string (currently only checks previous 2 indexes)
					hedge_quad = None
					if prev1 and prev1[0] == "<y_axis_approx>":
						hedge_quad = prev1
					elif prev2 and prev1[2] == "<y_axis_approx>":
						hedge_quad = prev2
					
					if hedge_quad is not None:
						hedge_str = hedge_quad[3]


					# set v1 and v2:
					sorted_y_values = sorted(corpus['y'], reverse=True)  
					y_ordered_index = {
						"highest":0, "Scnd":1, "3rd":2, "4th":3, "5th":4, "6th":5, "least":len(sorted_y_values)-1
						}
					
					# print(sorted_y_values)
					# set v1 and v2
					for key in y_ordered_index:
						#set v1
						if not v1 and curr_label.find("v1="+key) != -1:
							v1_key = key
							v1 = sorted_y_values[y_ordered_index[key]]
						# set v2
						if not v2 and curr_label.find("v2="+key) != -1:
							v2_key = key
							v2 = sorted_y_values[y_ordered_index[key]]
					
					if v1 and v2:
						if relex_type == "mul":
							va = v1/v2 
							relex_val_dict = {
									'half': 1/2, 'halved': 1/2, 'halving': 1/2, 'halves': 1/2, 
									'double': 2, 'twice': 2,  'doubled': 2, 'doubles': 2,  
									'third': 1/3, '1/3': 1/3, 'one-third': 1/3,
									'quadruple': 4, 'quadrupled': 4, 'four': 4
									}
							checker = None
							# assign vg							
							curr_text_list = curr_text.split()
							if len(curr_text_list) == 1:
								
								# TODO: cases dealing with percentages
								if next1 and next1[0] == "<y_axis_inferred_label>" and next1[2] == curr_end+1:
									checker = True
									pass

								# if its a digit convert directly (with preference for integer) otherwise check the value dictionary
								elif is_digit(curr_text_list[0]):
									vg = float(curr_text_list[0])
									if vg.is_integer():
										vg = int(vg)
								elif curr_text_list[0] in relex_val_dict:
									vg = relex_val_dict[curr_text_list[0]]

								
							elif len(curr_text_list) == 2:
								if curr_text_list[1] == "times":
									if is_digit(curr_text_list[0]):
										vg = float(curr_text_list[0])
										if vg.is_integer():
											vg = int(vg)
									elif curr_text_list[0] in relex_val_dict:
										vg = relex_val_dict[curr_text_list[0]]
					

						elif relex_type == "add":
							va = v1 - v2
							vg_sign = None
							if next1:
								if next1[0] == "<y_axis_trend_down>"  or (next1[0] ==  "<y_magnitude>" and next2 and next2[0] == "<y_axis_trend_down>"):
									vg_sign = -1
									if next1[0] ==  "<y_magnitude>":
										magnitude = next1[3]

								elif next1[0] == "<y_axis_trend_up>"  or (next1[0] ==  "<y_magnitude>" and next2 and next2[0] == "<y_axis_trend_up>"):
									vg_sign = 1
									if next1[0] ==  "<y_magnitude>":
										magnitude = next1[3]
								# did[next2[0]] +=1

							# assign vG + ##check
							# onl for digit-convertible
							if vg_sign and is_digit(curr_text):
								vg = float(curr_text)
								if vg.is_integer():
									vg = int(vg)

								# assign appropriate sign
								vg = str( vg_sign * vg)  + "##CHECK"
								# print(curr_text, vg, va, magnitude, "-------------------------")

							# print(curr_text, vg, va, "-------------------------")
					



					global a_vg
					global a_vg_total
					if vg == "##REVIEW":
						a_vg += 1
						# print(quad)
					a_vg_total += 1

					
					# #TODO: delete - used for debugging
					# this_id = topic_story_id + "-" + str(i)
					# if this_id ==  "02_02-18-1":
					# 	print("************************************************")
					# 	print("************************************************")

					# 	print(curr_text)
					# 	print(vg_sign)

					# 	print("************************************************")
					# 	print("************************************************")
					
					add_relex = ET.SubElement(relative_expressions, "relex") 

					
					add_relex.set("id", topic_story_id + "-" + str(i)) # label id
					add_relex.set("anno_id", relex_info_map[i][1]) 
					# add_relex.set("format", format)
					add_relex.set("type", relex_type)
					add_relex.set("rel_vg", str(vg))
					add_relex.set("rel_va", str(va))
					add_relex.set("v1", str(v1))
					add_relex.set("v2", str(v2))
					add_relex.set("magnitude", magnitude)
					add_relex.set("hedge_str", hedge_str)
					add_relex.set("text", curr_text)

			# TODO: 
			# 1. hedge  (hedge_type) currently only define for approx and relex. Generalise to non approx
			# 2. create tag for to tell if the hedged expression is rounded e.g rounded =
			# 3. rethink expr_id  (mabybe use annotation ID). Non rounded values don't have ids otherwise
			#		as custom tags werent created for them
			def build_hedges_tags(parent_tag):
				hedges = ET.SubElement(parent_tag, "hedges")
				
				for i in range(1, len(hedge_id_map)+1):		
					quad = hedge_info_map[i][0]  ## get the quadriplpes 
					(curr_label, curr_start, curr_end, curr_text) = quad

					hedge_str, hedge_type, expr_id = curr_text, "##REVIEW",  "##REVIEW" 
					approx_or_relex_hedge = None


					label_bound_sentence_key = hedge_id_map[quad][1]
					label_bound_sentence_index = hedge_id_map[quad][2]

					next1 = get_next(1, label_bound_sentence_key, label_bound_sentence_index)
					next2 = get_next(2, label_bound_sentence_key, label_bound_sentence_index)
					next3 = get_next(3, label_bound_sentence_key, label_bound_sentence_index)

					# check to see that an expression is an absolute expression (especially relevant to non-approximated absex)
					def isAbsEx(quad_arg):
						if quad_arg is None:
							return False
						if quad_arg[0].find("y_axis_") != -1 and quad_arg[0].find("_value") != -1:
							return True
						return False

					hedged_expr = None
					if next1 in approx_id_map or next1 in relex_id_map or isAbsEx(next1):
						hedged_expr = next1
					elif next2 in approx_id_map or next2 in relex_id_map or isAbsEx(next2):
						hedged_expr = next2
					elif next3 in approx_id_map or next3 in relex_id_map or isAbsEx(next3):
						hedged_expr = next3
					
					# set hedge_type
					if hedged_expr is not None: 
						if hedged_expr in approx_id_map or isAbsEx(hedged_expr):
							hedge_type = "abs_ex"
						elif hedged_expr in relex_id_map:
							hedge_type = "rel_ex" 

					# set expr_id


					add_hedge = ET.SubElement(hedges, "hedge") 

					
					add_hedge.set("id", topic_story_id + "-" + str(i)) # label id
					add_hedge.set("anno_id", hedge_info_map[i][1]) 
					# add_relex.set("format", format)
					add_hedge.set("hedge_str", hedge_str)
					add_hedge.set("hedge_type", hedge_type)
					add_hedge.set("expr_id", expr_id)


			numex = ET.SubElement(story, "numex")	

			# #TODO: delete 
			# topic_tag = ET.SubElement(numex, "topic_tag")
			# topic_tag.set("title", corpus['title'])
			# cont = ET.SubElement(numex, "cont")
			# cont.text = " ".join(d1["desc_tokens"])
			
			build_approx_tags(numex)	
			build_relex_tags(numex)	
			build_hedges_tags(numex)
			
			# print(did)

			# print(my_cnt)
	#s = prettify(story)
	s = prettify(main_root)

	return s




def save_into_file(xml_populated, name_out):
	#TODO: check encoding
	# myfile = open(name_out+".xml", "w", encoding="utf-8") 
	myfile = open(name_out+".xml", "w", encoding="ISO-8859-1") 
	myfile.write(s)
	myfile.close()
	return None




if __name__ == "__main__":
	from collections import Counter
	did = Counter()

	my_cnt = 0
	
	a_vg = 0
	a_vg_total = 0
	# annotations is the CLI argument - name of labeled file
	# annotations = args["labeled_file"]

	s = make_xml_tree() # xml subtree for a single chart

	# add s to S

	save_into_file(s, version_dir+"chart_summaries_b01_toktest2")
	print(a_vg, a_vg_total, a_vg/a_vg_total)




