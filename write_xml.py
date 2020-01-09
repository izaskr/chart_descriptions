
"""
Rewrite annotated vertical description corpora into XML following the InScript schema
"""

import json
import numpy as np
import argparse
from itertools import combinations
import spacy
from spacy.lang.en import English
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
#parser.add_argument("-data", required=True)
#parser.add_argument("-labeled_file", required=True, help="file with annotated description")
parser.add_argument("-labeled_file", required=False, help="file with annotated description")
#parser.add_argument("-img_id", required=True, help="int denoting graph id")
#parser.add_argument("-csv", required=True, help="csv file from Prolific")
args = vars(parser.parse_args())



nlp_web = spacy.load("en_core_web_sm") # used only for tokenization
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


version_dir = "corpora_v02/" # Rudy's version
descriptions_files = ["Money_spent_on_higher_education.txt", 
         "Number_of_top_Unis.txt",
         "gender_pay_gap.txt",
         "women_representation_in_different_departments.txt",
         "women_representation_in_different_sectors.txt",
         "what_causes_obesity.txt",
         "how_do_young_people_spend_their_evenings.txt",
         "what_do_students_choose_to_study.txt",
         "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
         "example_Median_salary_of_women.txt"]

descriptions_files = ["Money_spent_on_higher_education.txt", 
         "Number_of_top_Unis.txt",
         "gender_pay_gap.txt",
         "women_representation_in_different_departments.txt",
         "women_representation_in_different_sectors.txt",
         "what_causes_obesity.txt",
         "how_do_young_people_spend_their_evenings.txt",
         "what_do_students_choose_to_study.txt",
         "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
         "example_Median_salary_of_women.txt"]

description_files_order = ["gender_pay_gap.txt", "example_Median_salary_of_women.txt",
   "how_do_young_people_spend_their_evenings.txt", "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
   "Money_spent_on_higher_education.txt", "Number_of_top_Unis.txt", "what_causes_obesity.txt", 
   "what_do_students_choose_to_study.txt", 
   "women_representation_in_different_departments.txt", "women_representation_in_different_sectors.txt"]

topic_image_id = [("gender_paygap", "01", "nominal"), ("salary_women", "02", "interval"), ("evenings", "03", "nominal"), ("salary_se_degree", "04", "ordinal"), ("money_he", "05", "nominal"), ("top_unis", "06", "nominal"), ("obesity_cause", "07", "nominal"), ("study_prog", "08", "nominal"), ("women_dept", "09", "nominal"), ("women_sect", "10", "nominal") ]


# descriptions_files_json is a dict, where the keys are files names (as given by Rudy) with descriptions, and the values are tuples - its first element is the name of the json file with raw plot data, its second element is the title of the plot (it's unique for each plot)
descriptions_files_json = {"Money_spent_on_higher_education.txt":("train1","Money Spent on Higher Education in Year 2010"), 
         "Number_of_top_Unis.txt":("train1","Number of Top 100 Universities in Each Continent"),
         "gender_pay_gap.txt":("train1","Gender Pay Gap"),
         "women_representation_in_different_departments.txt":("train1","Women Representation in Different University Departments"),
         "women_representation_in_different_sectors.txt":("train1","Women Representation in Different Sectors"),
         "what_causes_obesity.txt":("val1","What causes Obesity"),
         "how_do_young_people_spend_their_evenings.txt":("val1","How do Young People Spend their Evenings"),
         "what_do_students_choose_to_study.txt":("train1","What do Students choose to study?"),
         "median_salary_per_year_for_se_with_respect_to_their_degrees.txt":("val2","Median Salary Per Year For Software Engineers with Respect to their Degree"),
         "example_Median_salary_of_women.txt":("train1","Median Salary of Women Per Year")}


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

		xy[i+1] = {"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, "x_axis_label_name": x_axis_label_name}


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


def is_int(s):
	try: 
		int(s)
		return True
	except ValueError:
		return False

def is_float(s): # currently not used, because it returns True for "6" although I'd like a False here
	try:
		float(s)
		return True
	except ValueError:
		return False




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
					temp_doc = nlp_web(temp)
					temp_tokens = [t.text for t in temp_doc]
					temp_join = " ".join(temp_tokens)
					i_end = i_token + len(temp_tokens) - 1
					desc_tokens += temp_tokens
					desc_labels.append((label, i_token, i_end, temp_join))
					i_token = i_end




			if len(line2) == 1 and line2[0] == '"' and previous_line_empty == False:
				if desc_tokens:
					desc = " ".join(desc_tokens)
					doc = nlp(desc)
					#for sent in doc.sents:
					doc_tokens = [t.text for t in doc]
					#print(desc_tokens,"\n" , "\n", doc_tokens, "\n \n", desc_labels, "\n",desc, "\n", len(desc_tokens), len(doc))

					for_xml[i_desc] = {"desc_tokens":desc_tokens,"doc_tokens":doc_tokens,"desc_labels":desc_labels}
					i_desc += 1
					
				desc_tokens, desc_labels, i_token = [], [], 0
				
				
	return for_xml




def prettify(elem):
	"""Return a pretty-printed XML string for the Element.
	"""
	rough_string = ET.tostring(elem, 'utf-8')
	reparsed = parseString(rough_string)
	return reparsed.toprettyxml(indent="\t")


def make_xml_tree():
	""" d is a dictionary, indices as keys, dict as values with keys desc_tokens, doc_tokens, desc_labels """

	main_root = ET.Element("summaries")

	for tri in zip(description_files_order, topic_image_id):
		annotations = tri[0]
		chart_topic, chart_id, chart_x_type = tri[1][0], tri[1][1], tri[1][2]

		data = descriptions_files_json[annotations][0] + "_annotations3.json"
		data_ext = get_x_y(data)
		title = descriptions_files_json[annotations][1]
		d = {}
		for i, d in data_ext.items():
			if d["title"] == title:
				corpus = d

		results = get_stat_info(corpus)
		for_xml = collect_parse(annotations, results)


		desc_ids = list(for_xml.keys())
		#print(desc_ids)

		topic_root = ET.SubElement(main_root,"topic")
		topic_root.set("topic", chart_topic)
		topic_root.set("topic_id", chart_id) # the name of the bar chart image file
		topic_root.set("scenario", chart_x_type)

		for story_id in desc_ids: # iterate over all descriptions of a single chart
			d1 = for_xml[story_id]

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
	
			boundaries = []
			sentence_lengths = []
			for sent in doc.sents:
				i += 1 # sentence index
				j = 0 # index for tokens within a single sentence
				add_sent = ET.SubElement(sentences, "sentence")
				add_sent.set("id", topic_story_id + "-" + str(i))
				for token in sent:
					k += 1 # index for tokens within the entire description
					j += 1
					add_token = ET.SubElement(add_sent, "token")
					add_token.set("content", token.text)
					add_token.set("content_fix", token.text)
					add_token.set("id", topic_story_id + "-" + str(i)+"-"+str(j))
				boundaries.append(k) # last token index of a sentence, starting with 1
				sentence_lengths.append(j)


			annotations = ET.SubElement(story, "annotations")
			events = ET.SubElement(annotations, "events")

			label_bound = {i:[] for i in range(1,len(boundaries)+1)}

			for (label, start, end, text_str) in label_info:
				for n,boundary in enumerate(boundaries):
					if end <= boundary:
						label_bound[n+1].append((label, start,end, text_str))
						break

			#print(label_bound)

			c = 0
			for i_sent, triplets in label_bound.items():
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
						#print("old start, new start, old end, new end, len prev", start, start2, end, end2, sentence_lengths[i_sent-1])
			
					add_label = ET.SubElement(events, "label")
					add_label.set("from", topic_story_id + "-" + from_value)
					add_label.set("to", topic_story_id + "-" + to_value)
					add_label.set("id", topic_story_id + "-" + str(c)) # label id
					add_label.set("name", label[1:-1])
					add_label.set("text", text_str)
					add_label.set("type", "event")

	#s = prettify(story)
	s = prettify(main_root)

	return s




def save_into_file(xml_populated, name_out):
	myfile = open(name_out+".xml", "w", encoding="utf-8") # TODO change name given title (arg) and ID
	myfile.write(s)
	myfile.close()
	return None




if __name__ == "__main__":

	# annotations is the CLI argument - name of labeled file
	# annotations = args["labeled_file"]

	s = make_xml_tree() # xml subtree for a single chart

	# add s to S

	save_into_file(s, version_dir+"chart_summaries_b01")





