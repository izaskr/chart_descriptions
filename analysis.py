
import json
import numpy as np
import argparse
import sys
from itertools import combinations
import mord
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
#parser.add_argument("-data", required=True)
parser.add_argument("-labeled_file", required=True, help="labeled file with chart descriptions")
parser.add_argument("-show", required=True, nargs="+", help="write one or both: info (this will show info extracted from chart json), map (this will show the mapped values from json given labels in the labeled_file)")
args = vars(parser.parse_args())

annotations = args["labeled_file"]
show = args["show"]
show_info, show_mapping = False, False
if "info" in show: show_info = True
if "map" in show: show_mapping = True
if set(show).issubset({"info","map"}) == False: sys.exit("Use 'info' and/or 'map' as value for -show")


version_dir = "corpora_v01/" # Rudy's version
#version_dir = "corpora_v02" # Iza's relabeled
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
		x_major_ticks = k["general_figure_info"]["x_axis"]["major_ticks"]["values"][:-(len(x))] # why double
		len_yt = int(len(k["general_figure_info"]["y_axis"]["major_ticks"]["values"]) / 2)
		y_major_ticks = k["general_figure_info"]["y_axis"]["major_ticks"]["values"][:-len_yt]

		xy[i+1] = {"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, "x_axis_label_name": x_axis_label_name, "x_major_ticks":x_major_ticks,"y_major_ticks":y_major_ticks}


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




def analyze_coverage(annotations, calculated):
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

	# covered is a set of labels we can provide a value for deterministically from the raw plot data
	# TODO check for duplicates
	covered = {"<x_axis_label_highest_value>", "<x_axis_label_Scnd_highest_value>", "<x_axis_label_3rd_highest_value>", "<x_axis_label_4th_highest_value>","<x_axis_label_5th_highest_value>","<x_axis_label_least_value>", "<y_axis_highest_value_val>", "<y_axis_Scnd_highest_val>", "<y_axis_3rd_highest_val>", "<y_axis_label_4th_highest_value_val>","<y_axis_label_5th_highest_value_val>", "<y_axis_least_value_val>", "<y_axis_highest_value_val>", "<y_axis_Scnd_highest_val>", "<y_axis_3rd_highest_val>", "<y_axis_4th_highest_val>","<y_axis_5th_highest_val>", "<y_axis_least_value_val>", "<x_axis_label_count>", "<y_axis_inferred_value_mul>", "<y_axis_inferred_value_add>","<y_axis_inferred_label>","<x_axis>", "<y_axis>"}

	# TODO: approximations and rounding


	not_covered_count = 0
	total_label_count = 0
	with open(version_dir + annotations, "r", encoding="ISO-8859-1") as f:
		end_desc = False
		for line in f:
			#print(line)
			#line = line.encode('utf-8').strip() # a.encode('utf-8').strip()
			line2 = line.split()
			line3= " ".join(line2)

			if line2 == []:
				end_desc = True

			if len(line2) == 1 and line2[0] == '"':
				input("Press Enter to show next plot description with labels and automatic mapping")
				

			if len(line2) > 1 and line2[-1].startswith("<"):

				label = line2[-1]
				total_label_count += 1
				if label not in covered:

					print(line3 + "\t NOT COVERED")
					not_covered_count += 1
				if label in covered:

					if label.startswith("<x_axis_label_") and label.endswith("_value>"):
						val = "NOT COVERED"
						cur = label_name_pairs_x
						if label == "<x_axis_label_highest_value>":
							val = cur["<x_axis_label_highest_value>"]
							
						if "Scnd" in label:
							val = cur["<x_axis_label_Scnd_highest_value>"]
						if "3rd" in label:
							val = cur["<x_axis_label_3rd_highest_value>"]

						if "4th" in label:
							val = cur["<x_axis_label_4th_highest_value>"]

						if "5th" in label:
							val = cur["<x_axis_label_5th_highest_value>"]

						if "least" in label:
							val = cur["<x_axis_label_least_value>"]
						
						if val == "NOT COVERED": not_covered_count += 1
						print(line3, "\t FROM DATA JSON: ", val)

					if label.startswith("<y_axis_") and label.endswith("_val>"):
						val = "NOT COVERED"
						cur = label_value_pairs_y
						if label == "<y_axis_highest_value_val>":
							val = cur["<y_axis_highest_value_val>"]
							
						if "Scnd" in label:
							val = cur["<y_axis_Scnd_highest_val>"]
						if "3rd" in label:
							val = cur["<y_axis_3rd_highest_val>"]

						if "4th" in label:
							val = cur["<y_axis_4th_highest_val>"]

						if "5th" in label:
							val = cur["<y_axis_5th_highest_val>"]

						if "least" in label:
							val = cur["<y_axis_least_value_val>"]

						if val == "NOT COVERED": not_covered_count += 1
						print(line3, "\t FROM DATA JSON: ", val)


					if label == "<y_axis_inferred_value_mul>":

						#val = ("\t UNABLE TO FIND THE RIGHT PAIR", differences["times"])
						val1, val2 = "", ""
						word_num = {"half":0.5, "third":0.3, "quarter":0.25, "fourth":0.25, "fifth":0.2}
						if line2[0] in word_num:
							c_float = word_num[line2[0]]
						#if line2[0] == "half":
							#c_float = 0.5
							cur = differences["times"]["pairs_float"]
							for pair, k in cur.items():
								if c_float == round(k,1):
									pairk = " ".join([str(pair),str(k)])
									val1 += " PAIR - CALCULATED "+pairk

							cur = differences["times"]["pairs_float_reverse"]
							for pair, k in cur.items():
								#print(pair,k)
								if c_float == round(k,1):
									pairk = " ".join([str(pair),str(k)])
									val2 += " PAIR - CALCULATED "+pairk

						if is_int(line2[0]) == True: # TODO: either fix statement or remove

							c = int(line2[0])
							cur = differences["times"]["pairs_int"]
							for pair, k in cur.items():

								if c == k:
									val = ("PAIR - CALCULATED ", pair, k)
						val = val1 + val2
						if not val:
							val = ("\t UNABLE TO FIND THE RIGHT PAIR", differences["times"])
							not_covered_count += 1
						print(line3, val)

					if label == "<y_axis_inferred_value_add>":
						print(line3, differences["plus"])

					if label == "<x_axis_labels_count>":
						print(line3, "\t FROM DATA JSON", label_count)

					if label == "<y_axis_inferred_label>": 
						print(line3, "\t FROM DATA JSON", y_axis_unit_name)

					if label == "<y_axis>":
						print(line3, "\t FROM DATA JSON", y_axis_unit_name)

					if label == "<x_axis>":
						print(line3, "\t FROM DATA JSON", x_axis_label_name)

			else:
				print(line3)

	print("# not covered label mapping %s , of total %s label mappings" % (not_covered_count, total_label_count))
	return None



data = descriptions_files_json[annotations][0] + "_annotations2.json"

data_ext = get_x_y(data)

title = descriptions_files_json[annotations][1]

d = {}
for i, d in data_ext.items():
	if d["title"] == title:
		corpus = d
#print(corpus)
results = get_stat_info(corpus)
if show_mapping:
	analyze_coverage(annotations, results)


