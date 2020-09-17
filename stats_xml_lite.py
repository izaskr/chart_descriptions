#!usr/bin/python3


import argparse
import xml.etree.ElementTree as ET
from string import punctuation
from collections import defaultdict, Counter
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-xml", required=False, help="xml corpus with chart summaries and labels", default="/home/iza/chart_descriptions/data/chart_summaries_b01_toktest2.xml")
parser.add_argument("-hdir", required=False, help="path to home dir of the project", default="/home/CE/skrjanec/chart_descriptions/")
args = vars(parser.parse_args())

xml_file = args["xml"]
home_dir = args["hdir"]


"""
TODO
for each chart, check the most frequent order of entities (bar names) given their order on the x axis

entities (bars) and relations (their heights; relations between bars)
"""



# bar names
anno_size_x = ["<x_axis_label_highest_value>", "<x_axis_label_Scnd_highest_value>",
 	"<x_axis_label_3rd_highest_value>",
 	"<x_axis_label_4th_highest_value>","<x_axis_label_5th_highest_value>","<x_axis_label_least_value>"]

# exact bar heights
anno_size_y = ["<y_axis_highest_value_val>", "<y_axis_Scnd_highest_val>", "<y_axis_3rd_highest_val>",
 	"<y_axis_4th_highest_val>","<y_axis_5th_highest_val>", "<y_axis_least_value_val>"]

# exact bar heights : approximated bar heights
exact_approx = {"<y_axis_highest_value_val>":"<y_axis_inferred_highest_value_approx>",
"<y_axis_Scnd_highest_val>":"<y_axis_inferred_Scnd_highest_value_approx>",
 "<y_axis_3rd_highest_val>":"<y_axis_inferred_3rd_highest_value_approx>",
 "<y_axis_4th_highest_val>":"<y_axis_inferred_4th_highest_value_approx>",
 "<y_axis_5th_highest_val>":"<y_axis_inferred_5th_highest_value_approx>",
 "<y_axis_least_value_val>":"<y_axis_inferred_least_value_approx>"}

approx_exact = {v:k for k,v in exact_approx.items()}

# mul and add
# if "_mul_v1=" in label or "_add_v1=" in label

# group_y for height; group_X (X != "y") for bar names

# x_axis, y_axis

# maybe include: x_axis_labels_count, other_operation

# neutral order: no a, b or c in name
index2bar_count = {"01":3, "02":4, "03": 4, "04":4, "05":4, "06": 4, "07": 3, "08": 4, "09": 5, "10": 4,
					"11": 5, "12": 5, "13": 5, "14": 6, "15": 6, "16": 6, "17": 6, "18": 5}


# barCount : list_of_lists listing entity sequences
bar_count2sequences = {k: [] for k in index2bar_count.values()}


def get_basics(corpus, individual_chart=None):
	"""
	corpus : str : path to the xml corpus
	individual_chart : bool or str : chart ID of a chart for which we request an analysis as to avoid parting the entire dataset

	"""
	tree = ET.parse(corpus)
	root = tree.getroot()

	topic_entity_seq = {}

	for topic in root:
		chart_id = topic.attrib["topic_id"]
		topic_id = chart_id[:2]
		nbars = index2bar_count[topic_id]

		if individual_chart and individual_chart == chart_id:
			pass
		else:
			continue

		print(".... parsing chart id", chart_id)


		# a - proportional, b - inverse, c - one bar emphasis

		#if "a" in chart_id or "b" in chart_id or "c" in chart_id:
		#	#print("to ignore", chart_id)
		#	continue


		#if "c" not in chart_id: # inverse
		#	continue

		for story in topic:
			# annotations = story[1]
			events = story[1][0]
			summary_seq = []
			for e in events:
				label = "<"+ e.attrib["name"] + ">"
				#print(label)
				if label in anno_size_x or label in anno_size_y:
					summary_seq.append(label[1:-1])
					continue
				if label in approx_exact:
					summary_seq.append(approx_exact[label][1:-1])
					continue
				if "_mul_" in label or "_add_" in label or "group_" in label:
					print(" ... will be other",label)
					summary_seq.append("other")
			if len(summary_seq) == 0:
				print("this summary has no entities",story.attrib["story_id"])
			if len(summary_seq) > 0:
				bar_count2sequences[nbars].append(summary_seq)
			summary_seq = []

	for bc, seqs in bar_count2sequences.items():
		print("# BARS",bc)
		#print("Longest seq has this many entities", sorted([len(s) for s in seqs]))
		#print("The average number of entities", sum([len(s) for s in seqs])/len(seqs)) # less than bc / 2



	return bar_count2sequences


substitution = {"y_axis": "YLABEL",
  "x_axis_label_least_value": "XLEAST",
  "x_axis_label_highest_value": "XHIGHEST",
  "x_axis_label_Scnd_highest_value": "XSECOND",
  "y_axis_least_value_val": "YLEAST",
  "y_axis_Scnd_highest_val": "YSECOND",
  "y_magnitude": "YMAG",
  "x_axis_label_3rd_highest_value": "XTHIRD",
  "y_axis_highest_value_val": "YHIGHEST",
  "y_axis_inferred_label": "YUNIT",
  "x_axis": "XLABEL",
  "y_axis_3rd_highest_val": "YTHIRD",
  "y_axis_5th_highest_value_val": "YFIFTH",
  "y_axis_5th_highest_val": "YFIFTH",
  "y_axis_inferred_highest_value_approx": "YHIGHESTAPPROX",
  "x_axis_label_4th_highest_value": "XFOURTH",
  "x_axis_label_5th_highest_value": "XFIFTH",
  "y_axis_inferred_least_value_approx": "YLEASTAPPROX",
  "y_axis_4th_highest_val": "YFOURTH",
  "x_axis_labels_count": "COUNT",
  "y_axis_inferred_3rd_highest_value_approx": "YTHIRDAPPROX",
  "y_axis_inferred_5th_highest_value_approx": "YFIFTHAPPROX",
  "x_axis_range_start": "XSTART",
  "x_axis_labels": "BARNAMES",
  "y_axis_inferred_Scnd_highest_value_approx": "YSECONDAPPROX",
  "y_axis_inferred_4th_highest_value_approx": "YFOURTHAPPROX",
  "x_interval": "INTERVAL",
  "slope_x_value": "SLOPEX",
  "slope_y_value": "SLOPEY"
}

def collapse(bc, list_of_lists):
	longest = max([len(l) for l in list_of_lists])
	# position as key, dict as value; the dict will be populated with entities and their counts
	position2counter = {i: {} for i in range(1, longest+1)}
	for sequence in list_of_lists:
		for i, entity in enumerate(sequence):

			if entity in substitution:
				entity = substitution[entity]
			else:
				print("no substitution", entity)
				entity = "other"

			j = i+1
			if entity in position2counter[j]:
				position2counter[j][entity] += 1
			else:
				position2counter[j][entity] = 1

	return position2counter





def create_csv(barcount_sequences, name=""):
	# bar count as key, list of lists of stings as value

	# for each bar count, collapse given the position in sequence
	for bc, sequences in barcount_sequences.items():
		if len(sequences) == 0:
			print("no data for this condition", bc)
			continue

		pos2count = collapse(bc, sequences)
		print(pos2count)

		# create new var to write rows in csv
		rows = []
		for position, entityCountDict in pos2count.items():
			for entity, count in entityCountDict.items():
				rows.append([entity, str(position), str(count)])

		fname = home_dir + "stats_analysis/" +  str(bc) + "positions_" + name + ".csv" # NOTE name
		# writing to csv file
		with open(fname, 'w') as csvfile:
			# creating a csv writer object
			csvwriter = csv.writer(csvfile)

			# writing the fields
			csvwriter.writerow(["Entity", "Position", "Count"])

			# writing the data rows
			csvwriter.writerows(rows)

	#input("enter for next")


def plot_scatter(condition):
	""" condition is either neutral, a, b or c as a string """

	nbars = ["3", "4", "5", "6"]
	ending = ".csv"

	if condition == "a" or condition == "b":
		nbars = ["3", "5", "6"]
		ending = "_" + condition + ".csv"
	if condition == "c":
		nbars = ["3", "4", "5"]
		ending = "_" + condition + ".csv"

	e = {"04_02c": "3", "05_01c":"4", "09_02c":"4", "10_02c":"5", "11_02c":"5"}
	if condition in e:
		nbars = [e[condition]]
		ending = "_" + condition + ".csv"

	for nbar in nbars:
		csv_name = home_dir + 'stats_analysis/' + nbar + 'positions' + ending
		# open csv and generate a scatter plot
		df = pd.read_csv(csv_name)
		#print(df.columns)

		sns.set_style("whitegrid")
		#s = "Entity count given their position in summaries (%s-bar charts)" % (nbar)
		sns_plot = sns.catplot(x="Position", y="Count", hue="Entity", kind="swarm", data=df)
		#sns_plot.set_title(s)
		sns_plot.savefig(nbar + "_position_" + condition + ".png")




if __name__ == "__main__":

	#tes = get_basics(xml_file)
	c_highest = {"01_02c", "05_01c", "11_02c"}
	c_lowest = {"09_02c", "10_02c"}
	all_c = {"01_02c", "05_01c", "11_02c", "09_02c", "10_02c"}
	for chid in c_highest:
		tes = get_basics(corpus=xml_file, individual_chart=chid)
		create_csv(tes, chid)
		plot_scatter(chid)
	#create_csv(tes)
	#for cnd in ["a", "b", "c"]: # neutral has been done
	#	plot_scatter(cnd)

