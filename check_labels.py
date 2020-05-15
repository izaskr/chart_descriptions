"""
Check .txt files for labels: bracketing with <> and frequency (typos)
"""

import os
import json


batch01_path = "/home/iza/chart_descriptions/corpora_v02/"

batch02_path = "/home/iza/chart_descriptions/corpora_v02/run2_chart_summaries/auto_labeled_revised/"
labels = {}

def check_dir(path):
	#labels = {}
	
	symbols = {"x", "y", "=", "_"}
	# _ topic separator interpretation
	symbols = {"_", "topic", "separator", "interpretation"}

	for f in os.listdir(path):
		if f.endswith(".txt") and "README" not in f and "Issues" not in f:

			with open(path+f, "r", encoding="utf-8") as summaries:

				for line in summaries:
					str_line = line
					line = line.split()

					if line:
						last = set(line[-1])
						if last.intersection(symbols) and not last.intersection({"<",">"}):
							print(path+f, "\t", line)

						if ">" in str_line and "<" not in str_line:
							print(path+f, "\t", str_line)

						if "<" in last and ">" in last and "description" not in str_line:
							if line[-1] in labels:
								labels[line[-1]] += 1
							if line[-1] not in labels:
								labels[line[-1]] = 1

						if "<\u200bx_axis_labels_count\u200b" in str_line:
							print(path+f, "\t",line)

	# write label set and counts into file


def sort_show(d):
	new = open("label_frequencies.txt", "w", encoding="utf-8")
	sorted_d = sorted(d.items(), key=lambda p: p[1], reverse=True)
	for (label, count) in sorted_d:
		print("\n", label,"\t",count)
		new.write(label + "\t" + str(count) + "\n")

	new.close()


	

check_dir(batch01_path)
check_dir(batch02_path)
#print(labels)
sort_show(labels)




