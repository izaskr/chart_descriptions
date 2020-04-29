"""

"""

import os
import allennlp
from allennlp.predictors.predictor import Predictor


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

data_path = "" # local "/home/iza/chart_descriptions/coref/data/"
files = []

for file_name in os.listdir(data_path):
	if file_name.endswith(".txt"):
		files.append(file_name)


def open_resolve_write(fname):
	""" fname is a string : name of a file with summaries """

	current_summary = []
	i_summary = 0	
	with open(fname, "r") as f:
		for line in f:
			line = line.split()

			if len(line) > 0:
				if line[0] == "<end_of_description>":
					current_text = " ".join(current_summary)

					# resolve coreferences
					#predictor.predict(document="The woman reading a newspaper sat on the bench with her dog.")
					# returns a list of tuples
					clusters = predictor.predict(document=current_text)
					
					current_summary = []

					# write into file: i_summary, current text and clusters
					print(i_summary, current_text, clusters, "\n", "\n")

				if line[0] == "<start_of_description>":
					i_summary += 1
				
				else: # a line of token(s), possibly with a label or ## comment
					if len(line) == 1:
						current_summary.append(line[0])
						continue

					first_ch = [s[:1] for s in line]
					first2_ch = [s[:2] for s in line]

					if "<" in first_ch:
						inx = first_ch.index("<")
						current_summary += line[:inx]
						continue

					if "##" in first2_ch:
						inx = first_ch.index("##")
						current_summary += line[:inx]
						continue

						

					
	return None


for f in files:
	open_resolve_write(f)
	input("ENTER for next file")



					
