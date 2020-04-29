"""

"""

import os
import allennlp
from allennlp.predictors.predictor import Predictor


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

data_path = "/home/CE/skrjanec/stuff_jones1/chart_descriptions/coref/data/" # local "/home/iza/chart_descriptions/coref/data/"
files = []

for file_name in os.listdir(data_path):
	if file_name.endswith(".txt"):
		files.append(file_name)


def open_resolve_write(fname, datap):
	""" fname is a string : name of a file with summaries """

	current_summary = []
	i_summary = 0
	n_sum_coref = 0

	fname_new = fname[:-4] + "_COREF.txt"
	new = open(fname_new, "w+")
	

	with open(fname, "r") as f:
		for line in f:
			line = line.split()

			if len(line) > 0:
				if line[0] == "<end_of_description>":
					current_text = " ".join(current_summary)

					# resolve coreferences
					#predictor.predict(document="The woman reading a newspaper sat on the bench with her dog.")
					# returns a list of tuples
					results = predictor.predict(document=current_text)
					
					
					clusters = results["clusters"]
					doc = results["document"] # tokenized text, a list of tokens
					# write into file: i_summary, current text and clusters
					if clusters:
						print(i_summary, doc,"\n", clusters, "\n", )
						new.write(str(i_summary) + " " + current_text + "\n")
						for cluster in clusters: # c is a list: start_index, end_index
							new.write("\t ---")
							for startend in cluster:
								ent=" ".join(doc[startend[0]:startend[1]+1])
								print("ENT", ent)
								new.write(ent + "\n")
						new.write("\n")
						new.write("\n")
						n_sum_coref += 1

					current_summary = []
					continue

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
						inx = first2_ch.index("##")
						current_summary += line[:inx]
						continue

						

	new.close()				
	return n_sum_coref

all_coref = 0
for f in files:
	coref_count = open_resolve_write(data_path+f, data_path)
	all_coref += coref_count
	#input("ENTER for next file")

print("NO. OF SUMMARIES WITH A COREF", all_coref)


					
