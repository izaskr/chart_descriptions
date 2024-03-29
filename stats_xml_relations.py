#!usr/bin/python3

"""
Give some corpus statistics about the given corpus:
	tokens
	words
	sentences

	label general frequency
	label in-summary frequency: on average
	are there cycles

"""

import argparse
import xml.etree.ElementTree as ET
from string import punctuation
from collections import defaultdict, Counter
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator

parser = argparse.ArgumentParser()
parser.add_argument("-xml", required=False, help="xml corpus with chart summaries and labels", default="corpora_v02/chart_summaries_b01.xml")
args = vars(parser.parse_args())

xml_file = args["xml"]


"""
TODO
for each chart, check the most frequent order of entities (bar names) given their order on the x axis

entities (bars) and relations (their heights; relations between bars)
"""

def get_plot_info(chart_name):
	"""
	open the corresponding json file with plot info to return info  about the bars and their heights anf their
	order on the x axis
	"""
	description_files_json = {"money_he":("train1","Money Spent on Higher Education in Year 2010"), 
         "top_unis":("train1","Number of Top 100 Universities in Each Continent"),
         "gender_paygap":("train1","Gender Pay Gap"),
         "women_dept":("train1","Women Representation in Different University Departments"),
         "women_sect":("train1","Women Representation in Different Sectors"),
         "obesity_cause":("val1","What causes Obesity"),
         "evenings":("val1","How do Young People Spend their Evenings"),
         "study_prog":("train1","What do Students choose to study?"),
         "salary_se_degree":("val2","Median Salary Per Year For Software Engineers with Respect to their Degree"),
         "salary_women":("train1","Median Salary of Women Per Year")}

	"""
	Function for extracting only the information needed to analyze the data, return a dict for every plot,
	including the values of x axis, y axis, plot type, plot title

	:param split_name: name of annotations.json file, which follows a certain format
	:type filename: str
	"""

	split_name = description_files_json[chart_name][0] + "_annotations3.json"
	# descriptions_files_json[annotations][0] + "_annotations3.json"

	with open(split_name) as f:
		data = json.load(f)

	xy = {}
	
	for i,k in enumerate(data):
		title = k["general_figure_info"]["title"]["text"]

		if title == description_files_json[chart_name][1]:
			x = k["models"][0]["x"]
			y = k["models"][0]["y"]
			x_order_info = k["models"][0]["x_order_info"]
			plot_type = k["type"] # "vbar_categorical"
			y_axis_unit_name = k["general_figure_info"]["y_axis"]["label"]["text"] # label of y from the plot
			x_axis_label_name = k["general_figure_info"]["x_axis"]["label"]["text"]
			x_major_ticks = k["general_figure_info"]["x_axis"]["major_ticks"]["values"][:-(len(x))] # why double
			len_yt = int(len(k["general_figure_info"]["y_axis"]["major_ticks"]["values"]) / 2)
			y_major_ticks = k["general_figure_info"]["y_axis"]["major_ticks"]["values"][:-len_yt]

			xy = {"title":title, "x":x, "y":y, "type":plot_type, "x_order_info": x_order_info, "y_axis_unit_name":y_axis_unit_name, "x_axis_label_name": x_axis_label_name, "x_major_ticks":x_major_ticks,"y_major_ticks":y_major_ticks}
			break
	#print(xy)
	return xy
	
	


def get_basics(corpus):
	tree = ET.parse(corpus)
	root = tree.getroot()

	token_count, word_count, label_count = defaultdict(int), defaultdict(int), defaultdict(int)
	storyc, sc, tc, wc = 0, 0, 0, 0
	multi_token_labels = 0
	within_summary_sequence, label_ids = [], [] # within.. is a list of labels within each story
						# label_ids is the same, but with ids of labels (as given in the xml)
	summaries_sequences, summaries_lid = [], [] # summaries_s.. is a list of within_summary_sequence of each story
						# summaries_lid is the same, but with ids of labels
	vocabulary = set() # lower case vocabulary
	min_length = 1000
	max_length = -1
	story_length = 0


	entities = {"x_axis_label_least_value", "x_axis_label_4th_highest_value", "x_axis_label_3rd_highest_value", "x_axis_label_Scnd_highest_value", "x_axis_label_highest_value"}

	# TODO add relation labels to the collection process
	relations = {"y_axis_least_value_val", "y_axis_Scnd_highest_val", "y_axis_highest_value_val", "y_axis_3rd_highest_val", "y_axis_inferred_highest_value_approx", "x_axis_label_4th_highest_value", "y_axis_inferred_least_value_approx", "y_axis_4th_highest_val", "y_axis_inferred_3rd_highest_value_approx", "y_axis_inferred_Scnd_highest_value_approx", "y_axis_inferred_value_mul_v1=highest_v2=least", "y_axis_inferred_value_mul_v1=highest_v2=Scnd", "y_axis_inferred_value_add_v1=highest_v2=least", "y_axis_inferred_value_mul_v1=least_v2=highest", "y_axis_inferred_value_mul_v1=Scnd_v2=least", "y_axis_inferred_value_mul_v1=Scnd_v2=3rd", "y_axis_inferred_value_mul_v1=3rd_v2=highest", "y_axis_inferred_value_add_v1=highest_v2=Scnd","y_axis_inferred_value_add_v1=highest_v2=3rd", "y_axis_inferred_value_add_v1=Scnd_v2=highest", "y_axis_inferred_value_mul_v1=least_v2=Scnd", "y_axis_inferred_value_mul_v1=least_v2=3rd","y_axis_inferred_value_mul_v1=highest_v2=3rd", "y_axis_inferred_value_mul_v1=Scnd_v2=highest","y_axis_inferred_value_mul_v1=4th_v2=Scnd", "y_axis_inferred_value_mul_v1=3rd_v2=least","y_axis_inferred_value_mul_v1=3rd_v2=Scnd", "y_axis_inferred_value_add_v1=least_v2=3rd","y_axis_inferred_value_add_v1=Scnd_v2=3rd", "y_axis_inferred_value_add_v1=3rd_v2=least"} # i did not include the ?> labels and <other_operation>

	topic_entity_seq = {}

	for topic in root:
		#print(topic.attrib["topic"])

		#plot_info = get_plot_info(topic.attrib["topic"])
		entities_within_topic, entities_within_story = [], [] # now extended to relations as well, but var name stays the same

		for story in topic:
			story_length = 0
			within_summary_sequence = []
			storyc += 1 # story (= description) counter

			# text, annotations are children of story; story[0] is text
			# text[0] is content, text[1] is sentences
			sentences = story[0][1]
			for sent in sentences:
				sc += 1
				for t in sent:
					tc += 1
					#print(t.attrib["content_fix"])
					token_fix = t.attrib["content_fix"].lower()
					vocabulary.add(token_fix)
					story_length += 1
					if token_count[token_fix] > 0: token_count[token_fix] += 1
					if token_count[token_fix] == 0: token_count[token_fix] = 1

					if token_fix not in punctuation:
						wc += 1
						if word_count[token_fix] > 0: word_count[token_fix] += 1
						if word_count[token_fix] == 0: word_count[token_fix] = 1

			#annotations = story[1]
			events = story[1][0]
			for e in events:
				label = e.attrib["name"]
				if e.attrib["from"] != e.attrib["to"]: multi_token_labels += 1
				if label_count[label] > 0: label_count[label] += 1
				if label_count[label] == 0: label_count[label] = 1

				within_summary_sequence.append(label)
				label_ids.append(e.attrib["id"])
				if label in entities or label in relations: entities_within_story.append(label)

			summaries_sequences.append(within_summary_sequence)
			summaries_lid.append(label_ids)
			entities_within_topic.append(entities_within_story)

			glued = " ".join(within_summary_sequence)
			if "add" in glued and "mul" in glued:
				print("Addition and multiplication relation in one story:",story.attrib["story_id"])

			within_summary_sequence, label_ids = [], []
			entities_within_story = []

			if story_length > max_length:
				max_length = story_length
			if story_length < min_length:

				min_length = story_length
		
		topic_entity_seq[topic.attrib["topic"]] = entities_within_topic
			
	#print(summaries_sequences[3][0], summaries_lid[3][0])
	#print(len(summaries_sequences[0]))# == len(summaries_lid))
	print("stories, sentences, tokens, words", storyc, sc, tc, wc)
	print("vocabulary size", len(vocabulary))
	print("min and max story length", min_length, max_length)
	#print("distinct labels, label usage, multi token times",len(label_count), sum(label_count.values()), multi_token_labels)
	# CHECK FOR POTENTIAL TYPOS in summaries
	#for w,c in word_count.items():
	#	if c <= 3: print(w,c)

	# CHECK FOR POTENTIAL TYPOS in labels
	all_anno = sum(label_count.values())
	"""	
	ordered = []
	for l,c in label_count.items():
		#print(l, round(c/all_anno,4))
		ordered.append((round(c/all_anno,4), l))

	ordered.sort(reverse=True)
	for (f, l) in ordered:
		print("-" + " "+ str(l) + " "+ str(f))
	"""
	lbigram_count = defaultdict(int) # label bigram
	cycle_bigrams, cyc_ids = [], []
	for j,sequence in enumerate(summaries_sequences):

		id_sequence = summaries_lid[j]
		bigram_id_sequence = list(zip(id_sequence, id_sequence[1:]))

		for k,bigram  in enumerate(zip(sequence, sequence[1:])):

			if lbigram_count[bigram] > 0: lbigram_count[bigram] += 1
			if lbigram_count[bigram] == 0: lbigram_count[bigram] = 1

			if bigram[0] == bigram[1]:
				#print("CYCLE", bigram)
				cycle_bigrams.append(bigram)
				cyc_ids.append(bigram_id_sequence[k])
				
	# Label cycles: bigrams
	#print("Cycle bigrams", sum(dict(Counter(cycle_bigrams)).values()))
	#print("Cyle bigrams IDs", len(dict(Counter(cyc_ids))))

	# TODO check what kind of cycles
	#print("Cycle bigrams", dict(Counter(cycle_bigrams)))
	# 46: topic_related_property', 'topic_related_property
	# 21: 'x_axis_label_least_value', 'x_axis_label_least_value'
	
	"""
	ordered = []
	all_lbi = sum(lbigram_count.values())
	for b,c in lbigram_count.items():
		#print(l, round(c/all_anno,4))
		ordered.append((round(c/all_lbi,4), b))

	ordered.sort(reverse=True)
	for (f, l) in ordered:
		print("-" + " "+ str(l) + " "+ str(f)) # sorted label bigram
	"""
	return topic_entity_seq


def compare_bar_order(tes):
	# tes is a dict; topics as keys, the value is a list of lists of entity sequences from stories

	# substitution
	sub_yorder = {"x_axis_label_least_value": "leastY", "x_axis_label_4th_highest_value":"fourthY", "x_axis_label_3rd_highest_value":"thirdY", "x_axis_label_Scnd_highest_value":"secondY", "x_axis_label_highest_value":"firstY", "x_axis_label_5th_highest_value":"fifthY"}

	sub_yorder_relations = {"y_axis_least_value_val":"leastHeight", "y_axis_Scnd_highest_val":"secondHeight", "y_axis_highest_value_val":"firstHeight", "y_axis_3rd_highest_val":"thirdHeight", "y_axis_inferred_highest_value_approx":"firstHeightApprox", "y_axis_inferred_least_value_approx":"leastHeightApprox", "y_axis_4th_highest_val":"fourthHeight", "y_axis_inferred_3rd_highest_value_approx":"thirdHeightApprox", "y_axis_inferred_Scnd_highest_value_approx":"secondHeightApprox", "y_axis_inferred_value_mul_v1=highest_v2=least":"mulFirstLeast", "y_axis_inferred_value_mul_v1=highest_v2=Scnd":"mulFirstSecond", "y_axis_inferred_value_add_v1=highest_v2=least":"addFirstLeast", "y_axis_inferred_value_mul_v1=least_v2=highest":"mulLeastFirst", "y_axis_inferred_value_mul_v1=Scnd_v2=least":"mulSecondLeast", "y_axis_inferred_value_mul_v1=Scnd_v2=3rd":"mulSecondThird", "y_axis_inferred_value_mul_v1=3rd_v2=highest":"mulThirdFirst", "y_axis_inferred_value_add_v1=highest_v2=Scnd":"addFirstSecond","y_axis_inferred_value_add_v1=highest_v2=3rd":"addFirstThird", "y_axis_inferred_value_add_v1=Scnd_v2=highest":"addSecondFirst", "y_axis_inferred_value_mul_v1=least_v2=Scnd":"mulLeastSecond", "y_axis_inferred_value_mul_v1=least_v2=3rd":"mulLeastThird","y_axis_inferred_value_mul_v1=highest_v2=3rd":"mulFirstThird", "y_axis_inferred_value_mul_v1=Scnd_v2=highest":"mulSecondFirst","y_axis_inferred_value_mul_v1=4th_v2=Scnd":"mulFourthSecond", "y_axis_inferred_value_mul_v1=3rd_v2=least":"mulThirdLeast","y_axis_inferred_value_mul_v1=3rd_v2=Scnd":"mulThirdSecond", "y_axis_inferred_value_add_v1=least_v2=3rd":"addLeastThird","y_axis_inferred_value_add_v1=Scnd_v2=3rd":"addSecondThird", "y_axis_inferred_value_add_v1=3rd_v2=least":"addThirdLeast"} # TODO check if complete and correct

	sub_yorder2 = ["x_axis_label_highest_value", "x_axis_label_Scnd_highest_value", "x_axis_label_3rd_highest_value", "x_axis_label_4th_highest_value", "x_axis_label_5th_highest_value", "x_axis_label_least_value"]
	sub_xorder = ["firstX", "secondX", "thirdX", "fourthX", "fifthX","lastX"]

	topics_elength_counter = {}
	# for each topic and its stories
	for topic, sequences in tes.items():
		plot_info = get_plot_info(topic)
		# replace entity label names with shorter names
		#replace_y = lambda x: [[sub_yorder[x2] for x2 in x1] for x1 in x]
		#sequences_y = replace_y(sequences)

		rel_sequences_y = []
		for x1 in sequences:
			current_rel_sequences_y = []
			for x2 in x1:
				if x2 in sub_yorder: current_rel_sequences_y.append(sub_yorder[x2])
				elif x2 in sub_yorder_relations: current_rel_sequences_y.append(sub_yorder_relations[x2])
			rel_sequences_y.append(current_rel_sequences_y)
		

		x_plot = plot_info["x"]
		y_plot = plot_info["y"]
		consider_xorder = sub_xorder[:len(x_plot)-1] # sequence of labels for X order starting with first bar
		consider_xorder.append("lastX")

		consider_yorder = sub_yorder2[:len(x_plot)-1] # sequence of labels for Y order starting with highest
		consider_yorder.append("x_axis_label_least_value")

		# list of triplets (bar height, bar name, bar order on X) sorted by descending heights
		pairs_sorted = sorted(list(zip(y_plot, x_plot, consider_xorder)), reverse=True)

		#print("\t")
		#if topic == "top_unis" and topic == "study_prog":
		#	break
			# topic: top_unis, study_prog have two bars of the same height
		#print(list(zip(pairs_sorted, sub_yorder2)))

		# get corresponding pairs for each bar: its height (largest, second...) and its order on X axis
		get_oxy_mapping = lambda a, b: {j:i[-1] for i,j in zip (a,b)}
		oxy_map = get_oxy_mapping(pairs_sorted, consider_yorder) # mapping as a dictionary

		# replace the entity labels given their bar heights given their order on X; skip if relations included
		#replace_x = lambda x: [[oxy_map[x2] for x2 in x1] for x1 in x]
		#sequences_x = replace_x(sequences)

		# analyze order of appearance
		#sequences_yx = list(zip(sequences_y, sequences_x)) # list of tuples of lists
		#lengths = {i:0 for i in range(1,15)}

		topics_elength_counter[topic] = {}

		#for sequence in sequences_yx:
		for sequence in rel_sequences_y:
			#within_story_yx = list(zip(sequence[0], sequence[1]))
			#print(within_story_yx)
			#if sequence[0] == len(x_plot):
			if len(sequence) in topics_elength_counter[topic]:
				topics_elength_counter[topic][len(sequence)].append(sequence)
			if len(sequence) not in topics_elength_counter[topic]:
				topics_elength_counter[topic][len(sequence)] = [sequence]
				#topics_elength_counter[topic][len(sequence)].append(sequence)

	#freq_len, freq_seq, k = -1, [], 0 # the most frequent length / number of occurring entities
	#freq_len2, freq_seq2, k2 = -1, [], 0
	for t_name, a in topics_elength_counter.items():
		print("- - - - ",t_name)

		keys_val_lengths = lambda x: {k: len(v) for k,v in x.items()}
		kvl = keys_val_lengths(a) # useful info: number of entities: number of stories with this no. entities
		keymax = max(kvl, key=kvl.get) # get number of entities that is most frequent
		print("\t First N:",keymax,", number of such stories:" ,len(a[keymax])) # a[keymax]

		# to analyze the second most frequent N of entities, remove keymax and recalculate
		#a.pop(keymax)
		#kvl = keys_val_lengths(a) # new
		#keymax = max(kvl, key=kvl.get) # new
		#print("\t Second N:", keymax,", number of such stories:", len(a[keymax])) # a[keymax]

		#if len(a) > 1:
		#	a.pop(keymax)
		#	kvl = keys_val_lengths(a) # new
		#	keymax = max(kvl, key=kvl.get) # new
		#	print("\t Third N:", keymax,", number of such stories:", len(a[keymax])) # a[keymax]

		# continue with keymax: a[keymax] - analyze its frequency distribution given the order in the story

		# in case no entity is referred to
		if keymax == 0:
			continue # skip the plotting part, continue from the top of the loop with a new topic

		info_topic = get_plot_info(t_name)
		no_bars = len(info_topic["x"])
		#if keymax > no_bars:
		#	print("N > bars",no_bars, keymax)
		#	for q in a[keymax]: print(q)

		counter_i = {} # each topic its own dict; order index as key, counter dict as value (entity as key)
		entities = set()
		for i in range(keymax): # order indices
			counter_i[i] = {}
			for story_entities in a[keymax]: # a[keymax] is a list of lists of tuples
				current_ent = story_entities[i]
				entities.add(current_ent)
				if current_ent in counter_i[i]:
					counter_i[i][current_ent] += 1
				if current_ent not in counter_i[i]:
					counter_i[i][current_ent] = 1

		#print(counter_i)
		#print("\n")
		
		counter_i2 = {e:{v:0 for v in range(len(counter_i))} for e in entities} #{ (l1,l2): {0:2, 1:10,3:0}, (l... } 
		for ent in entities:
			for k in range(len(counter_i)):
				#print(k, counter_i)
				if ent in counter_i[k]:
					counter_i2[ent][k] = counter_i[k][ent]
			
		#print(counter_i2)

		# plot these stats
		
		plot_labels = [k+1 for k in counter_i]
		x_plot = np.arange(1,len(counter_i)+1)  # the label locations
		width = 0.3
		fig, ax = plt.subplots()
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
		for label_tuple, d in counter_i2.items():
			current_x = list(d.keys()) # the indices
			current_y = list(d.values()) # the counts
			#print(current_x)
			_ = ax.bar(x_plot, current_y, width/len(x_plot),label=label_tuple,color=colors.pop())
			x_plot = [x + width/len(x_plot) for x in x_plot]

		ax.set_ylabel('Counts')
		ax.set_xlabel('Index of appearance')
		info_topic = get_plot_info(t_name)
		no_bars = len(info_topic["x"])
		s = " ".join([t_name+":", str(no_bars) ,"bars in total,", "most often mention ",str(len(x_plot)), "ents/rels"])
		ax.set_title(s)
		ax.set_xticks(plot_labels)
		ax.set_xticklabels(plot_labels)
		
		handles, labels = ax.get_legend_handles_labels()
		#print(handles, labels)
		hl = sorted(zip(handles,labels),key=operator.itemgetter(1))
		handles2, labels2 = zip(*hl)
		ax.legend(handles2, labels2)
		
		#ax.legend()
		fig.tight_layout()

		plt.show()
		
		



if __name__ == "__main__":

	tes = get_basics(xml_file)
	#print(tes)
	compare_bar_order(tes)
