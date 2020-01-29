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
	topic_entity_seq = {}

	for topic in root:
		#print(topic.attrib["topic"])

		#plot_info = get_plot_info(topic.attrib["topic"])
		entities_within_topic, entities_within_story = [], []

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
				if label in entities: entities_within_story.append(label)

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
	sub_yorder2 = ["x_axis_label_highest_value", "x_axis_label_Scnd_highest_value", "x_axis_label_3rd_highest_value", "x_axis_label_4th_highest_value", "x_axis_label_5th_highest_value", "x_axis_label_least_value"]
	sub_xorder = ["firstX", "secondX", "thirdX", "fourthX", "fifthX","lastX"]

	topics_elength_counter = {}
	# for each topic and its stories
	for topic, sequences in tes.items():
		plot_info = get_plot_info(topic)
		# replace entity label names with shorter names
		replace_y = lambda x: [[sub_yorder[x2] for x2 in x1] for x1 in x]
		sequences_y = replace_y(sequences)


		x_plot = plot_info["x"]
		y_plot = plot_info["y"]
		consider_xorder = sub_xorder[:len(x_plot)-1] # sequence of labels for X order starting with first bar
		consider_xorder.append("lastX")

		consider_yorder = sub_yorder2[:len(x_plot)-1] # sequence of labels for Y order starting with highest
		consider_yorder.append("x_axis_label_least_value")

		# list of triplets (bar height, bar name, bar order on X) sorted by descending heights
		pairs_sorted = sorted(list(zip(y_plot, x_plot, consider_xorder)), reverse=True)


		#print("\t")
		if topic == "top_unis" and topic == "study_prog":
			break
			# topic: top_unis, study_prog have two bars of the same height
		#print(list(zip(pairs_sorted, sub_yorder2)))

		# get corresponding pairs for each bar: its height (largest, second...) and its order on X axis
		get_oxy_mapping = lambda a, b: {j:i[-1] for i,j in zip (a,b)}
		oxy_map = get_oxy_mapping(pairs_sorted, consider_yorder)

		# replace the entity labels given their bar heights given their order on X
		replace_x = lambda x: [[oxy_map[x2] for x2 in x1] for x1 in x]
		sequences_x = replace_x(sequences)
		#print(sequences_y)
		#print(sequences_x)

		# analyze order of appearance
		sequences_yx = list(zip(sequences_y, sequences_x)) # list of tuples of lists
		#lengths = {i:0 for i in range(1,15)}
		topics_elength_counter[topic] = {}

		for sequence in sequences_yx:
			within_story_yx = list(zip(sequence[0], sequence[1]))
			#print(within_story_yx)
			#if sequence[0] == len(x_plot):
			if len(within_story_yx) in topics_elength_counter[topic]:
				topics_elength_counter[topic][len(within_story_yx)].append(within_story_yx)
			if len(within_story_yx) not in topics_elength_counter[topic]:
				topics_elength_counter[topic][len(within_story_yx)] = []
				topics_elength_counter[topic][len(within_story_yx)].append(within_story_yx)

	freq_len, freq_seq, k = -1, [], 0 # the most frequent length / number of occurring entities
	freq_len2, freq_seq2, k2 = -1, [], 0
	for t_name,a in topics_elength_counter.items():
		print(t_name)

		keys_val_lengths = lambda x: {k: len(v) for k,v in x.items()}
		kvl = keys_val_lengths(a) # useful info: number of entities: number of stories with this no. entities
		keymax = max(kvl, key=kvl.get) # get number of entities that is most frequent
		print(keymax) 

		# continue with keymax: a[keymax] - analyze its frequency distribution given the order in the story

		#for length_e,seqs in a.items():
			#print("\t",length_e,seqs, len(seqs))

		#keymax = max(a, key=len(a.get))
		
				
				
		

		
		
		
#a, b, c
#5, 2, 3
#highest, least, second

# highest: firstX
# least: secondX
# second: lastX


if __name__ == "__main__":

	tes = get_basics(xml_file)
	compare_bar_order(tes)
