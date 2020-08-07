"""
Delexicalize the corpus (in XML) and write it into a file to be processed by the NLG system

"""

import argparse
import xml.etree.ElementTree as ET
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("-xml", required=False, help="xml corpus with chart summaries and labels", default="/home/iza/chart_descriptions/corpora_v02/all_descriptions/chart_summaries_b01_toktest2.xml")
#parser.add_argument("-out", required=False, help="name of output file", default="corpora_v02/b01_delex")
args = vars(parser.parse_args())

xml_file = args["xml"]
#out_name = args["out"]

"""
lex_delex = {"y_axis":"YLABEL", "x_axis_label_least_value":"XLEAST", "x_axis_label_highest_value":"XHIGHEST", "x_axis_label_Scnd_highest_value":"XSECOND", "y_axis_least_value_val":"YLEAST", "y_axis_Scnd_highest_val":"YSECOND", "y_magnitude": "YMAG", "x_axis_label_3rd_highest_value":"XTHIRD", "y_axis_highest_value_val": "YHIGHEST", "y_axis_inferred_label":"YUNIT", "x_axis":"XLABEL", "y_axis_3rd_highest_val": "YTHIRD", "y_axis_inferred_highest_value_approx": "YHIGHESTAPPROX", "x_axis_label_4th_highest_value":"XFOURTH", "y_axis_inferred_least_value_approx":"YLEASTAPPROX",
"y_axis_4th_highest_val":"YFOURTH", "x_axis_labels_count":"COUNT", "y_axis_inferred_3rd_highest_value_approx":"YTHIRDAPPROX",
"x_axis_range_start":"XSTART", "x_axis_range_end":"XEND", "x_axis_labels":"BARNAMES", "y_axis_inferred_Scnd_highest_value_approx":"YSECONDAPPROX","x_interval":"INTERVAL", 
"y_axis_inferred_value_mul_v1=highest_v2=least":"MULHIGHESTLEAST", "slope_x_value":"SLOPEX", "y_axis_inferred_value_mul_v1=highest_v2=Scnd":"MULHIGHESTSECOND",
"y_axis_inferred_value_add_v1=highest_v2=least":"ADDHIGHESTLEAST",
"slope_y_value":"SLOPEY", "y_axis_inferred_value_mul_v1=least_v2=highest":"MULLEASTHIGHEST", "y_axis_inferred_value_mul_v1=Scnd_v2=least":"MULSECONDLEAST", "y_axis_inferred_value_mul_v1=Scnd_v2=3rd":"MULSECONDTHIRD", "y_axis_inferred_value_mul_v1=3rd_v2=highest":"MULTHIRDHIGHEST", "y_axis_inferred_value_add_v1=highest_v2=Scnd":"ADDHIGHESTSECOND",
"y_axis_inferred_value_add_v1=highest_v2=3rd":"ADDHIGHESTTHIRD",
"y_axis_inferred_value_add_v1=Scnd_v2=highest":"ADDSECONDHIGHEST",
"y_axis_inferred_value_mul_v1=least_v2=Scnd":"MULLEASTSECOND",
"y_axis_inferred_value_mul_v1=least_v2=3rd":"MULLEASTTHIRD",
"y_axis_inferred_value_mul_v1=highest_v2=3rd":"MULHIGHESTTHIRD",
"y_axis_inferred_value_mul_v1=Scnd_v2=highest":"MULSECONDHIGHEST",
"y_axis_inferred_value_mul_v1=4th_v2=Scnd":"MULFOURTHSECOND","y_axis_inferred_value_mul_v1=3rd_v2=least":"MULTHIRDLEAST",
"y_axis_inferred_value_mul_v1=3rd_v2=Scnd":"MULTHIRDSECOND","y_axis_inferred_value_add_v1=least_v2=3rd":"ADDLEASTTHIRD",
"y_axis_inferred_value_add_v1=Scnd_v2=3rd":"ADDSECONDTHIRD",
"y_axis_inferred_value_add_v1=3rd_v2=least":"ADDTHIRDLEAST" }

# x_axis_labels_rest , y_axis_inferred_value_add? , y_axis_inferred_label? , other_operation TODO

stays = {"topic_related_property","topic", "topic_related_object", "y_axis_highest_value", "y_axis_least_value", "order_Scnd", "y_axis_approx", "y_axis_trend", "separator", "y_axis_trend_up","slope_up", "interpretation", "y_x_comparison_rest", "final_label_left", "order_3rd", "y_axis_trend_down", "y_x_comparison", "x_axis_comparison", "x_axis_up", "every", "x_axis_trend", "order_rest", "order_4th", "x_axis_period", "x_axis_trend_down"}
"""

# each story starts with '' and ends with <end_of_description>_* where * is the script name

def open_delex_write(corpus):
	tree = ET.parse(corpus)
	root = tree.getroot()

	topic_name = ""

	for topic in root:
		topic_name = topic.attrib["topic"] # will be the name of the file

		f = open(topic_name, "w", encoding="utf-8")
		f.write("''" + "\n")
		i= 0
		for story in topic:
			i +=1
			sentences = story[0][1]
			tokens = [] # list of tokens as they appear in the description
			token_ids = [] # list of token IDs as they appear in the description
			vocab = {} # for each description, tokenID: token as key: value pairs
			for sent in sentences:
				#for t in sent:
					#print(t.attrib["content_fix"])
				get_tokens = lambda x: [t.attrib["content_fix"] for t in x]
				get_token_ids = lambda x: [t.attrib["id"] for t in x]
				make_dict = lambda x: {t.attrib["id"]: t.attrib["content_fix"] for t in x}
				vocab = {**vocab, **make_dict(sent)}
				tokens += get_tokens(sent)
				token_ids += get_token_ids(sent)
			#print(tokens)

			events = story[1][0]
			get_labels = lambda x: [(e.attrib["name"], e.attrib["from"], e.attrib["to"]) for e in x]
			labels = get_labels(events)

			all_label_ids = set() # IDs of labeled tokens
			segmented_label_ids = [] # token IDs, segmented as they are labeled or not
			for i,(label, start, end) in enumerate(labels):
				if start != end:
					i1, i2 = token_ids.index(start), token_ids.index(end)
					all_label_ids = all_label_ids.union(set(token_ids[i1:i2+1]))
					segmented_label_ids.append((token_ids[i1:i2+1], label))

				if start==end:
					all_label_ids.add(start)
					segmented_label_ids.append((start,label))

			#print(segmented_label_ids)

			"""
			^ above we collected IDs of labeled tokens; now we should collect unlabeled and labeled tokens in the correct order. This is done in the for loop below
			"""
			segmented_ids = []
			for j in token_ids:
				if j in all_label_ids:

					for seg, label in segmented_label_ids:
						if j == seg: # one token label
							segmented_ids.append((j,label))

						if type(seg) == list:
							if j in seg:
								if (seg,label) not in segmented_ids:
									segmented_ids.append((seg,label))

				if j not in all_label_ids: # not a labeled token
					segmented_ids.append((j, None))

			#print(segmented_ids)

			# the loop below replaces token IDs with tokens after checking if they should be delexicalized
			for m, label in segmented_ids: # m can be a list or a string
				s = "NOTHING"
				if label in lex_delex:
					s = lex_delex[label]
					#print(m)

				else:
					if type(m) == str:
						get_text = lambda x,d: d[x]
						s = get_text(m,vocab)
					if type(m) == list:
						get_text_list = lambda x: [vocab[e] for e in x]
						w_list = get_text_list(m)
						s = "+".join(w_list)
				if label:
					s = s + "\t" + "<"+ label + ">"

				f.write(s + "\n")
			f.write("<end_of_description>" + "\n")
		f.close()


def open_delex_write_json(corpus):
	"""
	Open the corpus XML, iteratively collects the summaries, delexicalizing chosen token
	By default it delexicalizes tokes labeled with labels found in lex_delex.json, which comprises bar names, heights and relations
	Other tokens (labeled or not) stay as is, with the labels removed.
	The delexicalized version of the summaries is written into a .json
	"""
	tree = ET.parse(corpus)
	root = tree.getroot()

	with open("lex_delex.json", "r") as jf:
		all_lex_delex = json.load(jf)
	lex_delex = {**all_lex_delex["bar_information"], **all_lex_delex["topic_information"]}

	topic_name = ""
	# join the summaries by topic, e.g. all 02_X into 02, then shuffle and split into train and val
	topicwise = {}

	for topic in root:
		# topic ID has 4 integers, the first 2 are the actual topic ID, the last 2 are the plot ID
		# for example, 01_01, 01_02 ...
		topic_id = topic.attrib["topic_id"]
		short_topic_id = topic_id[:2]
		if short_topic_id not in topicwise:
			topicwise[short_topic_id] = {}

		topic_summaries = {}
		#f = open(topic_name, "w", encoding="utf-8")
		#f.write("''" + "\n")
		s_counter= 0
		for story in topic:
			new_summary_content = ""
			story_id = story.attrib["story_id"]
			s_counter +=1
			sentences = story[0][1]
			tokens = [] # list of tokens as they appear in the description
			token_ids = [] # list of token IDs as they appear in the description
			vocab = {} # for each description, tokenID: token as key: value pairs
			for sent in sentences:
				#for t in sent:
					#print(t.attrib["content_fix"])
				get_tokens = lambda x: [t.attrib["content_fix"] for t in x]
				get_token_ids = lambda x: [t.attrib["id"] for t in x]
				make_dict = lambda x: {t.attrib["id"]: t.attrib["content_fix"] for t in x}
				vocab = {**vocab, **make_dict(sent)}
				tokens += get_tokens(sent)
				token_ids += get_token_ids(sent)
			#print(tokens)

			events = story[1][0]
			get_labels = lambda x: [(e.attrib["name"], e.attrib["from"], e.attrib["to"]) for e in x]
			labels = get_labels(events) # list of tuples [ (label_name, start, end) ]

			all_label_ids = set() # IDs of labeled tokens
			segmented_label_ids = [] # token IDs, segmented as they are, labeled or not
			for i,(label, start, end) in enumerate(labels):
				# collect all token IDs that are in a multi-span label
				if start != end:
					i1, i2 = token_ids.index(start), token_ids.index(end)
					all_label_ids = all_label_ids.union(set(token_ids[i1:i2+1]))
					segmented_label_ids.append((token_ids[i1:i2+1], label))

				if start==end:
					all_label_ids.add(start)
					segmented_label_ids.append((start,label))

			#print(segmented_label_ids)

			"""
			^ above we collected IDs of labeled tokens; now we should collect unlabeled and labeled tokens in the correct order. 
			This is done in the for loop below
			"""
			segmented_ids = []
			for j in token_ids:
				if j in all_label_ids:

					for seg, label in segmented_label_ids:
						if j == seg: # one token label
							segmented_ids.append((j,label))

						if type(seg) == list: # multi-token label
							if j in seg:
								if (seg,label) not in segmented_ids:
									segmented_ids.append((seg,label))

				if j not in all_label_ids: # not a labeled token
					segmented_ids.append((j, None))

			#print(segmented_ids)

			# the loop below replaces token IDs with tokens after checking if they should be delexicalized
			content_plan = []
			for m, label in segmented_ids: # m can be a list or a string of token ID(s), label is a string
				s = "NOTHING"
				if label in lex_delex:
					s = lex_delex[label]
					# append the delex placeholder of the label
					content_plan.append(s)
					#print(m)

				else:
					if type(m) == str:
						s = vocab[m]
					if type(m) == list:
						get_text_list = lambda x: [vocab[e] for e in x]
						w_list = get_text_list(m)
						s = " ".join(w_list)

				new_summary_content = new_summary_content + " " + s
			print(s_counter, new_summary_content)
			topic_summaries[story_id] = new_summary_content
			# save the content plans (list of labels) and summaries given the topic, both are strings
			content_plan = " ".join(content_plan)
			topicwise[short_topic_id][story_id] = (content_plan, new_summary_content)

		# write the delexicalized summaries into a json file
		# with open("/home/iza/chart_descriptions/corpora_v02/delexicalized/delex_"+topic_id+".json", "w", encoding="utf-8") as new_jf:
		# 	json.dump(topic_summaries , new_jf)
	#print(topicwise)
	#import pdb; pdb.set_trace()
	write_to_json = True
	if write_to_json == True:

		print("Number of topics", len(topicwise))
		# For each topic, write the content plans and summaries into a parallel
		# prior to that shuffle and split into train and validation
		for shortTopicID, storyIDs0 in topicwise.items():
			storyIDs = list(storyIDs0.keys())
			random.shuffle(storyIDs)
			size_val = int(0.2 * len(storyIDs))
			trainIDs, valIDs = storyIDs[size_val:], storyIDs[:size_val]
			print("Topic %s has %d in train and %d in validation" % (shortTopicID, len(trainIDs), len(valIDs)))

			# write the train files
			with open("/home/iza/chart_descriptions/corpora_v02/delexicalized/delex_"+shortTopicID+"_train.txt", "w", encoding="utf-8") as parallel:
				for id in trainIDs:
					parallel.write(topicwise[shortTopicID][id][0] + "\t" + topicwise[shortTopicID][id][1] + "\n")

			with open("/home/iza/chart_descriptions/corpora_v02/delexicalized/delex_"+shortTopicID+"_val.txt", "w", encoding="utf-8") as parallel2:
				for id in valIDs:
					parallel2.write(topicwise[shortTopicID][id][0] + "\t" + topicwise[shortTopicID][id][1] + "\n")
			input("ENTER for next topic/chart")


def create_split(dict_ids, split_type):
	""" dict_ids : dict, major topic IDs as keys, the value is a set of minor topic IDs """
	train_IDs, test_IDs = set(), set()
	if split_type == "2":
		for major, minor_list in dict_ids.items():
			if len(minor_list) == 1:
				train_IDs.add(minor_list[0])

			if len(minor_list) == 2:
				_temp = random.sample(minor_list, len(minor_list)) # create new list and shuffle it without change the original
				train_IDs.add(_temp[0])
				test_IDs.add(_temp[1])

	if split_type == "3":
		# 14 topics in total; take 10 for training+val and 4 for testing
		k = list(dict_ids.keys())
		random.shuffle(k)

		for i in k[:10]:
			train_IDs = train_IDs.union(set(dict_ids[i]))
		print("train IDs",train_IDs)

		for j in k[10:]:
			test_IDs = test_IDs.union(set(dict_ids[j]))
		print("test IDs",test_IDs)

	if train_IDs == {} or test_IDs == {}:
		print("Check the splitting step!")

	if train_IDs.intersection(test_IDs):
		print("Check the splitting step, repeating IDs in train and test")

	return train_IDs, test_IDs




def open_delex_key_value(corpus):
	"""
	Open the corpus XML, iteratively collects the summaries, delexicalizing chosen token
	By default it delexicalizes tokes labeled with labels found in lex_delex.json, which comprises bar names, heights and relations
	Other tokens (labeled or not) stay as is, with the labels removed.
	The delexicalized version of the summaries is written into a .json
	"""

	# keep track of topics: their major and minor IDs and names
	minor2name = {} # "01_02": "gender_pay_gap"
	major2minor = {} # "01": {"01_01", "01_02"}

	tree = ET.parse(corpus)
	root = tree.getroot()

	with open("lex_delex.json", "r") as jf:
		all_lex_delex = json.load(jf)
	lex_delex = {**all_lex_delex["bar_information"], **all_lex_delex["topic_information"]}

	topic_name = ""
	# join the summaries by topic, e.g. all 02_X into 02, then shuffle and split into train and val
	topicwise = {}

	for topic in root:
		# topic ID has 4 integers, the first 2 are the actual topic ID, the last 2 are the plot ID
		# for example, 01_01, 01_02 ...
		topic_id = topic.attrib["topic_id"]
		short_topic_id = topic_id[:2]
		if short_topic_id not in topicwise:
			topicwise[short_topic_id] = {}

		minor2name[short_topic_id] = topic.attrib["topic"]
		if short_topic_id in major2minor:
			major2minor[short_topic_id].append(topic_id)
		if short_topic_id not in major2minor:
			major2minor[short_topic_id] = [topic_id]


		topic_summaries = {}
		#f = open(topic_name, "w", encoding="utf-8")
		#f.write("''" + "\n")
		s_counter = 0
		for story in topic:
			new_summary_content = ""
			story_id = story.attrib["story_id"]
			s_counter +=1
			sentences = story[0][1]
			tokens = [] # list of tokens as they appear in the description
			token_ids = [] # list of token IDs as they appear in the description
			vocab = {} # for each description, tokenID: token as key: value pairs
			for sent in sentences:
				#for t in sent:
					#print(t.attrib["content_fix"])
				get_tokens = lambda x: [t.attrib["content_fix"] for t in x]
				get_token_ids = lambda x: [t.attrib["id"] for t in x]
				make_dict = lambda x: {t.attrib["id"]: t.attrib["content_fix"] for t in x}
				vocab = {**vocab, **make_dict(sent)}
				tokens += get_tokens(sent)
				token_ids += get_token_ids(sent)
			#print(tokens)

			events = story[1][0]
			get_labels = lambda x: [(e.attrib["name"], e.attrib["from"], e.attrib["to"]) for e in x]
			labels = get_labels(events) # list of tuples [ (label_name, start, end) ]

			all_label_ids = set() # IDs of labeled tokens
			segmented_label_ids = [] # token IDs, segmented as they are, labeled or not
			for i,(label, start, end) in enumerate(labels):
				# collect all token IDs that are in a multi-span label
				if start != end:
					i1, i2 = token_ids.index(start), token_ids.index(end)
					all_label_ids = all_label_ids.union(set(token_ids[i1:i2+1]))
					segmented_label_ids.append((token_ids[i1:i2+1], label))

				if start==end:
					all_label_ids.add(start)
					segmented_label_ids.append((start,label))

			#print(segmented_label_ids)

			"""
			^ above we collected IDs of labeled tokens; now we should construct SRC
			for TG take plain summary text
			"""

			#print(segmented_ids)
			#import pdb; pdb.set_trace()
			key_value = []
			summary_text = []

			# segmented_label_ids
			# e.g. [(['01_01-01-1-5', '01_01-01-1-6', '01_01-01-1-7'], 'topic'), (['01_01-01-1-9', '01_01-01-1-10', '01_01-01-1-11', '01_01-01-1-12', '01_01-01-1-13'], 'x_axis_labels'), ('01_01-01-2-6', 'x_axis_label_highest_value'), ('01_01-01-2-9', 'y_axis_highest_value'), (['01_01-01-2-10', '01_01-01-2-11'], 'y_axis'), ('01_01-01-2-13', 'order_Scnd'), ('01_01-01-2-15', 'x_axis_label_Scnd_highest_value'), ('01_01-01-2-17', 'x_axis_label_least_value'), ('01_01-01-2-20', 'y_axis_least_value')]
			for (tokenIDs, label) in segmented_label_ids:
				if label in lex_delex:
					_key = lex_delex[label]
					_value = None

					if type(tokenIDs) == str: # single token ID for a single token
						_value = vocab[tokenIDs]
					if type(tokenIDs) == list:
						_value = " ".join([vocab[tid] for tid in tokenIDs])
					if not _value:
						print("What's in the tokens??", tokenIDs)

					key_value.append([_key + "[" + _value + "]"])
					#final = _key + "[" + _value + "]"

			#import pdb; pdb.set_trace()
			key_value = ", ".join([u[0] for u in key_value])
			topicwise[short_topic_id][story_id] = (key_value, " ".join(tokens))


			# segmented_ids looks like
			# [ ('01_01-01-1-1', None), ('01_01-01-1-2', None), ('01_01-01-1-3', None), ('01_01-01-1-4', None),
			#  (['01_01-01-1-5', '01_01-01-1-6', '01_01-01-1-7'], 'topic'), ('01_01-01-1-8', None),
			#  (['01_01-01-1-9', '01_01-01-1-10', '01_01-01-1-11', '01_01-01-1-12', '01_01-01-1-13'], 'x_axis_labels'),
			#  ('01_01-01-1-14', None), ('01_01-01-2-1', None), ('01_01-01-2-2', None), ('01_01-01-2-3', None),
			#  ('01_01-01-2-4', None), ('01_01-01-2-5', None), ('01_01-01-2-6', 'x_axis_label_highest_value'),
			#  ('01_01-01-2-7', None), ('01_01-01-2-8', None), ('01_01-01-2-9', 'y_axis_highest_value'),
			#  (['01_01-01-2-10', '01_01-01-2-11'], 'y_axis'), ('01_01-01-2-12', None), ('01_01-01-2-13', 'order_Scnd'),
			#  ('01_01-01-2-14', None), ('01_01-01-2-15', 'x_axis_label_Scnd_highest_value'), ('01_01-01-2-16', None),
			#  ('01_01-01-2-17', 'x_axis_label_least_value'), ('01_01-01-2-18', None), ('01_01-01-2-19', None),
			#  ('01_01-01-2-20', 'y_axis_least_value'), ('01_01-01-2-21', None) ]
			o = None
	# loop through the collected pairs of SRC and TG and write into files
	dir_path = "/home/iza/chart_descriptions/corpora_v02/keyvalue/"
	#src = open(dir_path + "src_split2.txt", "w", encoding="utf8")
	#tg = open(dir_path + "tg_split2.txt", "w", encoding="utf8")

	split_type = "3"

	src_trainval = open(dir_path + "src_trainval_3.txt", "w", encoding="utf8")
	src_test = open(dir_path + "src_test_3.txt", "w", encoding="utf8")
	tg_trainval = open(dir_path + "tg_trainval_3.txt", "w", encoding="utf8")
	tg_test = open(dir_path + "tg_test_3.txt", "w", encoding="utf8")

	# split2: split topics, such that 1 chart goes to train, the other to test. If a topic has a single chart, put it into train
	# split3: put some topics entirely into test (testing on unseen topic)
	train_minor_IDs, test_minor_IDs = create_split(major2minor, split_type)
	print("Split mode", split_type)
	print("Train",train_minor_IDs, "\n", "Test ",test_minor_IDs)

	#if split_type == "split2":

	for greatTopicID, summaryIDs in topicwise.items():
		#print("great topic ID", greatTopicID)
		#print("summary IDs", summaryIDs.keys(), "\n"*3)
		for one_story_id, (src_s, tg_s) in summaryIDs.items():
			#print(one_story_id)
			current_minor_id = one_story_id[:-3]
			if current_minor_id in train_minor_IDs:
				src_trainval.write(src_s + "\n")
				tg_trainval.write(tg_s + "\n")
			else:
				src_test.write(src_s + "\n")
				tg_test.write(tg_s + "\n")
			#src.write(src_s + "\n")
			#tg.write(tg_s + "\n")
	#print(major2minor)
	#print(minor2name)
	src_trainval.close()
	tg_trainval.close()
	src_test.close()
	tg_test.close()
	#src.close()
	#tg.close()




if __name__ == "__main__":
	#open_delex_write(xml_file)
	open_delex_key_value(xml_file)

"""
Traceback (most recent call last):
  File "/home/iza/chart_descriptions/delexicalize.py", line 252, in <module>
    open_delex_write_json(xml_file)
  File "/home/iza/chart_descriptions/delexicalize.py", line 187, in open_delex_write_json
    i1, i2 = token_ids.index(start), token_ids.index(end)
ValueError: '11_01-10-2--5' is not in list
"""
