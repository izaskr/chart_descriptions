#!/usr/bin/python3

"""
For discourse-related (not data-dependant) labels, learn the vocabulary, e.g. slope_up labels tokens, such as 'grows', 'rising' etc.

order_Scnd
y_axis_approx
y_axis_trend
separator
y_axis_trend_up
slope_up
y_x_comparison_rest
order_3rd
y_axis_trend_down
y_x_comparison
x_axis_range_start
x_axis_range_end
order_rest
order_4th
x_axis_trend
x_axis_trend_down
y_axis_least_value

"""

import xml.etree.ElementTree as ET
from string import punctuation
from collections import defaultdict, Counter
import json
import numpy as np


def get_discourse_tokens(corpus):
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


	entities = {"x_axis_label_least_value", "x_axis_label_4th_highest_value", "x_axis_label_3rd_highest_value", "x_axis_label_Scnd_highest_value", "x_axis_label_highest_value","x_axis_label_5th_highest_value"}

	discourse = {"order_Scnd","y_axis_approx","y_axis_trend","separator",
"y_axis_trend_up","slope_up","y_x_comparison_rest","order_3rd",
"y_axis_trend_down","y_x_comparison","x_axis_range_start","x_axis_range_end","order_rest",
"order_4th","x_axis_trend","x_axis_trend_down", "y_axis_least_value"}

	vocab_discourse = set() # only word types that are labeled for discourse
	dlabel_words = {d:set() for d in discourse}
	words_dlabel = {}
	

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
			"""
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
			"""
			#annotations = story[1]
			events = story[1][0]
			for e in events:
				label = e.attrib["name"]

				if label in dlabel_words:
					labeled_text = e.attrib["text"].lower()
					vocab_discourse.add(labeled_text)
					
					dlabel_words[label].add(labeled_text)
					if labeled_text in words_dlabel and label in words_dlabel[labeled_text]:
						words_dlabel[labeled_text][label] += 1
					if labeled_text in words_dlabel and label not in words_dlabel[labeled_text]:
						words_dlabel[labeled_text][label] = 1
					if labeled_text not in words_dlabel:
						words_dlabel[labeled_text]= {}
						words_dlabel[labeled_text][label] = 1

					#dlabel_words = {d:set() for d in discourse}
					#words_dlabel = {}
					
	# add manually
	words_dlabel["around"] = {"<y_axis_approx>":1}
	return dlabel_words, words_dlabel

"""
if __name__ == "__main__":
	xml_file = "corpora_v02/chart_summaries_b01.xml"
	lw, wl = get_discourse_tokens(xml_file)
	#print(lw, "\n")
	#print(wl)
"""
