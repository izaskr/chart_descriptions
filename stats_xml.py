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

parser = argparse.ArgumentParser()
parser.add_argument("-xml", required=False, help="xml corpus with chart summaries and labels", default="corpora_v02/chart_summaries_b01.xml")
args = vars(parser.parse_args())

xml_file = args["xml"]

def get_basics(corpus):
	tree = ET.parse(corpus)
	root = tree.getroot()

	token_count, word_count, label_count = defaultdict(int), defaultdict(int), defaultdict(int)
	storyc, sc, tc, wc = 0, 0, 0, 0
	multi_token_labels = 0
	within_summary_sequence, label_ids = [], [] # sequence of labels
	summaries_sequences, summaries_lid = [], []
	vocabulary = set()
	min_length = 1000
	max_length = -1
	story_length = 0
	for topic in root:
		for story in topic:
			story_length = 0
			within_summary_sequence = []
			storyc += 1
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

			summaries_sequences.append(within_summary_sequence)
			summaries_lid.append(label_ids)

			glued = " ".join(within_summary_sequence)
			if "add" in glued and "mul" in glued:
				print(story.attrib["story_id"])

			within_summary_sequence, label_ids = [], []
			if story_length > max_length:
				max_length = story_length
			if story_length < min_length:

				min_length = story_length
			
	#print(summaries_sequences[3][0], summaries_lid[3][0])
	print(len(summaries_sequences[0]))# == len(summaries_lid))
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
				

	print("Cycle bigrams", sum(dict(Counter(cycle_bigrams)).values()))
	print("Cyle bigrams IDs", len(dict(Counter(cyc_ids))))

	# TODO check what kind of cycles
	print("Cycle bigrams", dict(Counter(cycle_bigrams)))
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


if __name__ == "__main__":

	get_basics(xml_file)
