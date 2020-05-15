
import os
import json
import xml.etree.ElementTree as ET
import allennlp
from allennlp.predictors.predictor import Predictor


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

xml_path = "./corpora_v02/all descriptions/chart_summaries_b01_toktest2-xml"

def open_xml_collect_coref(corpus):
	id_clusters = {}

	tree = ET.parse(corpus)
	root = tree.getroot()

	topic_name = ""

	for topic in root:
		topic_name = topic.attrib["topic"]
		topic_id = topic.attrib["topic_id"]

		for story in topic:
			story_id = story.attrib["story_id"]

			# story[0] is the text child
			# context is the entire summary
			content = story[0][0]

			# resolve corefences

			results = predictor.predict(document=content)
					
			clusters = results["clusters"]
			doc = results["document"] # tokenized text, a list of tokens
			# write into file: i_summary, current text and clusters

			if clusters:
				lexicalized = {}

				for j,cluster in enumerate(clusters): # cluster is a list: start_index, end_index
					entities = []
				
					for startend in cluster:
						ent = " ".join(doc[startend[0]:startend[1]+1])
						entities.append(ent)

					lexicalized[j] = entities
										

				id_clusters[story_id] = lexicalized

	return id_clusters


all_clusters = open_xml_collect_coref(xml_path)


with open("id_corefer.json", "w", encoding="utf-8") as jf:
	json.dump(all_clusters, jf)





