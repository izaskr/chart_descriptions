import json
import xml.etree.ElementTree as ET

corpus = "data/chart_summaries_b01_toktest2.xml"


tree = ET.parse(corpus)
root = tree.getroot()

topic_entity_seq = {}

relevant_story_ids = set()

for topic in root:
	chart_id = topic.attrib["topic_id"]
	topic_id = chart_id[:2]
	if topic_id in {"12", "18"}:
		continue
	
	for story in topic:
		story_id = story.attrib["story_id"]
		relevant_story_ids.add(story_id)
		

print(relevant_story_ids)

list_relevant_ids = list(relevant_story_ids)
size85 = int(0.85 * len(relevant_story_ids))

print("size for split 85-15", size85, len(relevant_story_ids)-size85)
train_ids, val_ids = list_relevant_ids[:size85], list_relevant_ids[size85:]

print("Intersection size ", len(set(train_ids).intersection(set(val_ids))))

d = {"train":train_ids, "val":val_ids}
with open("train_val_ids.json", "w") as f:
	json.dump(d, f)


#import pdb; pdb.set_trace()
		
	
	
