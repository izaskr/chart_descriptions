"""
Parse the main XML file with the entire corpus, use also the delexicalized version
from delexicalize.py

Check per-topic and per-chart statistics: x and y labels versus relations of addition/multiplication/trend/mean
"""

import json
import itertools
from collections import Counter
from delexicalize import open_delex_key_value

class Corpus():
    def __init__(self):

        self.xml_file = "/home/skrjanec/chart_descriptions/data/chart_summaries_b01_toktest2.xml"

        # get the delexicalized version of the corpus - topicwise - a nested dict
        # topicwise is a dict with topics as the keys: '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
        # '11', '12', '13', '14', '18', '16', '17', '15'
        # its value is a dict with summary IDs as keys ('15_01b-01', '15_01b-02' ...)
        # the value is a tuple:
        # 1st: key-value pairs as a copy format
        # 2nd: lexicalized summary
        # 3rd key-value pair is a set
        # 4th: delexicalized summary
        self.topicwise = open_delex_key_value(self.xml_file)

        #self.minor_topicIDs = set(summaryID[:-3] for greatTopicID, summaryIDs in self.topicwise.items() for summaryID, tup in summaryIDs.items())
        self.major2minor = {k: set() for k in self.topicwise}

        # I need: minor topic id 2 summaryIDs
        self.minorTopicID2summaryID2formats = {}
        for greatTopicID, summaryIDs in self.topicwise.items():
            for summaryID, tup in summaryIDs.items():
                current_minor = summaryID[:-3]
                self.major2minor[greatTopicID].add(current_minor)
                if current_minor in self.minorTopicID2summaryID2formats:
                    self.minorTopicID2summaryID2formats[current_minor][summaryID] = {}

                if current_minor not in self.minorTopicID2summaryID2formats:
                    self.minorTopicID2summaryID2formats[current_minor] = {summaryID: {}}

        #print(self.minorTopicID2summaryID2formats)

        #self.per_chart_count("15", "15_01b")
        self.label_stats = Counter()

        self.topic2chart2label_stats = {}
        for major_id, minor_ids in self.major2minor.items():
            for minor_id in minor_ids:
                chart_temp_stats = self.per_chart_count(major_id, minor_id)
                self.label_stats = self.label_stats + chart_temp_stats

        # print the percentage of inferred labels given all labels in the entire dataset in general
        self.ratio_basic_inferred(self.label_stats)


    def get_keys_from_kv_string(self, s):
        list_keys = []
        s_split = s.split("],")
        for kv_pair in s_split:
            k = kv_pair.split("[")[0]
            list_keys.append(k)
        #print(list_keys)
        list_keys = list(set(list_keys))
        return Counter(list_keys)

    def per_chart_count(self, topic_id, chart_id):
        # count the number of basic and inferred labels
        # inferred labels: MUL*, ADD*, *APPROX, OTHEROPERATION, SLOPE*,  GRY*, YMEAN, x_axis_trend (?)
        chart_stats = Counter()
        for summary_ID, tup in self.topicwise[topic_id].items():
            if summary_ID.startswith(chart_id):
                kv = tup[0] # key-value in copy format - check frequency
                #print("current st", chart_stats)
                #print("adding ", self.get_keys_from_kv_string(kv))
                chart_stats = chart_stats + self.get_keys_from_kv_string(kv)

        return chart_stats

    def ratio_basic_inferred(self, d):
        # based on the label counter, calculate the ratio between inferred and basic labels
        inferred, basic = Counter(), Counter()
        for label, count in d.items():
            if (label.startswith("MUL") or label.startswith("ADD") or label.startswith("OTHEROPERATION") or
                label.startswith("SLOPE") or label.startswith("YMEAN") or label.startswith("GR") or
                label.endswith("APPROX")):  # parentheses :)
                inferred[label] = count
            else:
                basic[label] = count

        r = sum(inferred.values()) / sum(d.values())
        v = sum(inferred.values()) / sum(basic.values())
        print("The ratio between inferred and all labels is %f" % r)
        print("The ratio between inferred and basic labels is %f" % v)
        return r


corpus = Corpus()

import pdb; pdb.set_trace()

"""
About 8% of all labels are inferred (and not basic)
The ratio between inferred and basic labels is 9/100.

The ratio between inferred and all labels is 0.113921
The ratio between inferred and basic labels is 0.128568
"""
