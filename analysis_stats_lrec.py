"""
Parse the main XML file with the entire corpus, use also the delexicalized version
from delexicalize.py

Check per-topic and per-chart statistics: x and y labels versus relations of addition/multiplication/trend/mean
"""

import json
import itertools
import git
import os
from collections import Counter
from delexicalize import open_delex_key_value

repo = git.Repo('.', search_parent_directories=True)
root_repo_dir = repo.working_tree_dir


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

        # #self.per_chart_count("15", "15_01b")
        # self.label_stats = Counter()
        #
        # self.topic2chart2label_stats = {}
        # for major_id, minor_ids in self.major2minor.items():
        #     for minor_id in minor_ids:
        #         chart_temp_stats = self.per_chart_count(major_id, minor_id)
        #         self.label_stats = self.label_stats + chart_temp_stats
        #
        # # print the percentage of inferred labels given all labels in the entire dataset in general
        # self.ratio_basic_inferred(self.label_stats)

        self.labels_stats_bychart, self.label_stats, self.labels_stats_bysummary = \
            self.read_from_parallel_txt("corpora_v02/keyvalue/complete_fixed_tok/cpy")
        self.ratio_basic_inferred(self.label_stats)
        #self.gate_portion_inferred(self.label_stats)

        # analyze the entity distribution for each chart
        #self.per_chart_analysis(self.labels_stats_bychart, "chart")

        # analyze the entity distribution for each summary
        self.per_chart_analysis(self.labels_stats_bysummary, "summary")

    def get_keys_from_kv_string(self, s):
        list_keys = []
        s_split = s.split("],")
        for kv_pair in s_split:
            k = kv_pair.split("[")[0]
            k = k.strip()
            list_keys.append(k)

        return Counter(list_keys)


    def read_from_parallel_txt(self, path_to_folder_with_copy_input):
        """ Reading from the XML file results in some noise,
        so read in the source side of the copy-format data and also the summary IDs
        complete_fixed_tok/*_src_a.txt  and ids_*_a.txt
        Return a collection of counters by chart and for the entire corpus
        """
        splits = ["train", "val", "test"]
        counter_by_chart = Counter()
        counter_by_summary = Counter()
        counter_all = Counter()  # counter for the labels for the entire corpus disregarding charts and summaries
        for split in splits:
            source_file = os.path.join(root_repo_dir, path_to_folder_with_copy_input, split + "_src_a.txt")
            id_file = os.path.join(root_repo_dir, path_to_folder_with_copy_input, "ids_" + split + "_a.txt")

            chart_ids, summary_ids = [], []
            with open(id_file) as fid:
                for line in fid:
                    summaryid = line.split()[0]
                    summary_ids.append(summaryid)  # append the summary ID
                    chart_ids.append(summaryid[:-3])  # append the chart ID

            with open(source_file) as fs:
                for j, line in enumerate(fs):
                    line = line.strip()
                    keys_counter_per_summ = self.get_keys_from_kv_string(line)
                    id = chart_ids[j]
                    summary_id = summary_ids[j]

                    # if summary_id == "04_01-20":
                    #     import pdb; pdb.set_trace()
                    # SHALLOW COPIES
                    if id in counter_by_chart:
                        #counter_by_chart[id] += keys_counter_per_summ
                        counter_by_chart[id] = counter_by_chart[id].copy() + keys_counter_per_summ.copy()
                    if id not in counter_by_chart:
                        counter_by_chart[id] = keys_counter_per_summ.copy()

                    counter_by_summary[summary_id] = keys_counter_per_summ.copy()


                    counter_all = counter_all.copy() + keys_counter_per_summ.copy()
        #import pdb; pdb.set_trace()
        return counter_by_chart, counter_all, counter_by_summary


    def generalize_counts(self, input_var):
        """
        height approx: height approximation for a single bar
        group names: grouping of bars and referring to them
        group heights: describing the height of the bar group
        add: additive relation
        mul: multiplicative relation
        slope: numerical value of slope
        height mean: the mean height of all bars in the chart
        other ops: another operation/description
        """
        generalized = {"height_approx": 0, "group_names": 0, "group_heights": 0, "mul": 0, "add": 0, "slope": 0,
                       "height_mean": 0, "other_ops": 0}
        if isinstance(input_var, list):
            d = Counter(input_var)
        if isinstance(input_var, dict):
            d = input_var
        for label, count in d.items():
            if label.startswith("GRY"):
                generalized["group_heights"] += count
            elif label.startswith("GR"):
                generalized["group_names"] += count
            elif label.startswith("MUL"):
                generalized["mul"] += count
            elif label.startswith("ADD"):
                generalized["add"] += count
            elif label.startswith("SLOPE"):
                generalized["slope"] += count
            elif label.startswith("YMEAN"):
                generalized["height_mean"] += count
            elif label.endswith("APPROX"):
                generalized["height_approx"] += count
            else:
                generalized["other_ops"] += count
        return generalized

    def dump_counter_into_file(self, counter_dict, fname):
        out_fname = os.path.join(root_repo_dir, "stats_analysis/lrec", fname + ".json")
        with open(out_fname, "w") as fout:
            json.dump(counter_dict, fout)
        print("... Wrote file ", out_fname)

    def gate_portion_inferred(self, counter_of_labels):
        """ Calculate the portion of labels: approximated bar height, inferred labels and approx. height,
         and inferred labels (without approx height) """
        nlab = sum(counter_of_labels.values())
        portion_approx, portion_approx_other_inferred, portion_other_inferred = 0, 0, 0

        for l, c in counter_of_labels.items():
            # any kind of inferred
            if (l.startswith("MUL") or l.startswith("ADD") or l.startswith("OTHEROPERATION") or
            l.startswith("SLOPE") or l.startswith("YMEAN") or l.startswith("GR") or
            l.endswith("APPROX")):
                portion_approx_other_inferred += c

            # height approximation
            if l.endswith("APPROX"):
                portion_approx += c

            # inferred without height approximation
            if (l.startswith("MUL") or l.startswith("ADD") or l.startswith("OTHEROPERATION") or
                l.startswith("SLOPE") or l.startswith("YMEAN") or l.startswith("GR")):
                portion_other_inferred += c

        #import pdb; pdb.set_trace()
        #print("Portion of height approx given all entities", portion_approx/nlab)
        #print("Portion of all inferred entities given all entities", portion_approx_other_inferred/nlab)
        #print("Portion of inferred (no height approx) entities given all entities",portion_other_inferred/nlab)
        return portion_approx/nlab, portion_approx_other_inferred/nlab, portion_other_inferred/nlab

    def per_chart_count(self, topic_id, chart_id):
        # count the number of basic and inferred labels for every chart
        # inferred labels: MUL*, ADD*, *APPROX, OTHEROPERATION, SLOPE*,  GRY*, YMEAN, x_axis_trend (?)
        chart_stats = Counter()
        for summary_ID, tup in self.topicwise[topic_id].items():
            if summary_ID.startswith(chart_id):
                kv = tup[0]  # key-value in copy format - check frequency
                # chart stats is a Counter dictionary; these can be added: if the keys overlap,
                # their values will be added
                keys_counter_per_summ = self.get_keys_from_kv_string(kv)
                r_approx, r_all_inferred, r_inferred_without_approx = self.gate_portion_inferred(keys_counter_per_summ)
                if r_inferred_without_approx > 0:
                    print("--Approx", summary_ID)
                #if r_all_inferred > 0:
                #    print("All inferred", summary_ID)
                #if r_approx > 0:
                #    print("++Approx", summary_ID)
                chart_stats = chart_stats + keys_counter_per_summ

        return chart_stats

    def per_chart_analysis(self, label_counter_bychart_or_summary, summary_or_chart):
        """ for the given chart or summary label distribution print out some stats about inferred entities """
        summaries_at_least_one_inferred_without_height = []  # summary OR chart IDs
        summaries_at_least_one_height_approx = []  # summary OR chart IDs
        summaries_at_least_one_any_inferred = []
        n_all = len(label_counter_bychart_or_summary)  # number of all summaries or charts
        for _id, di in label_counter_bychart_or_summary.items():

            r_approx, r_all_inferred, r_inferred_without_approx = self.gate_portion_inferred(di)
            if r_inferred_without_approx > 0:
                summaries_at_least_one_inferred_without_height.append(_id)
                #print("--Approx", _id, r_inferred_without_approx)
                #import pdb; pdb.set_trace()
            if r_all_inferred > 0:
            #   print("All inferred", chart_id)
                summaries_at_least_one_any_inferred.append(_id)

            if r_approx > 0:
            #   print("++Approx", chart_id)
                summaries_at_least_one_height_approx.append(_id)
        n1 = len(summaries_at_least_one_inferred_without_height)
        print("Number of %s with at least 1 inferred entity that isn't approx height: %d (%f)" %
              (summary_or_chart, n1, n1/n_all))
        n2 = len(summaries_at_least_one_height_approx)
        print("Number of %s with at least 1 approx height: %d (%f)" %
              (summary_or_chart, n2, n2/n_all))
        n3 = len(summaries_at_least_one_any_inferred)
        print("Number of %s with at least 1 any inferred entity: %d (%f)" %
              (summary_or_chart, n3, n3/n_all))

        # TODO: find good examples
        print(summaries_at_least_one_inferred_without_height)



    def ratio_basic_inferred(self, dct):
        # based on the label counter, calculate the ratio between inferred and basic labels
        inferred, basic = Counter(), Counter()
        #import pdb; pdb.set_trace()
        for label, count in dct.items():
            if (label.startswith("MUL") or label.startswith("ADD") or label.startswith("OTHEROPERATION") or
                label.startswith("SLOPE") or label.startswith("YMEAN") or label.startswith("GR") or
                label.endswith("APPROX")):  # parentheses :)
                inferred[label] = count
            else:
                basic[label] = count

        r = sum(inferred.values()) / sum(dct.values())
        v = sum(inferred.values()) / sum(basic.values())
        print("The ratio between inferred and all labels is %f" % r)
        print("The ratio between inferred and basic labels is %f" % v)
        #import pdb; pdb.set_trace()
        generalized_counter = self.generalize_counts(inferred)
        #import pdb; pdb.set_trace()
        print("Corpus stats, generalized, inferred label counts", generalized_counter)
        return r


corpus = Corpus()

#import pdb; pdb.set_trace()

"""
About 8% of all labels are inferred (and not basic)
The ratio between inferred and basic labels is 9/100.

The ratio between inferred and all labels is 0.113921
The ratio between inferred and basic labels is 0.128568
"""
