"""
This script relies on delexicalize.py where the XML corpus processing and delexicalization happens.

This script has functions that performs the following:
- given a chart, prepare an exhaustive content plan of possible key-value pairs
- pair up the summaries from delexicalize.py with these content plans
- write these into files (txt and json) into
    corpora_v02/keyvalue/complete/copy_tgt
    corpora_v02/keyvalue/complete/copy_tgt_set
    corpora_v02/keyvalue/complete/exhaustive
"""
import json
import itertools
from delexicalize import open_delex_key_value

class DelexicalizedCorpus():
    def __init__(self):

        self.xml_file = "/home/iza/chart_descriptions/data/chart_summaries_b01_toktest2.xml"
        self.topicwise = open_delex_key_value(self.xml_file)
        self.data_dir = "/home/iza/chart_descriptions/corpora_v02/keyvalue/complete"
        self.plot_info_path = "/home/iza/chart_descriptions/data/json_data/chartID2plotInfo.json"

        #print(self.topicwise["11"].keys())
        self.generate_exhaustive()


    def create_combinations(self, bnames, bheights, lnames):
        """
        bnames : list of str : barnames ordered - descending
        bheights : list of floats : bar heights ordered - desceding
        lnames : list of str : names of labels

        return the keys and values of MUL, ADD, GROUP, and APPROX
        """

        triple = list((lnames[i], bheights[i], bnames[i]) for i in range(len(bnames)))
        relations = {}

        # pairs : bigrams
        for (l1, h1, n1), (l2, h2, n2) in itertools.combinations(triple, 2):
            relations["MUL"+l1+l2] = round(h1 / h2)
            relations["ADD" + l1 + l2] = round(h1 - h2)
            relations["GR"+l1+l2] = n1 + " and " + n2
            relations["GRY" + l1 + l2] = round((h1+h2)/2)

            if "X" + l1 not in relations:
                relations["X"+l1] = n1

            if "X" + l2 not in relations:
                relations["X"+l2] = n2

            if "Y" + l1 not in relations:
                relations["Y" + l1] = h1

            if "Y" + l2 not in relations:
                relations["Y" + l2] = h2

            if "Y" + l1 + "APPROX" not in relations:
                relations["Y" + l1 + "APPROX"] = round(h1)
            if "Y" + l2 + "APPROX" not in relations:
                relations["Y" + l2 + "APPROX"] = round(h2)

        if len(bnames) < 4:
            return relations

        # group relations for a three-bar groups only in case a chart has more than 3 bars!
        for (l1, h1, n1), (l2, h2, n2), (l3, h3, n3) in itertools.combinations(triple, 3):
            relations["GR" + l1 + l2 + l3] = n1 + ", " + n2 + " and " + n3
            relations["GRY" + l1 + l2 + l3] = round((h1 + h2 + h3) / 3)

        return relations

    def generate_exhaustive(self):
        # for every chart, generate a set of possible key-value pairs
        # use the json file with all plot data
        with open(self.plot_info_path, "r") as jf:
            self.info = json.load(jf)

        print(self.info.keys())

        self.minor_topicIDs = set(summaryID[:-3] for greatTopicID, summaryIDs in self.topicwise.items() for summaryID, tup in summaryIDs.items())
        print(self.minor_topicIDs)

        # cut and reassign given the number of bars
        names = ["HIGHEST", "SECOND", "THIRD", "FOURTH", "FIFTH", "LEAST"]
        # to generate: bar names, bar heights, their approximations (round), group names, group heights
        # XFOURTH, YFORTH, "YFOURTHAPPROX", MULSECONDTHIRD, ADDHIGHESTFOURTH
        # GRYSECONDTHIRD, GRSECONDTHIRD
        # axis names: XLABEL, YLABEL
        # title: PLOTTITLE
        # bar count: COUNT

        for minor_topicID in self.minor_topicIDs:
            current = self.info[minor_topicID]
            plottitle = current["general_figure_info"]["title"]["text"]
            xaxis = current["general_figure_info"]["x_axis"]["label"]["text"]
            yaxis = current["general_figure_info"]["y_axis"]["label"]["text"]
            barnames = current["models"][0]["x"]
            heights = current["models"][0]["y"]
            count = len(barnames)

            cnames = names[:count-1] + [names[-1]]
            #print(barnames, count, cnames)
            # sorted heights and barnames in descending order
            sorted_heights, sorted_barnames = (list(t) for t in zip(*sorted(zip(heights,barnames), reverse=True)))

            keyvalues = self.create_combinations(sorted_barnames, sorted_heights, cnames)
            keyvalues["PLOTTITLE"] = plottitle
            keyvalues["XAXIS"] = xaxis
            keyvalues["YAXIS"] = yaxis
            keyvalues["COUNT"] = count

            print(keyvalues["PLOTTITLE"], len(keyvalues))





c = DelexicalizedCorpus()

"""
  "y_axis": "YLABEL",
  "x_axis_label_least_value": "XLEAST",
  "x_axis_label_highest_value": "XHIGHEST",
  "x_axis_label_Scnd_highest_value": "XSECOND",
  "y_axis_least_value_val": "YLEAST",
  "y_axis_Scnd_highest_val": "YSECOND",
"""