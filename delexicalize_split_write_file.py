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

        # major/great topic id: 14
        # minor topic id: 03_01, 07_01b
        # summary id: '14_01-03', '14_01b-15'

        #self.minor_topicIDs = set(summaryID[:-3] for greatTopicID, summaryIDs in self.topicwise.items() for summaryID, tup in summaryIDs.items())
        self.major2minor = {k: set() for k in self.topicwise}

        # I need: minor topic id 2 summaryIDs
        self.minorTopicID2summaryID2formats = {}
        for greatTopicID, summaryIDs in self.topicwise.items():
            for summaryID, tup in summaryIDs.items():
                current_minor = summaryID[:-3]
                self.major2minor[greatTopicID].add(current_minor)
                if current_minor in self.minorTopicID2summaryID2formats:
                    self.minorTopicID2summaryID2formats[current_minor][summaryID] = tuple()

                if current_minor not in self.minorTopicID2summaryID2formats:
                    self.minorTopicID2summaryID2formats[current_minor] = {summaryID : tuple()}

        #print(self.minorTopicID2summaryID2formats)
        self.data_dir = "/home/iza/chart_descriptions/corpora_v02/keyvalue/complete"
        self.plot_info_path = "/home/iza/chart_descriptions/data/json_data/chartID2plotInfo.json"

        # the function below will populate self.minorTopicID2summaryID2formats
        self.generate_exhaustive()

        #for s, ios in self.minorTopicID2summaryID2formats["11_02"].items():
        #    print(s, ios)
        self.io_types = ["copy_tgt", "copy_tgt_set", "exhaustive"]
        self.split_into_train_val_test()

        for iot in self.io_types:
            #self.write_output_parallelF(iot)
            self.write_output_tab_and_parallel(iot)


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

        #print(self.info.keys())

        # cut and reassign given the number of bars
        names = ["HIGHEST", "SECOND", "THIRD", "FOURTH", "FIFTH", "LEAST"]
        # to generate: bar names, bar heights, their approximations (round), group names, group heights
        # XFOURTH, YFORTH, "YFOURTHAPPROX", MULSECONDTHIRD, ADDHIGHESTFOURTH
        # GRYSECONDTHIRD, GRSECONDTHIRD
        # axis names: XLABEL, YLABEL
        # title: PLOTTITLE
        # bar count: COUNT

        for minor_topicID in self.minorTopicID2summaryID2formats:
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
            #print(keyvalues)

            # keyvalues is the same for every summary in minorTopicID
            # to every story id in the topicwise[minorTopicID], change the value of current value to (current-value,keyvalues)

            # prepare keyvalues into a a string
            # 'key[value], key[value]
            exhaustive = ""
            for k, v in keyvalues.items():
                exhaustive = exhaustive + k + "[" + str(v) + "], "

            exhaustive = exhaustive[:-2]

            current_majorTopicID = minor_topicID[:2]
            for summaryID in self.minorTopicID2summaryID2formats[minor_topicID]:
                #for x in self.topicwise[current_majorTopicID][summaryID]:
                #    print(type(x),x)
                # self.topicwise[current_majorTopicID][summaryID] is a tuple of 4
                # 1 : key-value-plan-COPY
                # 2: fully lexicalized summary
                # 3: key-value-plan-COPY-SET
                # 4: delexicalized summary
                kv_copy, lex_sum, kv_copy_set, delex_sum = self.topicwise[current_majorTopicID][summaryID]

                # for every summary in the current minor topic ID, create a value
                # 1: key-value-plan-COPY
                # 2: key-value-plan-COPY-SET
                # 3: exhaustive
                # 4: lexicalized summary
                # 5: delexicalized summary
                self.minorTopicID2summaryID2formats[minor_topicID][summaryID] = (kv_copy,kv_copy_set,exhaustive,lex_sum,delex_sum)

    def split_into_train_val_test(self):
        # split type = 2
        # each minor topic is split between train/val and test, so it appears in both main splits
        print(self.major2minor)
        assigned = set()
        train_val_minor_IDs, test_minor_IDs, shared = set(), set(), set()
        for major, minors in self.major2minor.items():
            minor_endings_special = [x for x in minors if x[-1] in {"a", "b", "c"}]

            for spc in minor_endings_special:
                if "c" in spc:
                    shared.add(spc)
                    assigned.add(spc)

            # some charts will be shared among the splits - the topics with 2 charts with the neutral condition
            if len(minors) == 2 and len(minor_endings_special) == 0:
                shared = shared.union(minors)
                assigned = assigned.union(minors)

            if len(minor_endings_special) == 1:
                train_val_minor_IDs.add(minor_endings_special[0])
                assigned.add(minor_endings_special[0])

            # if a major topic has a single minor topic, put it into the train-val split
            if len(minors) == 1:
                (m2,) = minors
                train_val_minor_IDs.add(m2)
                assigned.add(m2)

            # non-neutral: one train-val, one test
            if len(minor_endings_special) == 2:
                t1, t2 = minor_endings_special
                train_val_minor_IDs.add(t1)
                test_minor_IDs.add(t2)
                assigned.add(t1), assigned.add(t2)

            if len(minor_endings_special) == 3:
                t1, t2, t3 = minor_endings_special
                train_val_minor_IDs.add(t1)
                train_val_minor_IDs.add(t3)
                # put t2 among shared
                shared.add(t2)
                assigned.add(t1), assigned.add(t3), assigned.add(t2)

            todo = [x for x in minors if x not in assigned]
            if len(todo) == 3:
                (t1,t2, t3) = minors
                test_minor_IDs.add(t1)
                train_val_minor_IDs.add(t2)
                shared.add(t3)
                assigned = assigned.union({t1,t2,t3})


            todo = [x for x in minors if x not in assigned]
            #print("\t",todo)
            minor_endings_special = [x for x in todo if x[-1] in {"a", "b", "c"}]

            if len(todo) == 2:
                #("todo", todo)
                (t1, t2) = todo
                if len(train_val_minor_IDs) < (500/23):
                    #print("current tv, t",len(train_val_minor_IDs), len(test_minor_IDs))
                    train_val_minor_IDs = train_val_minor_IDs.union({t1,t2})
                else:
                    train_val_minor_IDs.add(t2), test_minor_IDs.add(t1)
                assigned = assigned.union({t1,t2})

            # if len(todo) == 3:
            #     (t1, t2, t3) = todo
            #     train_val_minor_IDs.add(t1), train_val_minor_IDs.add(t3), test_minor_IDs.add(t2)
            #     assigned = assigned.union({t1, t2, t3})

            todo = [x for x in todo if x not in assigned]
            if todo:
                print("a minor topic ID not assigned or missing adding to assigned")

        print("Number of charts in train-val %d, and in test %d" % (len(train_val_minor_IDs), len(test_minor_IDs)))
        print(len(train_val_minor_IDs), len(test_minor_IDs))

        # for each shared minor ID, put 80% into train-val, and the rest into test
        train_summaryIDs, test_summaryIDs = [], []
        for minor_id, all_sumIDs in self.minorTopicID2summaryID2formats.items():
            if minor_id in shared:
                to_split = list(all_sumIDs)
                k = int(0.93 * len(to_split))
                train_summaryIDs += to_split[:k]
                test_summaryIDs += to_split[k:]
            else:
                if minor_id in train_val_minor_IDs:
                    train_summaryIDs += list(all_sumIDs)
                if minor_id in test_minor_IDs:
                    test_summaryIDs += list(all_sumIDs)

        print("Number of summaries in train-val %d, and in test %d" % (len(train_summaryIDs), len(test_summaryIDs)))
        j = round((len(train_summaryIDs) + len(test_summaryIDs)) * 0.2)
        #print("size of val",j)
        self.train_summaryIDs = train_summaryIDs[j:]
        self.val_summaryIDs = train_summaryIDs[:j]
        self.test_summaryIDs = test_summaryIDs


    def write_output_tab_and_parallel(self, iotype): # source \t target
        current_folder = self.data_dir + "/" + iotype + "/"
        # d shows what to take as SRC and what as TGT in formats of self.minorTopicID2summaryID2formats
        d = {"a": (0, 3), "b": (0,4), "c":(1,3), "d":(1,4), "e":(2,3), "f":(2,4)}

        if iotype == "exhaustive":
            scope = ["e", "f"]

        if iotype == "copy_tgt":
            scope = ["a", "b"]

        if iotype == "copy_tgt_set":
            scope = ["c", "d"]

        for x in scope:
            #tsv
            train = current_folder + "tab_train_" + x + ".txt"
            val = current_folder + "tab_val_" + x + ".txt"
            test = current_folder + "tab_test_" + x + ".txt"
            ids = open(current_folder + "ids_test_" + x + ".txt", "w")
            tr_ids = open(current_folder + "ids_train_" + x + ".txt", "w")
            val_ids = open(current_folder + "ids_val_" + x + ".txt", "w")

            #parallel 2 files
            src_train = open(current_folder + "train_src_" + x + ".txt", "w")
            tgt_train = open(current_folder + "train_tgt_" + x + ".txt", "w")
            src_val = open(current_folder + "val_src_" + x + ".txt", "w")
            tgt_val = open(current_folder + "val_tgt_" + x + ".txt", "w")
            src_test = open(current_folder + "test_src_" + x + ".txt", "w")
            tgt_test = open(current_folder + "test_tgt_" + x + ".txt", "w")

            with open(train, "w") as tf:
                for id in self.train_summaryIDs:
                    src_i = d[x][0]
                    tgt_i = d[x][1]
                    cm = id[:-3]
                    src = self.minorTopicID2summaryID2formats[cm][id][src_i]
                    tgt = self.minorTopicID2summaryID2formats[cm][id][tgt_i]
                    tf.write(src + "\t" + tgt + "\n")
                    src_train.write(src + "\n")
                    tgt_train.write(tgt + "\n")
                    tr_ids.write(id + "\n")

            with open(val, "w") as tf:
                for id in self.val_summaryIDs:
                    src_i = d[x][0]
                    tgt_i = d[x][1]
                    cm = id[:-3]
                    src = self.minorTopicID2summaryID2formats[cm][id][src_i]
                    tgt = self.minorTopicID2summaryID2formats[cm][id][tgt_i]
                    tf.write(src + "\t" + tgt + "\n")
                    src_val.write(src + "\n")
                    tgt_val.write(tgt + "\n")
                    val_ids.write(id + "\n")

            with open(test, "w") as tf:
                for id in self.test_summaryIDs:
                    src_i = d[x][0]
                    tgt_i = d[x][1]
                    cm = id[:-3]
                    src = self.minorTopicID2summaryID2formats[cm][id][src_i]
                    tgt = self.minorTopicID2summaryID2formats[cm][id][tgt_i]
                    tf.write(src + "\t" + tgt + "\n")
                    src_test.write(src + "\n")
                    tgt_test.write(tgt + "\n")
                    ids.write(id + "\n")

            ids.close()
            src_test.close(), tgt_test.close(), src_val.close(), tgt_val.close(), src_train.close(), tgt_train.close()
            print("... wrote into files")





    def write_output_parallelF(self, iotype): # 1 file for source, 1 for target
        pass




c = DelexicalizedCorpus()

"""
  "y_axis": "YLABEL",
  "x_axis_label_least_value": "XLEAST",
  "x_axis_label_highest_value": "XHIGHEST",
  "x_axis_label_Scnd_highest_value": "XSECOND",
  "y_axis_least_value_val": "YLEAST",
  "y_axis_Scnd_highest_val": "YSECOND",
"""