"""
Create parallel data for seq2seq training

Source: content plan
Target: chart summary (delex)

The content plan is sequence of labels, i.e. entities such as bar names, heights and relations. Delex.
"""

import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("-json", required=False, help="json of a topic/chart with delex summaries", default="/home/iza/chart_descriptions/corpora_v02/delexicalized/delex_09_01.json")
args = vars(parser.parse_args())

json_file = args["json"]

with open("lex_delex.json", "r") as jf:
    all_lex_delex = json.load(jf)
lex_delex = {**all_lex_delex["bar_information"], **all_lex_delex["topic_information"]}

def open_plan_save(json_name):
    """ Extract the content plan from summaries and write into a tab-separated file """
    new_file = json_name[:-5] + "_parallel.txt"

    with open(json_name, "r", encoding="utf-8") as jf:
        data=json.load(jf)

    with open(new_file, "w", encoding="utf-8") as parallel_file:
        for summaryID, summaryText in data.items():
            # extract entities as a sequence
            tokens = summaryText.split()
            content_plan = []
            for t in tokens:
                if t in lex_delex.values():
                    content_plan.append(t)
            print(content_plan, tokens)
            parallel_file.write(" ".join(content_plan) + "\t" + summaryText + "\n")

open_plan_save(json_file)