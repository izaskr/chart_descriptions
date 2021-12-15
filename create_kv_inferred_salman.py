"""
Salman has run experiments with our dataset with only the basic information from charts.
Let's include also the inferred information.
First extract it for every chart from the exhaustive source format
"""
import os
import git
import json

repo = git.Repo('.', search_parent_directories=True)
root_repo_dir = repo.working_tree_dir


def keep_or_ignore(key, version):
    """ Keep only key-value pairs that are inferred, so not basic """
    # if version == "inferred":
    #     return True
    # if version == "basic":
    #     if (key.startswith("MUL") or key.startswith("ADD") or key.startswith("OTHEROPERATION") or
    #             key.startswith("SLOPE") or key.startswith("YMEAN") or key.startswith("GR") or key.endswith("APPROX")):
    #         return False
    #     else:
    #         return True

    if (key.startswith("MUL") or key.startswith("ADD") or key.startswith("OTHEROPERATION") or
            key.startswith("SLOPE") or key.startswith("YMEAN") or key.startswith("GR") or key.endswith("APPROX")):
        return True
    else:
        return False



def reformat_src(src_string, version):
    """
    Clean the src_string, map it into key-value pairs and make a subset if needed
    """
    title = ""
    list_keys, list_values, list_keyvalues = [], [], []
    src_string = src_string.strip()  # remove any spaces or newlines at the beginning or end
    s_split = src_string.split("],")
    for kv_pair in s_split:
        if not kv_pair:
            continue
        k, v = kv_pair.split("[")[0], kv_pair.split("[")[1]  # get the key and its value
        k = k.strip()
        if len(v) == 0:
            import pdb; pdb.set_trace()
        if v[-1] == "]":
            v = v[:-1]

        if k == "PLOTTITLE":
            title = v
            title = title.strip()
            title = title.replace("\\n", "")
            title = title.replace("\\", "")
            title = title.replace("\\\\\\ ", "")

        # cleaning
        keep_gate = keep_or_ignore(k, version)
        if keep_gate:
            list_keyvalues.append([k, v])
    # sort the list, it's a list of lists [ [key, value], [key, value] ... ]
    #list_keyvalues = sorted(list_keyvalues)
    # into a dictionary
    dict_keyvalues = {k:v for (k,v) in list_keyvalues}
    return dict_keyvalues, list_keyvalues, title


def read_in_ids(path_to_id_file, type_id_file, split):
    """ Read in the summary IDs from the file and return them as a list """
    if type_id_file == "json":
        with open(path_to_id_file, "r") as f:
            data = json.load(f)[split]  # a list of summary IDs, e.g. "09_01a-06"
    if type_id_file == "txt":
        with open(path_to_id_file, "r") as f:
            data = []
            for line in f:
                data.append(line.strip())
    print("... Loaded the IDs from ", path_to_id_file)
    return data


def read_in_data(file_path, split, version, output_dir, path_to_id_file0, type_id_file, write):
    assert version in {"basic", "plusinferred"}
    if type_id_file == "txt":
        path_to_id_file = os.path.join(root_repo_dir, path_to_id_file0, "ids_" + split + "_e.txt")
    if type_id_file == "json":
        path_to_id_file = os.path.join(root_repo_dir, path_to_id_file0)
    print("--------- id file",path_to_id_file)
    list_of_summary_ids = read_in_ids(path_to_id_file, type_id_file, split)
    chartID_2_additional_info = dict()

    src_file_path = os.path.join(root_repo_dir, file_path, split + "_src_e.txt")
    # read in source
    with open(src_file_path, "r") as src_f:
        print("opening %s" % src_file_path)
        for i, line in enumerate(src_f):
            line = line.strip()

            current_keyvalues_dict, current_keyvalues_list, title = reformat_src(line, version)
            current_summary_id = list_of_summary_ids[i]
            current_chart_id = current_summary_id[:-3]  # go from "09_01a-06" to "09_01a"
            if current_chart_id not in chartID_2_additional_info:
                chartID_2_additional_info[current_chart_id] = current_keyvalues_dict

    #import pdb; pdb.set_trace()  # looks okay so far except for \\
    # if write:
    #     # output_dir, split)
    #     # write the chartID_2_additional_info into file
    #     with open(os.path.join(output_dir, "chartID2additional_info_salman.json"), "w") as fout:
    #         json.dump(chartID_2_additional_info, fout)

    #import pdb; pdb.set_trace()

    return chartID_2_additional_info, list_of_summary_ids

# MIXED
# source files: "complete_fixed_tok/exh"
# id files: "corpora_v02/keyvalue/complete_fixed_tok/exh"
# id type: txt
#

splits = ["train", "val", "test"]
topic_combinations = {"mixed": ["corpora_v02/keyvalue/complete_fixed_tok/exh",
                                "corpora_v02/keyvalue/complete_fixed_tok/exh",
                                "txt"],
                      "exclusive": ["corpora_v02/keyvalue/complete_different_split_fixed/exh",
                                    os.path.join(root_repo_dir, "train_val_ids_exclusive.json"),
                                    "json"]}



all_ids = {"mixed": {}, "exclusive": {}}
all_chart2info = {}

for combination_name, lst in topic_combinations.items():
    for split_name in splits:
        chart2info, summary_ids = read_in_data(lst[0], split_name, "plusinferred",
                     "corpora_v02/keyvalue", lst[1], lst[2], True)
        all_ids[combination_name][split_name] = summary_ids
        for chartid, dictionary in chart2info.items():
            if chartid not in all_chart2info:
                all_chart2info[chartid] = dictionary


#import pdb; pdb.set_trace()

with open(os.path.join(root_repo_dir, "corpora_v02/keyvalue", "splits_combinations_ids.json"), "w") as fid:
    json.dump(all_ids, fid)

with open(os.path.join(root_repo_dir, "corpora_v02/keyvalue", "chartID2additional_info_salman.json"), "w") as fout:
    json.dump(all_chart2info, fout)

# the reamd is reamde_salman.txt
