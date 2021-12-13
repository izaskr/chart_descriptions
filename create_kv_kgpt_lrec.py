"""
This script rewrites the key-value format files into a format readable for KGPT https://github.com/wenhuchen/KGPT

To start with, rewrite the exhaustive source into a json list

Required format
list of dictionaries(1)
Each dictionary(1) has the keys "id", "text", "kbs"
    "id": int (from 0 to len-1, can overlap between train/val/test)
    "text": a list of len 1, the element is a lexicalized single document
    "kbs": dictionary(2)
        (the "chart_identifier" is an identifier for MR)
        "chart_identifier": a list of len 3
                element 0: restaurant name (here char title)
                element 1: repeat element 0
                element 2: a list of lists
                    each list is of len 2 and contains [key1, value1]

TODO: check the format in train/val/test json for e2enlg again

data_dir = "/home/skrjanec/chart_descriptions/corpora_v02/keyvalue/complete_different_split/exh"

AND

chart_descriptions/corpora_v02/keyvalue/complete_fixed_tok/exh



        for label, count in d.items():
            if (label.startswith("MUL") or label.startswith("ADD") or label.startswith("OTHEROPERATION") or
                label.startswith("SLOPE") or label.startswith("YMEAN") or label.startswith("GRY") or
                label.endswith("APPROX")):  # parentheses :)
                inferred[label] = count

keyvalue/KGPT
    ├──basic
    │   ├──mixed
    │   └──exclusive
    └──basic_plus_inferred
        ├──mixed
        └──exclusive

"""

import json
import os
import git

repo = git.Repo('.', search_parent_directories=True)
root_repo_dir = repo.working_tree_dir


# load the data given the split_name arg, load the source and the target file plus the ID files
# train and val IDs list are in chart_descriptions/train_val_ids_exclusive.json
# for val and test create also txt files with groups by source separated with an empty line between

def keep_or_ignore(key, version):
    if version == "inferred":
        return True
    if version == "basic":
        if (key.startswith("MUL") or key.startswith("ADD") or key.startswith("OTHEROPERATION") or
                key.startswith("SLOPE") or key.startswith("YMEAN") or key.startswith("GR") or key.endswith("APPROX")):
            return False
        else:
            return True


def reformat_src(src_string, version):
    """
    Clean the src_string, map it into key-value pairs and make a subset if needed
    """
    title = ""
    list_keys, list_values, list_keyvalues = [], [], []
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
            title = title.replace("\\\\\\ ", "")

        # cleaning
        keep_gate = keep_or_ignore(k, version)
        if keep_gate:
            list_keyvalues.append([k, v])
    # sort the list, it's a list of lists [ [key, value], [key, value] ... ]
    list_keyvalues = sorted(list_keyvalues)
    return list_keyvalues, title


def write_json_files_and_mr(list_of_dictionaries, output_dir, split):
    """
    Write the list of dictionaries into a file
    """
    all_mr_ids = set()
    chart_id_mr_id = []
    out_file_path = os.path.join(output_dir, split + ".json")
    with open(out_file_path, "w") as fout:
        json.dump(list_of_dictionaries, fout)
    print("... Wrote file %s" % out_file_path)

    # Write also the mapping of MR IDs
    # in the file, there will be the MR IDs _separator_ and the chart ID
    # get the chart ID - read in the ID from the split
    if split in {"train", "val", "test"}:
        with open(os.path.join(root_repo_dir, "train_val_ids_exclusive.json"), "r", encoding="utf8") as fid:
            ids = json.load(fid)[split]

    # if split == "test":
    #     # open the file with test ids
    #     ids = []
    #     with open(
    #         os.path.join(root_repo_dir, "corpora_v02/keyvalue/complete_different_split_fixed/exh/test_ids.txt"),
    #             "r") as test_f:
    #         for line in test_f:
    #             ids.append(line.strip())

    assert len(ids) == len(list_of_dictionaries)
    # write into file
    ids_file_path = os.path.join(output_dir, split + ".ids")
    with open(ids_file_path, "w") as idout:
        for j, d in enumerate(list_of_dictionaries):
            chart_id = ids[j]
            mr_id = list(d["kbs"].keys())[0]
            s = chart_id + "\tSEP\t" + mr_id + "\n"
            idout.write(s)
            chart_id_mr_id.append((chart_id, mr_id))
            all_mr_ids.add(mr_id)

    print("... Wrote file %s" % ids_file_path)
    if split == "train":
        return None
    # KGPT expects a slightly different format for val and test files:
    # all the target documents that come from the same MR should be joined under the same instance ("id")
    arranged_final_parallel = []
    separate_dicts_by_mr_id = dict()
    new_ids_list = []
    all_mr_ids = sorted(list(all_mr_ids))
    for id_mr in all_mr_ids:
        for k, d in enumerate(list_of_dictionaries):
            instance_id = d["id"]
            text = d["text"][0]
            kbs = d["kbs"]
            current_mr_id = list(d["kbs"].keys())[0]
            if id_mr != current_mr_id:
                continue
            current_mr_id = list(d["kbs"].keys())[0]
            current_chart_id = chart_id_mr_id[k][0]

            if current_mr_id in separate_dicts_by_mr_id:
                separate_dicts_by_mr_id[current_mr_id]["text"].append(text)
                new_ids_list.append((current_chart_id, current_mr_id))

            if current_mr_id not in separate_dicts_by_mr_id:
                separate_dicts_by_mr_id[current_mr_id] = d
                new_ids_list.append((current_chart_id, current_mr_id))

    # now arrange these by-mr dictionaries into a list
    list_of_dictionaries_arranged = []
    for m, dictionary in separate_dicts_by_mr_id.items():
        list_of_dictionaries_arranged.append(dictionary)

    # write this new arranged list of dict into a json file
    out_file_path2 = os.path.join(output_dir, split + "_arranged.json")
    with open(out_file_path2, "w") as fout:
        json.dump(list_of_dictionaries_arranged, fout)
    print("... Wrote file %s" % out_file_path2)

    # write the IDs as well
    #import pdb; pdb.set_trace()
    with open(os.path.join(output_dir, split + "_arranged_ids.txt"), "w") as f:
        for (cid, mrid) in new_ids_list:
            f.write(str(cid) + "\tSEP\t" + str(mrid) + "\n")
    print("... Wrote file %s" % os.path.join(output_dir, split + "_arranged_ids.txt"))

    # write also the target sides into a separate .txt file
    txt_file_out = os.path.join(output_dir, split + ".txt")
    with open(txt_file_out, "w") as txt_out:
        for d in list_of_dictionaries_arranged:
            for document in d["text"]:
                txt_out.write(document + "\n")
            txt_out.write("\n")  # empty line between different MRs
    print("... Wrote file %s" % txt_file_out)


def read_in_data(file_path, split, version, output_dir, write):
    assert version in {"basic", "plusinferred"}

    all_sources = dict()
    all_sources_list = []
    all_mr_ids = set()
    final_parallel = []
    src_file_path = os.path.join(root_repo_dir, file_path, split + "_src_e.txt")
    tgt_file_path = os.path.join(root_repo_dir, file_path, split + "_tgt_e.txt")
    # read in source
    with open(src_file_path, "r") as src_f:
        print("opening %s" % src_file_path)
        for line in src_f:
            line = line.strip()
            all_sources_list.append(line)
            if line not in all_sources:
                all_sources[line] = str(len(all_sources))

            current_keyvalues_list, title = reformat_src(line, version)
            current_d = {"id": len(final_parallel), "text": [], "kbs": {"X": [title, title, current_keyvalues_list]}}

            final_parallel.append(current_d)

    # read in target
    with open(tgt_file_path, "r") as tgt_f:
        for i, line in enumerate(tgt_f):
            line = line.strip()
            tgt_text = line.replace("\\n", "")  # get rid of these newline symbols mid-text
            tgt_text = tgt_text.replace("\\", "")
            #old_d = final_parallel[i]
            final_parallel[i]["text"] = [tgt_text]
            temp = final_parallel[i]["kbs"]["X"]
            #import pdb; pdb.set_trace()
            chart_identifier = all_sources[all_sources_list[i]]
            new_d = {chart_identifier: temp}
            final_parallel[i]["kbs"] = new_d


    #import pdb; pdb.set_trace()  # looks okay so far except for \\
    if write:
        write_json_files_and_mr(final_parallel, output_dir, split)

    return final_parallel, all_sources, all_sources_list


read_in_data("corpora_v02/keyvalue/complete_different_split_fixed/exh", "train", "basic",
             "corpora_v02/keyvalue/KGPT/exclusive/basic", True)

# <story story_id="10_02c-23">
