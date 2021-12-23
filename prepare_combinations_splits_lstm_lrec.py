"""
To run the training and testing (with multi-reference bleu and single-reference BLEURT)

"""

import os
import git
import json

repo = git.Repo('.', search_parent_directories=True)
root_repo_dir = repo.working_tree_dir


# path to exhaustive source
# mixed:
mixed_source_dir = os.path.join(root_repo_dir, "corpora_v02/keyvalue/complete_fixed_tok/exh")

# exclusive:
exclusive_source_dir = os.path.join(root_repo_dir, "corpora_v02/keyvalue/complete_different_split_fixed/exh")

source_suffix = "_src_e.txt"


def keep_or_ignore(key, version):
    """ Keep only key-value pairs that are inferred, so not basic """
    if version == "basic_inferred":
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
    Version is either basic or basic_inferred
    """
    list_keys, list_values, list_keyvalues = [], [], []
    src_string = src_string.strip()  # remove any spaces or newlines at the beginning or end
    s_split = src_string.split("],")
    for kv_pair in s_split:
        if not kv_pair:
            continue

        k, v = kv_pair.split("[")[0], kv_pair.split("[")[1]  # get the key and its value
        k = k.strip()
        v = v.strip()
        if len(v) == 0:
            import pdb; pdb.set_trace()
        if v[-1] == "]":
            v = v[:-1]

        if k == "PLOTTITLE":
            v = v.strip()
            v = v.replace("\\n", "")
            v = v.replace("\\", "")
            v = v.replace("\\\\\\ ", "")
        # filtering
        keep_gate = keep_or_ignore(k, version)
        if keep_gate:
            list_keyvalues.append([k, v])

    list_keyvalues_str = []
    for (_key, _value) in list_keyvalues:
        _s = _key + "[" + _value + "]"
        list_keyvalues_str.append(_s)
    new_str = ", ".join(list_keyvalues_str)
    return new_str


def read_in_ids(path_to_id_file):
    """ read in SUMMARY ids, return a list of CHART ids """
    chart_ids = []
    with open(path_to_id_file) as fid:
        for line in fid:
            line = line.strip()
            if line:
                chart_ids.append(line)
    print("Number of summaries %d" % len(chart_ids))
    return chart_ids


def read_in_exhaustive_source_write_file(path_to_source, split, test_domain, scope, write):
    """ Read in the source, read the IDs (if split is test)
    Open the out file
    Given the scope (basic or basic_inferred), clean the source string and write it into the out file
    """
    out_file = open(os.path.join(root_repo_dir, "corpora_v02/keyvalue/LSTM_lrec", scope + "_" + test_domain,
                                 split + "_src.txt"), "w")

    if split == "test":
        out_file = open(os.path.join(root_repo_dir, "corpora_v02/keyvalue/LSTM_lrec", scope + "_" + test_domain,
                                     split + "_src_unique.txt"), "w")

    # load IDs - just for the test set
    with open(os.path.join(root_repo_dir, "splits_combinations_ids.json")) as fid:
        test_ids = json.load(fid)[test_domain]["test"]  # summary IDs

    written_check = set()
    with open(path_to_source, "r") as fs:
        for j, line in enumerate(fs):
            new_line = reformat_src(line, scope)

            if write:
                if split in {"train", "val"}:
                    out_file.write(new_line + "\n")
                if split == "test":
                    if test_ids[j][:-3] in written_check:
                        continue
                    if test_ids[j] not in written_check:
                        out_file.write(new_line + "\n")
                        written_check.add(test_ids[j][:-3])
    out_file.close()

    print("... Wrote file ", out_file)


splits = ["train", "val", "test"]
test_domains = ["mixed", "exclusive"]
scopes = ["basic", "basic_inferred"]

src_paths = {"mixed": mixed_source_dir, "exclusive": exclusive_source_dir}

if __name__ == "__main__":
    for test_domain in test_domains:
        for scope in scopes:
            for split in splits:
                source_file = os.path.join(src_paths[test_domain], split + source_suffix)
                read_in_exhaustive_source_write_file(source_file, split, test_domain, scope, True)
