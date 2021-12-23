"""
For his experiments with the Chart2Text model Salman needs delexicalized summaries:
for the mixed aas well as exclusive splits.


"""
import os
import json

output_dir = "corpora_v02/keyvalue/temp2"
delex_dir = "corpora_v02/keyvalue/complete_fixed_tok/exh"

# open the ids
def open_ids(path_to_ids):
    # load the ids as a list, combo is either "mixed" or "exclusive"
    #assert combo.lower() in {"mixed", "exclusive"}
    with open(path_to_ids, "r") as fid:
        ids = json.load(fid)  # will contain IDs for the train/val/test split
    return ids


def read_in_mixed_delexicalized(path_to_file, list_of_ids):
    """ Mixed data has already been delexicalized, so we will read it in here
    and pair it up with its summary IDs, return a dictionary
    """
    summaries = []
    with open(path_to_file, "r")  as f:
        for line in f:
            line = line.strip()
            line = line.replace("\\n", "")
            line = line.replace("\\\\\\ ", "")
            if line:
                summaries.append(line)

    di = {summary_id: summary for (summary_id, summary) in zip(list_of_ids, summaries)}
    return di


def read_in_exclusive_ids_write_output(list_of_ids_exclusive, split, summaryID2delex_summary):
    out_file = os.path.join(output_dir, "exclusive", split + ".txt")
    with open(out_file, "w") as fout:
        for i in list_of_ids_exclusive:
            summary = summaryID2delex_summary[i]
            fout.write(summary + "\n")

    print("Wrote file ", out_file)


all_ids = open_ids("splits_combinations_ids.json")
mixed_all_ids = all_ids["mixed"]
exclusive_all_ids = all_ids["exclusive"]

splits = ["train", "val", "test"]

# collect all mappings summaryID-to-delex_summary
mappings = {}

for split in splits:
    d = read_in_mixed_delexicalized(os.path.join(delex_dir, split + "_tgt_f.txt"), mixed_all_ids[split])
    mappings = {**mappings, **d}  # merge the two dictionaries

print("number of summaries with a mapping ", len(mappings))

# now iterate over the given ids of EXCLUSIVE
for split in splits:
    read_in_exclusive_ids_write_output(exclusive_all_ids[split], split, mappings)




