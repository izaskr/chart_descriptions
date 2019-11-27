# script for modifying the annotation.json files
# add info about order of categories on x axis
# x_ordered_info: {x_is_ordered: bool, x_order: [] if False, else [A,B,C..]
# add this key:value to "models"[0]

import json

def add_info(filename):
	with open(filename) as f:
		data = json.load(f)

	for i,k in enumerate(data): # iterate over each plot, k are dictionaries with where the value is a list len 1
		title = k["general_figure_info"]["title"]["text"]
		new = {"x_is_ordered":False, "x_order":[], "order": None, "x_is_temporal":False}
		if title == 'Median Salary of Women Per Year':
			new = {"x_is_ordered":True, "x_order":['2000', '2005', '2010', '2015'], "order":"ascending", "x_is_temporal":True}

		if title == 'Median Salary Per Year For Software Engineers with Respect to their Degree':
			new = {"x_is_ordered":True, "x_order":['No Degree','Bachelor', 'Master', 'PhD'], "order":"ascending","x_is_temporal":True} # in the original, x is given as ['Bachelor', 'Master', 'PhD','No Degree']

		k["models"][0]["x_order_info"] = new

	return data

def write_file(jsonvar,name):
	with open(name, 'w') as json_file:  
		json.dump(jsonvar, json_file)

new_data = add_info("original_data/train1_annotations.json")
write_file(new_data, "train1_annotations2.json")

new_data = add_info("original_data/val1_annotations.json")
write_file(new_data, "val1_annotations2.json")

new_data = add_info("original_data/val2_annotations.json")
write_file(new_data, "val2_annotations2.json")
