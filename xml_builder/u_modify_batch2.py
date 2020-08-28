# script for modifying the annotation.json files
# add info about order of categories on x axis
# x_ordered_info: {x_is_ordered: bool, x_order: [] if False, else [A,B,C..]
# add this key:value to "models"[0]

import json

count = 0
# new_data_changed = 0

def add_info(filename):
	with open(filename) as f:
		data = json.load(f)
	
	# global new_data_changed
	if f_name == "new":
			print("\n\n", filename)
			

	for i,k in enumerate(data): # iterate over each plot, k are dictionaries with where the value is a list len 1
		# print(filename.split('/')[1])
		title = k["general_figure_info"]["title"]["text"]

		if f_name == "new":
			print(title)

		new = {"x_is_ordered":False, "x_order":[], "order": None, "x_is_temporal":False, "x_is_ratio":False, "y_order_as_x":[]}

		if title == 'Median Salary of Women Per Year':
			new = {"x_is_ordered":True, "x_order":['2000', '2005', '2010', '2015'], "order":"ascending", "x_is_temporal":True, "x_is_ratio":True, "y_order_as_x":[40, 45, 50, 55]}

		elif title == 'Median Salary Per Year For Software Engineers with Respect to their Degree':
			new = {"x_is_ordered":True, "x_order":['No Degree','Bachelor', 'Master', 'PhD'], "order":"ascending","x_is_temporal":False, "x_is_ratio":False, "y_order_as_x":[35, 45, 55, 65]} # in the original, x is given as ['Bachelor', 'Master', 'PhD','No Degree']

		# Akef 1
		elif title == 'Akef Inc. closing stock prices for the week' and filename.split('/')[1] == 'train1_annotations.json':
			new = {"x_is_ordered":True, "x_order":['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[92.75, 69.32, 61.72, 60.10, 31.30]}
			# new_data_changed += 1

		# Akef 2
		elif title == 'Akef Inc. closing stock prices for the week' and filename.split('/')[1] == 'val2_annotations.json':
			new = {"x_is_ordered":True, "x_order":['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[51.40, 62, 74.38, 21.9, 9.42]}
			# new_data_changed += 1

		# Fatal Injuries Pula 1
		elif title == 'Number of Fatal Injuries at the Pula Steel Factory' and filename.split('/')[1] == 'train1_annotations.json':
			new = {"x_is_ordered":True, "x_order":["2012", "2013", "2014", "2015", "2016"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":True, \
					"y_order_as_x":[27, 32, 26, 22, 25]}
			# new_data_changed += 1


		# Fatal Injuries Pula 2
		elif title == 'Number of Fatal Injuries at the Pula Steel Factory' and filename.split('/')[1] == 'val1_annotations.json':
			new = {"x_is_ordered":True, "x_order":["2012", "2013", "2014", "2015", "2016"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":True, \
					"y_order_as_x":[30, 25, 16, 15, 12]}
			# new_data_changed += 1


		# Minority 1
		elif title == 'Minority Representation in the Parliament of Lybia' and filename.split('/')[1] == 'train1_annotations.json':
			new = {"x_is_ordered":True, "x_order":["1990-1994", "1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2019"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[0.5, 2.1, 7.5, 10, 12.7, 14]}
			# new_data_changed += 1


		# Minority 2
		elif title == 'Minority Representation in the Parliament of Lybia' and filename.split('/')[1] == 'train1_annotations.json':
			new = {"x_is_ordered":True, "x_order":["1990-1994", "1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2019"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[13.7, 12.0, 8.6, 7.9, 6.3, 6.8]}
			# new_data_changed += 1


		# Social Media 1
		elif title == 'Average Time Spent On Social Media Daily in Maputo by Age Group' and filename.split('/')[1] == 'train1_annotations.json':
			new = {"x_is_ordered":True, "x_order":["15-24", "25-34", "35-44", "45-54", "55-64"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[125, 75, 74, 150, 110]}
			# new_data_changed += 1


		# Social Media 2
		elif title == 'Average Time Spent On Social Media Daily in Maputo by Age Group' and filename.split('/')[1] == 'val2_annotations.json':
			new = {"x_is_ordered":True, "x_order":["15-24", "25-34", "35-44", "45-54", "55-64"], \
				"order":"ascending", "x_is_temporal":True, "x_is_ratio":False, \
					"y_order_as_x":[181, 157, 124, 99, 73]}
			# new_data_changed += 1
        
		# if f_name == "new":
		# 	print(title, "\n", new["y_order_as_x"], "\n")

		k["models"][0]["x_order_info"] = new

	return data

def write_file(jsonvar,name):
	with open(name, 'w') as json_file:  
		# print(jsonvar)
		json.dump(jsonvar, json_file)


# TODO: when you unifiy old and new annotations, change folder paths
for f_name in ("new", "original"):
	new_data = add_info(f_name + "_data_json/train1_annotations.json")
	write_file(new_data, "u_updated json/" + f_name + "/train1_annotations3.json")

	new_data = add_info(f_name + "_data_json/val1_annotations.json")
	write_file(new_data, "u_updated json/" + f_name + "/val1_annotations3.json")

	new_data = add_info(f_name + "_data_json/val2_annotations.json")
	write_file(new_data, "u_updated json/" + f_name + "/val2_annotations3.json")

# print(new_data_changed)
