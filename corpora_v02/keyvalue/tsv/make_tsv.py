"""
Take the key-value train/val/test files and make them into tsv files that can be read by the AllenNLP seq2seq module

Replace values with keys in the target and make the source consist of keys only

"""


def replace_write_tsv(dpath, splt):
	""" dpath is the path to the data dir, splt is a str, either train, val or test """

	new = open(splt + ".txt", "w", encoding="utf8")
	backup_keyvalue = open(splt + "_keyvalue.txt", "w", encoding="utf8")

	# list of strings, each line one string
	source_lines = open(dpath + "src_" + splt + ".txt", "r").read().splitlines()
	target_lines =  open(dpath + "tg_" + splt + ".txt", "r").read().splitlines()

	for src, tg in list(zip(source_lines, target_lines)):
		#print(src, "\t", tg, "\n")
		# split the source to get a list of 'PLOTTITLE[causes of Obesity in Kiribati', ', YLEASTAPPROX[25', ', YUNIT[%', ...
		src_split = src.split("]")
		tg_copy = tg
		#print(type(tg)) # string
		keys_list = []
		for keyvalue in src_split:
			if keyvalue:
				b = keyvalue.index("[")
				key = keyvalue[:b]
				if key.startswith(","):
					key = key[1:].strip()
				keys_list.append(key)
				values = keyvalue[b+1:]
				#print(key, "\t" ,values)
				#print("OLD", tg_copy)
				tg_copy = tg_copy.replace(values, key)
				#print("NEW", tg_copy)
				#input("ENTER NEXT")
		#print(" ".join(keys_list), "\n" ,tg, "\n", tg_copy, "\n"*3)
		new.write(" ".join(keys_list) + "\t" + tg_copy + "\n")
		backup_keyvalue.write(src + "\n")


	
	


dir_path = "/home/iza/chart_descriptions/corpora_v02/keyvalue/"
files = ["test", "val", "train"]

for s in files:
	replace_write_tsv(dir_path, s)