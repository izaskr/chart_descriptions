import spacy
import os
import re
from nltk import word_tokenize

nlp = spacy.load("en_core_web_sm")

"""
Change the tokenization in the source files of all formats

formats: cpy, set, exh

chart_descriptions/corpora_v02/keyvalue/complete_fixed/

"""

format_dirs = ["cpy", "set", "exh"]


def tokenize_source(format_type):
	original_dir = "complete_fixed/" + format_type
	new_dir = "complete_fixed_tok/" + format_type

	# source files
	sf = [f for f in os.listdir(original_dir) if "src" in f]
	print("ORIGINAL \t NLTK \t SPACY")
	for src_file in sf:

		# create a new file
		new_file = open(new_dir + "/" + src_file, "w")

		# open original (older version with faulty tokenization)
		with open(original_dir + "/" + src_file, "r") as f:
			for line in f:

				# split the line according to ], to get key-value pairs
				# in the shape of key[valuetoken1 valuetoken2
				# the last key-value pair is key[value] - it has the closing bracket
				line = line.split("],")

				to_write = []

				for kv in line:
					kv_clean = kv.strip()
					if kv_clean[-1] == "]":
						kv_clean = kv_clean[:-1]
					bindex = kv_clean.index("[")
					key, value = kv_clean[:bindex], kv_clean[bindex+1:]

					# nltk tokenization
					nltk_tokens = word_tokenize(value)
					new_nltk_tokens = []

					for nt in nltk_tokens:

						nt = nt.replace(" \\n", " \\\\n")


						if "'" in nt: # cases: 's or've and the like to make ' s
							new_nt = nt.replace("'", " ' ")
							nt = re.sub(" +", " ", new_nt)

						#if nt == "'s":
							#new_nltk_tokens.append("' s") # split 's into 2 tokens
							#continue

						if "-" in nt and len(nt) >= 2: # case 1995-1998, but not -
							# replace each "-" with " - "
							new_nt = nt.replace("-", " - ")
							# remove double/multiple white spaces
							new_nt = re.sub(" +", " ", new_nt)
							new_nltk_tokens.append(new_nt)
							continue	
	
						else:
							new_nltk_tokens.append(nt)

					value_new = " ".join(new_nltk_tokens)
					value_new = re.sub(" +", " ", value_new) # remove multiple white spaces

					# spacy tokenization
					#doc = nlp(value)
					#value_spacy = " ".join([t.text for t in doc])
					
					# CHECK PROBLEMATIC CASES
					#if "'" in value_new or "-" in value_new:
						#print(value, "\t", value_new, "\t")
						# fundamental difference: spacy splits NUMBER-NUMBER by whitespace
						# 2010-2014 	 2010-2014 	 2010 - 2014
					
					to_write.append(key+"["+value_new+"]")
				
				to_write = ", ".join(to_write)
				print(to_write)
				new_file.write(to_write + "\n")
		new_file.close()



#tokenize_source("exh")

def tokenize_target(format_type):
	original_dir = "complete_fixed/" + format_type
	new_dir = "complete_fixed_tok/" + format_type

	# source files
	sf = [f for f in os.listdir(original_dir) if "tgt" in f]
	#print("ORIGINAL \t NLTK \t SPACY")
	for src_file in sf:

		# create a new file
		new_file = open(new_dir + "/" + src_file, "w")
		
		# open original (older version with faulty tokenization)
		with open(original_dir + "/" + src_file, "r") as f:
			for line in f:

				# tokenize the line with nltk
				tokens = word_tokenize(line)
				tokens = " ".join(tokens)

				# replace - with space-space
				tokens = tokens.replace("'", " ' ")

				# replace ' with space'space
				tokens = tokens.replace("-", " - ")

				# replace single \n with \\n
				tokens = tokens.replace(" \\n", " \\\\n")
				#if "\\n" in tokens and "Libyan parliament" in tokens:
				#if " \\n" in tokens:
					#print(tokens)

				# remove multiple whitespaces
				tokens = re.sub(" +", " ", tokens)

				#print(tokens)
				# write into file
				new_file.write(tokens + "\n")

		# close file
		new_file.close()


for form in format_dirs:
	tokenize_source(form)
	tokenize_target(form)






















