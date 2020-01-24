"""
Delexicalize the corpus (in XML) and write it into a file to be processed by the NLG system

What to delexicalize?
Bar heights: exact and approximated
Bar names
x_axis_labels
Y axis unit
X axis unit
slope_x_value
slope_y_value
y_axis_inferred_value_mul_...
y_axis_inferred_value_add_...


"""



import argparse
import xml.etree.ElementTree as ET
from string import punctuation
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument("-xml", required=False, help="xml corpus with chart summaries and labels", default="corpora_v02/chart_summaries_b01.xml")
parser.add_argument("-out", required=False, help="name of output file", default="corpora_v02/b01_delex")
args = vars(parser.parse_args())

xml_file = args["xml"]
out_name = args["out"]

# lowercase
# for every topic a separate file
# collect all token IDs and label IDs
# for each token: 
# 	if its ID not among label IDs, write it lc in file in a single line
# 	if its token among label IDs, then check if from==to: if yes, check if token should be delex, write to file
#		if from != to:
#			if token should be delex: do nothing, keep the placeholder
#				once the to ID is reached, write the placeholder into file
#			if no delex is needed, collect the tokens until the to ID is reached, then write to file, joined with +

lex_delex = {"y_axis":"YLABEL", "x_axis_label_least_value":"XLEAST", "x_axis_label_highest_value":"XHIGHEST", "x_axis_label_Scnd_highest_value":"XSECOND", "y_axis_least_value_val":"YLEAST", "y_axis_Scnd_highest_val":"YSECOND", "y_magnitude": "YMAG", "x_axis_label_3rd_highest_value":"XTHIRD", "y_axis_highest_value_val": "YHIGHEST", "y_axis_inferred_label":"YUNIT", "x_axis":"XLABEL", "y_axis_3rd_highest_val": "YTHIRD", "y_axis_inferred_highest_value_approx": "YHIGHESTAPPROX", "x_axis_label_4th_highest_value":"XFOURTH", "y_axis_inferred_least_value_approx":"YLEASTAPPROX",
"y_axis_4th_highest_val":"YFOURTH", "x_axis_labels_count":"COUNT", "y_axis_inferred_3rd_highest_value_approx":"YTHIRDAPPROX",
"x_axis_range_start":"XSTART", "x_axis_range_end":"XEND", "x_axis_labels":"BARNAMES", "y_axis_inferred_Scnd_highest_value_approx":"YSECONDAPPROX","x_interval":"INTERVAL", 
"y_axis_inferred_value_mul_v1=highest_v2=least":"MULHIGHESTLEAST", "slope_x_value":"SLOPEX", "y_axis_inferred_value_mul_v1=highest_v2=Scnd":"MULHIGHESTSECOND",
"y_axis_inferred_value_add_v1=highest_v2=least":"ADDHIGHESTLEAST",
"slope_y_value":"SLOPEY", "y_axis_inferred_value_mul_v1=least_v2=highest":"MULLEASTHIGHEST", "y_axis_inferred_value_mul_v1=Scnd_v2=least":"MULSECONDLEAST", "y_axis_inferred_value_mul_v1=Scnd_v2=3rd":"MULSECONDTHIRD", "y_axis_inferred_value_mul_v1=3rd_v2=highest":"MULTHIRDHIGHEST", "y_axis_inferred_value_add_v1=highest_v2=Scnd":"ADDHIGHESTSECOND",
"y_axis_inferred_value_add_v1=highest_v2=3rd":"ADDHIGHESTTHIRD",
"y_axis_inferred_value_add_v1=Scnd_v2=highest":"ADDSECONDHIGHEST",
"y_axis_inferred_value_mul_v1=least_v2=Scnd":"MULLEASTSECOND",
"y_axis_inferred_value_mul_v1=least_v2=3rd":"MULLEASTTHIRD",
"y_axis_inferred_value_mul_v1=highest_v2=3rd":"MULHIGHESTTHIRD",
"y_axis_inferred_value_mul_v1=Scnd_v2=highest":"MULSECONDHIGHEST",
"y_axis_inferred_value_mul_v1=4th_v2=Scnd":"MULFOURTHSECOND","y_axis_inferred_value_mul_v1=3rd_v2=least":"MULTHIRDLEAST",
"y_axis_inferred_value_mul_v1=3rd_v2=Scnd":"MULTHIRDSECOND","y_axis_inferred_value_add_v1=least_v2=3rd":"ADDLEASTTHIRD",
"y_axis_inferred_value_add_v1=Scnd_v2=3rd":"ADDSECONDTHIRD",
"y_axis_inferred_value_add_v1=3rd_v2=least":"ADDTHIRDLEAST" }
 


# x_axis_labels_rest , y_axis_inferred_value_add? , y_axis_inferred_label? , other_operation

stays = {"topic_related_property","topic", "topic_related_object", "y_axis_highest_value", "y_axis_least_value", "order_Scnd", "y_axis_approx", "y_axis_trend", "separator", "y_axis_trend_up","slope_up", "interpretation", "y_x_comparison_rest", "final_label_left", "order_3rd", "y_axis_trend_down", "y_x_comparison", "x_axis_comparison", "x_axis_up", "every", "x_axis_trend", "order_rest", "order_4th", "x_axis_period", "x_axis_trend_down"}


# each story starts with '' and ends with <end_of_description>_* where * is the script name

def open_delex_write(corpus, out_corpus):
	tree = ET.parse(corpus)
	root = tree.getroot()

	topic_name = ""

	for topic in root:
		topic_name = topic.attrib["topic"] # will be the name of the file

		#f = open(topic_name, "w", encoding="utf-8")
		i= 0
		for story in topic:
			i +=1
			sentences = story[0][1]
			tokens = []
			for sent in sentences:
				#for t in sent:
					#print(t.attrib["content_fix"])
				get_tokens = lambda x: [(t.attrib["content_fix"],t.attrib["id"]) for t in x]
				tokens += get_tokens(sent)
			print(tokens)
			















if __name__ == "__main__":
	open_delex_write(xml_file, out_name)

