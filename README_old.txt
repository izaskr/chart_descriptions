NLG for plot descriptions with content selection 
Masther thesis project by Iza Škrjanec
skrjanec.iza@gmail.com

The data for plot generation
The data comes as a json file, specifically as annotations.json
More about the format of the json file: https://github.com/rudy-kh/FigureQA/blob/dev/docs/annotations_format.md
There is a training set (7 plots) and two validation sets (each 2 plots). In total, there are 11 datasets for generating plots. So far, these are all bar charts of categorical data.

The plot descriptions
For 10 out of the 11 datasets/plots, plot descriptions were collected via Amazon Mechanical Turk. The descriptions were labeled manually (by Rudy Khalil and another student). These descriptions are .txt files.

Automatic analysis: mapping between the raw plot data and some labels
To assign some labels, we do not need an entire plot description in natural language. If we have the raw plot data, we can easily automatically analyze it to get some information, which usually appears in the plot description. For example, the category with the maximum value can be found from the raw numbers passed to the plot generator.
This was the motivation to write a script that performs this mapping.

USAGE
python analysis.py -labeled_file <name_of_description_file.txt>

Dependencies:
- raw plot data: 3 json files
- plot descriptions: 11 .txt files


Requirements:
- numpy

