Train/val/test split for bar chart summaries

Sizes:
	* train: 634
	* val: 213
	* test: 216

Data preparation given the folder:
	- cpy : copy the entities (key-value) from target into source with repetitions
		a - target is lexicalized
		b - target is delexicalized
	- set : copy the entities (key-value) from target into source without repetitions
		c - target is lexicalized
		d - target is delexicalized
	- exh : source consists of an exhaustive set of possible entities (key-value) given the chart
		e - target is lexicalized
		f - target is delexicalized


In comparison to the previously sent data, this one has:
- more instances
- instances in non-neutral conditions
- options for a (de)lexicalized target
