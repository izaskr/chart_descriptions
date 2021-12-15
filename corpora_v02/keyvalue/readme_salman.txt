The file splits_combinations_ids.json doesn't contain anything new. It's a dictionary with summary IDs given the different data splits and combinations.
We have two combinations:
	- "mixed": each topic occurs in every data split if possible, so the topics in the test set are not unseen
	- "exclusive": there are 2 topics that occur only at testing, but not training
	
We have 3 splits: train, val, test.
The dictionary has the following structure:
	{"mixed": {
			"train": [summaryID, summaryID ...], "val": [summaryID, summaryID ...], "test": [summaryID, summaryID ...]
			}
	}

The order of summary IDs corresponds to th onces in the txt files.

The file chartID2additional_info_salman.json contains the additional information I talked about. Specifically, it contains information about the relations between bars: additive, multiplicative, height appoximations, group references. Let me give made-up examples for each relation:
	additive: UK pays 50 million more than Australia. --> 50 expresses an additive relation
	multiplicative: Teenagers spend 3 times the time of their parents on their phones. --> 3 expresses multiplication
	heigh approximation: A Leica camera costs about 70 euros. --> the actual height is (let's say) 72.5, so here the height is approximated
	group references: Australia and New Zealand each have about 90 mio. --> there 90 refers to the heights of both of these bars, calculated as an average of both heights

This file is a dicitonary, where each chart ID is a, its value is a dicionary. In this dictionary, the key is the relation name, the value is its value. 
	
This information is stored in a way that worked for me, but it's a bit annoying to read and especially process. 
Keys starting as ADD* are for additive relations,
		MUL* for multiplications
		GRY* for heights of group references
		GR[not Y]* for bars in the group references
Keys ending in *APPROX are for height approximations.

Why this isn't the best way to store information: if you look more closely, you will see that the key names also explain which bars they refer to, for example:
	"YLEASTAPPROX" - it's a height approximation for the lowest bar
	"MULSECONDTHIRD" - it's the multiplication relation between the second highest and the third highest bar
	"GRHIGHESTSECONDTHIRD" - the names of the following bars: highest, second and third highest
	"GRYHIGHESTSECONDTHIRD" - heights of the highest, second and third highest brs

In terms of how to add this information to the format which is used my your NLG model, I think this above descried format has to be adapted. Do you have any suggestions here?


		
	

