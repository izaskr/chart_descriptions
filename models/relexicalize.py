"""
Take the baselune seq2seq outputs of the test set and relexicalize them given the keyvalue pairs

"""

prediction_lines = open("predictions_2.txt").read().splitlines()
keyvalue_lines = open("/home/iza/chart_descriptions/corpora_v02/keyvalue/tsv/test_keyvalue.txt").read().splitlines()

relexicalized = open("relex_predictions_2.txt", "w", encoding="utf8")

for kv, prediction in list(zip(keyvalue_lines, prediction_lines)):
    #print(prediction)
    # collect the keys and their values
    # YLABEL[number of fatal injuries], T1AB[pula steel factory], XLABEL[year], XHIGHEST[2013],
    # YHIGHEST[32], XSECOND[2012], YSECOND[27], XTHIRD[2014],
    #   #YTHIRD[26], XFOURTH[2016], YLABEL[number of fatal injuries], YFOURTH[25], XLEAST[2015], YLEAST[22]
    splt = kv.split("]")
    keys, values = [], []
    for one_str in splt:
        #splt = one_str.split("]")
        #print(splt)
        for keyvalue in splt:
            #print(keyvalue)
            if keyvalue:
                if "[" not in keyvalue: print(keyvalue)
                t = keyvalue.index("[")
                key = keyvalue[:t]
                value = keyvalue[t+1:]
                if key.startswith(","):
                    #print(key)
                    key = key[1:].strip()
                keys.append(key)
                values.append(value)
    pred_tokens = prediction.split()
    prediction_copy = prediction
    for token in pred_tokens:
        if token in keys:
            j = keys.index(token)
            prediction_copy = prediction_copy.replace(token, values[j])
    #print(prediction, "*"*4 , prediction_copy)
    relexicalized.write(prediction_copy + "\n")
relexicalized.close()