import json

fname = "label_frequencies.txt"

jfname = "lex_delex.json"

with open(jfname, "r") as jf:
    data = json.load(jf)

with open(fname, "r", encoding="utf-8") as f:
    for line in f:
        line = line.split()
        label = line[0][1:-1]
        if label not in data:

            if "group" in label and "y_mean" not in label:
                #print("group")
                m = {"Scnd":"SECOND","5th":"FIFTH","highest":"HIGHEST","least":"LEAST","3rd":"THIRD","4th":"FOURTH"}
                #m = {"Scnd":"SECOND","5th":"FIFTH", "highest":"HIGHEST", "least":"LEAST", "3rd":"THIRD","4th":"FOURTH"}                                                                                                   }
                sp = label.split("_") # [group, y, Scnd, least] or [group, Scnd, least]
                new_label = "GR"
                look = sp[1:]
                #print(sp, look)
                if "y" in label:
                    #print(label, look)
                    look = sp[2:]
                    new_label = "GRY"
                for entity in look:
                    #if entity == "y":
                        #print(label)
                    #print(entity, m[entity])
                    new_label = new_label + m[entity]
                print(label, new_label)
                print('"%s" : "%s"' % (label, new_label))
            if "group" not in label:
                print("\t",label)