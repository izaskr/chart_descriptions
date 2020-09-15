import json

def all_chartID2topicName(f="/home/iza/chartID2topic.txt"):
    """ return a dict; chartID as key, chart-topic name as value """
    d = {}
    with open(f, "r") as pfile:
        for line in pfile:
            line = line.strip().split(" : ")
            d[line[0]] = line[1]
    return d


def get_info_from_map(mp_file="/home/iza/chart_descriptions/data_batch3/README.txt"):
    # BATCH 3
    # open the map file, return a list of tuples (filename, index_in_annotations_json)
    maps = []
    with open(mp_file, "r") as f:
        for line in f:
            if not line.startswith("#") and len(line.split()) > 1:
                line = line.split(" : ")
                # print(line)
                maps.append((line[0], int(line[1])))
    return maps

# TODO fix below
b2 = {"akef_inc_closing_stock_prices_1.txt":("train1", 9, "akef_stock_1"),
"average_time_spent_on_social_media_1.txt":("train1", 10, "time_on_SM_1"),
"fatal_injuries_at_pula_steel_factory_1.txt":("train1", 11, "pula_injuries_1"),
"gender_pay_gap_2.txt":("train1", 16, "gender_paygap_2"),
"how_young_people_spend_their_evenings_1.txt":("train1", 13, "zarqa_evenings"),
"minority_representation_in_libya_parliament_1.txt":("train1", 12, "minority_rep_1"),
"what_causes_obesity_2.txt":("train1", 15, "kiribati_obesity"),
"women_representation_in_different_uni_departments_2.txt":("train1", 7, "narvik_women_dept"),
"akef_inc_closing_stock_prices_2.txt":("val2", 4, "akef_stock_2"),
"average_time_spent_on_social_media_2.txt":("val2", 3, "time_on_SM_2"),
"fatal_injuries_at_pula_steel_factory_2.txt":("val1", 3, "pula_injuries_2"),
"median_salary_of_women_2.txt":("val2", 2, "Najaf_salary_women"),
"minority_representation_in_libya_parliament_2.txt":("train1", 14, "minority_rep_2"),
"money_spent_on_HE_2.txt":("val1", 4, "money_he_2"),
"what_students_study_at_lagos_uni.txt":("val1", 2, "lagos_study_prog"),
"women_representation_in_different_sectors_2.txt":("train1", 8, "benoni_women_sect")}

b1 = {("val1", 0, "03_01", "evenings"),
("val1", 1, "07_01", "obesity_cause"),
("val2", 0, None, "what_young_people_listen_to"),
("val2", 1, "04_01", "salary_se_degree"),
("train1", 0, "02_01", "salary_women"),
("train1", 1, "01_01", "gender_paygap"),
("train1", 2, "05_01", "money_he"),
("train1", 3, "06_01", "top_unis"),
("train1", 4, "09_01", "women_dept"),
("train1", 5, "10_01", "women_sect"),
("train1", 6, "08_01", "study_prog")}

b3_maps = get_info_from_map()
#print(b3_maps)
chartID2name = all_chartID2topicName()
name2chartID = {v:k for k,v in chartID2name.items()}
assert len(chartID2name) == len(name2chartID)


def get_b3(jfile="/home/iza/chart_descriptions/data/json_data/batch3.json"):
    tmp = {}
    with open(jfile, "r") as jf:
        data3 = json.load(jf)

    for (fname, i) in b3_maps:
        tname = fname[:-8]
        chID = name2chartID[tname]
        tmp[chID] = data3[i]

    return tmp



def get_b1b2():
    """ return a dictionary; chartID as key, the value is a dict (from FigureQA) with plotting info """
    dir_path = "/home/iza/chart_descriptions/data/json_data/"
    json_names = ["train1.json", "val1.json", "val2.json"]
    tmp = {}
    b1b2 = {}
    for jname in json_names:
        with open(dir_path+jname, "r") as jf:
            data = json.load(jf) # a list of dict
            tmp[jname[:-5]] = data # train1, val1, val2 as keys


    for (split, i, chrtid, topicName) in b1:
        #print(topicName, tmp[split][i]["general_figure_info"]["title"]["text"], chrtid)
        if chrtid: # the not included ones have None
            b1b2[chrtid] = tmp[split][i]

    #print("..... B2")
    for fname, (split, i, topicName) in b2.items():
        # get the correct plotting info
        #print(topicName, tmp[split][i])
        #print(topicName, tmp[split][i]["general_figure_info"]["title"]["text"], name2chartID[topicName])
        current_chartID = name2chartID[topicName]
        if current_chartID:
            b1b2[current_chartID] = tmp[split][i]

    #import pdb; pdb.set_trace()
    return b1b2


f12=get_b1b2()
f3=get_b3()
#import pdb; pdb.set_trace()
#print(chartID2name)
chartID2plotInfo = {**f12, **f3}

with open("/home/iza/chart_descriptions/data/json_data/chartID2plotInfo.json", "w") as jf:
    json.dump(chartID2plotInfo, jf)

print("wrote a new file chartID2plotInfo.json")
#import pdb; pdb.set_trace()
