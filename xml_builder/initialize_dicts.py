#TODO: change ids in old dataset

def get_dicts(version="both"):
    ##=======================================================================================================
    ##=======================================================================================================
    #                    USE THIS (UNCOMMENT) TO RUN ON ALL THE DATA
    ##=======================================================================================================
    ##=======================================================================================================


    descriptions_files = [
        "Money_spent_on_higher_education.txt", 
        "Number_of_top_Unis.txt",
        "gender_pay_gap.txt",
        "women_representation_in_different_departments.txt",
        "women_representation_in_different_sectors.txt",
        "what_causes_obesity.txt",
        "how_do_young_people_spend_their_evenings.txt",
        "what_do_students_choose_to_study.txt",
        "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
        "example_Median_salary_of_women.txt", 
            
            
        "average_time_spent_on_social_media_1.txt",
        "minority_representation_in_libya_parliament_2_iza.txt",
        "akef_inc_closing_stock_prices_1.txt",
        "gender_pay_gap_2.txt",
        "minority_representation_in_libya_parliament_1_iza.txt",
        "akef_inc_closing_stock_prices_2_iza.txt",
        "women_representation_in_different_uni_departments_2.txt",
        "what_students_study_at_lagos_uni_iza.txt",
        "women_representation_in_different_sectors_2_iza.txt",
        "fatal_injuries_at_pula_steel_factory_2_iza.txt",
        "how_young_people_spend_their_evenings_1.txt",
        "average_time_spent_on_social_media_2_iza.txt",
        "what_causes_obesity_2.txt",
        "median_salary_of_women_2_iza.txt",
        "fatal_injuries_at_pula_steel_factory_1.txt",
        "money_spent_on_HE_2_iza.txt"


        # batch3
        'new_cameras1.txt', 'new_cameras2_g3.txt', 'new_citySA1_g2.txt', 'new_citySA2_g3.txt',
        'new_glaciers1_g1.txt', 'new_glaciers2_g3.txt', 'new_minority1_g3.txt', 'new_minority2_g2.txt',
        'new_moneyHE1_g1.txt', 'new_paygap1_g2.txt', 'new_paygap2_g3.txt', 'new_paygap3_g1.txt',
        'new_quiz1_g2.txt', 'new_quiz2_g1.txt', 'new_socmedia1_g3.txt', 'new_socmedia2_g2.txt',
        'new_stock_price1_g2.txt', 'new_womendep1_g2.txt', 'new_womendep2_g1.txt', 
        'new_womendep3_g3.txt', 'new_womensec1_g1.txt'
    ]

    description_files_order = [
        "gender_pay_gap.txt", "example_Median_salary_of_women.txt",
        "how_do_young_people_spend_their_evenings.txt", "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
        "Money_spent_on_higher_education.txt", "Number_of_top_Unis.txt", 
        "what_causes_obesity.txt", "what_do_students_choose_to_study.txt", 
        "women_representation_in_different_departments.txt", "women_representation_in_different_sectors.txt",
    
        #new data 
        "akef_inc_closing_stock_prices_1.txt", "akef_inc_closing_stock_prices_2_iza.txt",
        "average_time_spent_on_social_media_1.txt", "average_time_spent_on_social_media_2_iza.txt",
        "fatal_injuries_at_pula_steel_factory_1.txt", "fatal_injuries_at_pula_steel_factory_2_iza.txt",
        "gender_pay_gap_2.txt", "how_young_people_spend_their_evenings_1.txt",
        "median_salary_of_women_2_iza.txt", "minority_representation_in_libya_parliament_1_iza.txt",
        "minority_representation_in_libya_parliament_2_iza.txt", "money_spent_on_HE_2_iza.txt",
        "what_causes_obesity_2.txt", "what_students_study_at_lagos_uni_iza.txt",
        "women_representation_in_different_sectors_2_iza.txt", "women_representation_in_different_uni_departments_2.txt",

     
        # batch3
        
        'new_stock_price1_g2.txt', 'new_minority1_g3.txt', 'new_minority2_g2.txt',
        'new_socmedia1_g3.txt', 'new_socmedia2_g2.txt', 'new_cameras1.txt', 
        'new_cameras2_g3.txt', 'new_citySA1_g2.txt', 'new_citySA2_g3.txt',
        'new_glaciers1_g1.txt', 'new_glaciers2_g3.txt', 'new_moneyHE1_g1.txt', 
        'new_paygap1_g2.txt', 'new_paygap2_g3.txt', 'new_paygap3_g1.txt',
        'new_quiz1_g2.txt', 'new_quiz2_g1.txt', 'new_womendep1_g2.txt', 
        'new_womendep2_g1.txt', 'new_womendep3_g3.txt', 'new_womensec1_g1.txt'
        
       ]

    #This corresponds to the order of description_files_order
    topic_image_id = [
        ("gender_paygap", "01_01", "nominal"), ("salary_women", "02_01", "interval"), 
        ("evenings", "03_01", "nominal"), ("salary_se_degree", "04_01", "ordinal"), 
        ("money_he", "05_01", "nominal"), ("top_unis", "06_01", "nominal"), 
        ("obesity_cause", "07_01", "nominal"), ("study_prog", "08_01", "nominal"), 
        ("women_dept", "09_01", "nominal"), ("women_sect", "10_01", "nominal"), 
        
        # batch2 data
        ("akef_stock_1", "11_01", "ordinal"), ("akef_stock_2", "11_02", "ordinal"), 
        ("time_on_SM_1", "12_01", "ordinal"), ("time_on_SM_2", "12_02", "ordinal"),
        ("pula_injuries_1", "13_01", "ordinal"),  ("pula_injuries_2", "13_02", "ordinal"), 
        ("gender_paygap_2", "01_02", "nominal"), ("zarqa_evenings", "03_02", "nominal"),
        ("Najaf_salary_women", "02_02", "interval"), ("minority_rep_1", "14_01", "ordinal"), 
        ("minority_rep_2", "14_02", "ordinal"), ("money_he_2", "05_02", "nominal"), 
        ("kiribati_obesity", "07_02", "nominal"), ("lagos_study_prog", "08_02", "nominal"), 
        ("benoni_women_sect", "10_02", "nominal"), ("narvik_women_dept", "09_02", "nominal"),


        # batch3 data  
        ("new_stock_price1_g2", "11_02c", "ordinal"), 
        ("new_minority1_g3", "14_01a", "ordinal"), ("new_minority2_g2", "14_01b", "ordinal"), 
        ("new_socmedia1_g3", "12_01a", "ordinal"), ("new_socmedia2_g2", "12_01b", "ordinal"),

        ("new_cameras1", "18_01a", "nominal"), ("new_cameras2_g3", "18_01b", "nominal"),
        ("new_citySA1_g2", "16_01a", "nominal"), ("new_citySA2_g3", "16_01b", "nominal"),
        ("new_glaciers1_g1", "17_01a", "nominal"), ("new_glaciers2_g3", "17_01b", "nominal"),
        ("new_moneyHE1_g1", "05_01c", "nominal"), ("new_paygap1_g2", "01_02b", "nominal"),
        ("new_paygap2_g3", "01_02a", "nominal"), ("new_paygap3_g1", "01_02c", "nominal"),
        ("new_quiz1_g2", "15_01b", "nominal"), ("new_quiz2_g1", "15_01a", "nominal"),
        ("new_womendep1_g2", "09_01a", "nominal"), ("new_womendep2_g1", "09_01b", "nominal"),
        ("new_womendep3_g3", "09_02c", "nominal"), ("new_womensec1_g1", "10_02c", "nominal")

        ]


    descriptions_files_json = {
        "Money_spent_on_higher_education.txt":("train1","Money Spent on Higher Education in Year 2010"), 
        "Number_of_top_Unis.txt":("train1","Number of Top 100 Universities in Each Continent"),
        "gender_pay_gap.txt":("train1","Gender Pay Gap"),
        "women_representation_in_different_departments.txt":("train1","Women Representation in Different University Departments"),
        "women_representation_in_different_sectors.txt":("train1","Women Representation in Different Sectors"),
        "what_causes_obesity.txt":("val1","What causes Obesity"),
        "how_do_young_people_spend_their_evenings.txt":("val1","How do Young People Spend their Evenings"),
        "what_do_students_choose_to_study.txt":("train1","What do Students choose to study?"),
        "median_salary_per_year_for_se_with_respect_to_their_degrees.txt":("val2","Median Salary Per Year For Software Engineers with Respect to their Degree"),
        "example_Median_salary_of_women.txt":("train1","Median Salary of Women Per Year"),
        
        #new data

        #train1
        "akef_inc_closing_stock_prices_2_iza.txt":("val2","Akef Inc. closing stock prices for the week"),
        "average_time_spent_on_social_media_1.txt":("train1","Average Time Spent On Social Media Daily in Maputo by Age Group"), 
        "fatal_injuries_at_pula_steel_factory_1.txt":("train1","Number of Fatal Injuries at the Pula Steel Factory"), 
        "gender_pay_gap_2.txt":("train1","Gender Pay Gap, 2017"), 
        "how_young_people_spend_their_evenings_1.txt":("train1","How Young People in Zarqa Spend their Evenings"),
        "minority_representation_in_libya_parliament_1_iza.txt":("train1","Minority Representation in the Parliament of Lybia 1"),
        "minority_representation_in_libya_parliament_2_iza.txt":("train1","Minority Representation in the Parliament of Lybia 2"), 
        "what_causes_obesity_2.txt":("train1","What causes Obesity in Kiribati"),     
        "women_representation_in_different_sectors_2_iza.txt":("train1","Women Representation in Different Sectors in Benoni"), 
        "women_representation_in_different_uni_departments_2.txt":("train1","Women Representation in Different University Departments in Narvik"),

        #val1
        "what_students_study_at_lagos_uni_iza.txt":("val1","What Students at Lagos State University choose to study"),
        "money_spent_on_HE_2_iza.txt":("val1","Money Spent on Higher Education in the Year 2010"),
        "fatal_injuries_at_pula_steel_factory_2_iza.txt":("val1","Number of Fatal Injuries at the Pula Steel Factory"),
        
        #val2
        "average_time_spent_on_social_media_2_iza.txt":("val2","Average Time Spent On Social Media Daily in Maputo by Age Group"),
        "akef_inc_closing_stock_prices_1.txt":("train1","Akef Inc. closing stock prices for the week"), 
        "median_salary_of_women_2_iza.txt":("val2","Median Salary of Women in Najaf Per Year"), 



        # batch3 
        "new_cameras1.txt":("batch3",  "The most expensive digital cameras by average price"),
        "new_cameras2_g3.txt":("batch3",  "The most affordable digital cameras by average price"),
        "new_citySA1_g2.txt":("batch3",  "The warmest cities in South America"),
        "new_citySA2_g3.txt":("batch3",  "The coldest cities in South America"),
        "new_glaciers1_g1.txt":("batch3",  "The most explored glaciers by the number of expeditions"),
        "new_glaciers2_g3.txt":("batch3",  "The least explored glaciers by the number of expeditions"),
        "new_minority1_g3.txt":("batch3",  "Diversity in the Libyan parliament: percentage of minority representation"),
        "new_minority2_g2.txt":("batch3",  "Lack of diversity in the Libyan parliament: \npercentage of minority representation"),
        "new_moneyHE1_g1.txt":("batch3",  "Money spent on higher education in Asia in 2010"),
        "new_paygap1_g2.txt":("batch3",  "The most equal countries given the gender pay gap"),
        "new_paygap2_g3.txt":("batch3",  "The least equal countries given the gender pay gap"),
        "new_paygap3_g1.txt":("batch3",  "Gender pay gap in Germany in 2017"),
        "new_quiz1_g2.txt":("batch3",  "The most successful teams in quiz competition"),
        "new_quiz2_g1.txt":("batch3",  "The least successful teams in quiz competition"),
        "new_socmedia1_g3.txt":("batch3",  "The most frequent daily users of social media"),
        "new_socmedia2_g2.txt":("batch3",  "The least frequent daily users of social media"),
        "new_stock_price1_g2.txt":("batch3",  "Closing stock prices for Akef Inc. on Wednesday"),
        "new_womendep1_g2.txt":("batch3",  "The most inclusive university departments given representation of women"),
        "new_womendep2_g1.txt":("batch3",  "The least inclusive university departments given representation of women"),
        "new_womendep3_g3.txt":("batch3",  "Women representation in the literature department"),
        "new_womensec1_g1.txt":("batch3",  "Women representation in law firms"),
        }


    if version == "both":
        return descriptions_files, description_files_order, topic_image_id, descriptions_files_json


    # to rectify, added 1 & 2 to the title of "Minority Representation in the Parliament of Lybia" in the json
    # review and correct y_order_as_x in Minority Representation in the Parliament of Lybia 2














    ##=======================================================================================================
    ##=======================================================================================================
    #                     TO RUN ON ONLY THE OLD DATA
    ##=======================================================================================================
    ##=======================================================================================================




    descriptions_files = ["Money_spent_on_higher_education.txt", 
             "Number_of_top_Unis.txt",
             "gender_pay_gap.txt",
             "women_representation_in_different_departments.txt",
             "women_representation_in_different_sectors.txt",
             "what_causes_obesity.txt",
             "how_do_young_people_spend_their_evenings.txt",
             "what_do_students_choose_to_study.txt",
             "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
             "example_Median_salary_of_women.txt"]

    descriptions_files = ["Money_spent_on_higher_education.txt", 
             "Number_of_top_Unis.txt",
             "gender_pay_gap.txt",
             "women_representation_in_different_departments.txt",
             "women_representation_in_different_sectors.txt",
             "what_causes_obesity.txt",
             "how_do_young_people_spend_their_evenings.txt",
             "what_do_students_choose_to_study.txt",
             "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
             "example_Median_salary_of_women.txt"]

    description_files_order = ["gender_pay_gap.txt", "example_Median_salary_of_women.txt",
       "how_do_young_people_spend_their_evenings.txt", "median_salary_per_year_for_se_with_respect_to_their_degrees.txt",
       "Money_spent_on_higher_education.txt", "Number_of_top_Unis.txt", "what_causes_obesity.txt", 
       "what_do_students_choose_to_study.txt", 
       "women_representation_in_different_departments.txt", "women_representation_in_different_sectors.txt"]

    #TODO:change
    topic_image_id = [("gender_paygap", "01", "nominal"), ("salary_women", "02", "interval"), ("evenings", "03", "nominal"), ("salary_se_degree", "04", "ordinal"), ("money_he", "05", "nominal"), ("top_unis", "06", "nominal"), ("obesity_cause", "07", "nominal"), ("study_prog", "08", "nominal"), ("women_dept", "09", "nominal"), ("women_sect", "10", "nominal") ]
    # topic_image_id = [
    #     ("gender_paygap", "01_01", "nominal"), ("salary_women", "02_01", "interval"), 
    #     ("evenings", "03_01", "nominal"), ("salary_se_degree", "04_01", "ordinal"), 
    #     ("money_he", "05_01", "nominal"), ("top_unis", "06_01", "nominal"), 
    #     ("obesity_cause", "07_01", "nominal"), ("study_prog", "08_01", "nominal"), 
    #     ("women_dept", "09_01", "nominal"), ("women_sect", "10_01", "nominal")
    # ]

    # descriptions_files_json is a dict, where the keys are files names (as given by Rudy) with descriptions, and the values are tuples - its first element is the name of the json file with raw plot data, its second element is the title of the plot (it's unique for each plot)
    descriptions_files_json = {"Money_spent_on_higher_education.txt":("train1","Money Spent on Higher Education in Year 2010"), 
             "Number_of_top_Unis.txt":("train1","Number of Top 100 Universities in Each Continent"),
             "gender_pay_gap.txt":("train1","Gender Pay Gap"),
             "women_representation_in_different_departments.txt":("train1","Women Representation in Different University Departments"),
             "women_representation_in_different_sectors.txt":("train1","Women Representation in Different Sectors"),
             "what_causes_obesity.txt":("val1","What causes Obesity"),
             "how_do_young_people_spend_their_evenings.txt":("val1","How do Young People Spend their Evenings"),
             "what_do_students_choose_to_study.txt":("train1","What do Students choose to study?"),
             "median_salary_per_year_for_se_with_respect_to_their_degrees.txt":("val2","Median Salary Per Year For Software Engineers with Respect to their Degree"),
             "example_Median_salary_of_women.txt":("train1","Median Salary of Women Per Year")}


    if version == "old":
        return descriptions_files, description_files_order, topic_image_id, descriptions_files_json



    ##=======================================================================================================
    ##=======================================================================================================
    #                     TO RUN ON ONLY THE NEW DATA
    ##=======================================================================================================
    ##=======================================================================================================





    descriptions_files = [
            
            "average_time_spent_on_social_media_1.txt",
            "minority_representation_in_libya_parliament_2_iza.txt",
            "akef_inc_closing_stock_prices_1.txt",
            "gender_pay_gap_2.txt",
            "minority_representation_in_libya_parliament_1_iza.txt",
            "akef_inc_closing_stock_prices_2_iza.txt",
            "women_representation_in_different_uni_departments_2.txt",
            "what_students_study_at_lagos_uni_iza.txt",
            "women_representation_in_different_sectors_2_iza.txt",
            "fatal_injuries_at_pula_steel_factory_2_iza.txt",
            "how_young_people_spend_their_evenings_1.txt",
            "average_time_spent_on_social_media_2_iza.txt",
            "what_causes_obesity_2.txt",
            "median_salary_of_women_2_iza.txt",
            "fatal_injuries_at_pula_steel_factory_1.txt",
            "money_spent_on_HE_2_iza.txt"
            ]

    description_files_order = [   
        #new data 
        "akef_inc_closing_stock_prices_1.txt", "akef_inc_closing_stock_prices_2_iza.txt",
        "average_time_spent_on_social_media_1.txt", "average_time_spent_on_social_media_2_iza.txt",
        "fatal_injuries_at_pula_steel_factory_1.txt", "fatal_injuries_at_pula_steel_factory_2_iza.txt",
        "gender_pay_gap_2.txt", "how_young_people_spend_their_evenings_1.txt",
        "median_salary_of_women_2_iza.txt", "minority_representation_in_libya_parliament_1_iza.txt",
        "minority_representation_in_libya_parliament_2_iza.txt", "money_spent_on_HE_2_iza.txt",
        "what_causes_obesity_2.txt", "what_students_study_at_lagos_uni_iza.txt",
        "women_representation_in_different_sectors_2_iza.txt", "women_representation_in_different_uni_departments_2.txt"

    ]

    #This corresponds to the order of description_files_order
    topic_image_id = [
        
        # new data
        ("akef_stock_1", "11_01", "ordinal"), ("akef_stock_2", "11_02", "ordinal"), 
        ("time_on_SM_1", "12_01", "ordinal"), ("time_on_SM_2", "12_02", "ordinal"),
        ("pula_injuries_1", "13_01", "ordinal"),  ("pula_injuries_2", "13_02", "ordinal"), 
        ("gender_paygap_2", "01_02", "nominal"), ("zarqa_evenings", "03_02", "nominal"),
        ("Najaf_salary_women", "02_02", "interval"), ("minority_rep_1", "14_01", "ordinal"), 
        ("minority_rep_2", "14_02", "ordinal"), ("money_he_2", "05_02", "nominal"), 
        ("kiribati_obesity", "07_02", "nominal"), ("lagos_study_prog", "08_02", "nominal"), 
        ("benoni_women_sect", "10_02", "nominal"), ("narvik_women_dept", "09_02", "nominal")

        ]


    descriptions_files_json = {    
        #new data

        #train1
        "akef_inc_closing_stock_prices_2_iza.txt":("val2","Akef Inc. closing stock prices for the week"),
        "average_time_spent_on_social_media_1.txt":("train1","Average Time Spent On Social Media Daily in Maputo by Age Group"), 
        "fatal_injuries_at_pula_steel_factory_1.txt":("train1","Number of Fatal Injuries at the Pula Steel Factory"), 
        "gender_pay_gap_2.txt":("train1","Gender Pay Gap, 2017"), 
        "how_young_people_spend_their_evenings_1.txt":("train1","How Young People in Zarqa Spend their Evenings"),
        "minority_representation_in_libya_parliament_1_iza.txt":("train1","Minority Representation in the Parliament of Lybia 1"),
        "minority_representation_in_libya_parliament_2_iza.txt":("train1","Minority Representation in the Parliament of Lybia 2"), 
        "what_causes_obesity_2.txt":("train1","What causes Obesity in Kiribati"),     
        "women_representation_in_different_sectors_2_iza.txt":("train1","Women Representation in Different Sectors in Benoni"), 
        "women_representation_in_different_uni_departments_2.txt":("train1","Women Representation in Different University Departments in Narvik"),

        #val1
        "what_students_study_at_lagos_uni_iza.txt":("val1","What Students at Lagos State University choose to study"),
        "money_spent_on_HE_2_iza.txt":("val1","Money Spent on Higher Education in the Year 2010"),
        "fatal_injuries_at_pula_steel_factory_2_iza.txt":("val1","Number of Fatal Injuries at the Pula Steel Factory"),
        
        #val2
        "average_time_spent_on_social_media_2_iza.txt":("val2","Average Time Spent On Social Media Daily in Maputo by Age Group"),
        "akef_inc_closing_stock_prices_1.txt":("train1","Akef Inc. closing stock prices for the week"), 
        "median_salary_of_women_2_iza.txt":("val2","Median Salary of Women in Najaf Per Year"), 
        }


    if version == "new":
        return descriptions_files, description_files_order, topic_image_id, descriptions_files_json





# init_dict_version = "old"

# descriptions_files, description_files_order, topic_image_id, descriptions_files_json = get_dicts(init_dict_version)


# print(len(descriptions_files))
# print(len(description_files_order))
# print(len(topic_image_id))
# print(len(descriptions_files_json))

