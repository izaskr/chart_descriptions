descriptions_files = ["Money_spent_on_higher_education.txt", 
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
    "women_representation_in_different_sectors_2_iza.txt", "women_representation_in_different_uni_departments_2.txt"

   ]

#This corresponds to the order of description_files_order
topic_image_id = [
    ("gender_paygap", "01_01", "nominal"), ("salary_women", "02_01", "interval"), 
    ("evenings", "03_01", "nominal"), ("salary_se_degree", "04_01", "ordinal"), 
    ("money_he", "05_01", "nominal"), ("top_unis", "06_01", "nominal"), 
    ("obesity_cause", "07_01", "nominal"), ("study_prog", "08_01", "nominal"), 
    ("women_dept", "09_01", "nominal"), ("women_sect", "10_01", "nominal"), 
    
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
    "akef_inc_closing_stock_prices_2_iza.txt":("train1","Akef Inc. closing stock prices for the week"),
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
    "akef_inc_closing_stock_prices_1.txt":("val2","Akef Inc. closing stock prices for the week"), 
    "median_salary_of_women_2_iza.txt":("val2","Median Salary of Women in Najaf Per Year"), 
    }




# to rectify, added 1 & 2 to the title of "Minority Representation in the Parliament of Lybia" in the json
# review and correct y_order_as_x in Minority Representation in the Parliament of Lybia 2