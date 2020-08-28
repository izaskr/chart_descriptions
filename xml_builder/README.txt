All the dependencies for the u_write_xml.py file are currently packed into the requirements.txt file.
And can be installed with the command "pip install -r requirements.txt". 

The file initialize_dicts.py contains all the initialization dictionary for both the old and new datasets. 
This is used by the u_write_xml.py

When parsing the xml for the "entire" dataset, the json files (train1_annotations3.json, val1_annotations3.json, val2_annotations3.json) should be the combined json for both the old and new dataset. 
For reference this file can always be found in the folder "data/json_data" (files would need to be renamed). 
And the old json files are currently beiing stored in the folder "old_jsons"

In for the current implementation, all relevant description files are expected to be stored in the folder "corpora_v02/all descriptions"
