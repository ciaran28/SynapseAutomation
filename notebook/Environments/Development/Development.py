#!/usr/bin/env python
# coding: utf-8

import json 

print("test")
# Data to be written 
dictionary_json = { 
    "SubscriptionId": "4f1bc772-7792-4285-99d9-3463b8d7f994",
    "Location": "uksouth", 
    "TemplateParamFilePath":"Infrastructure/DBX_CICD_Deployment/Bicep_Params/Development/Bicep.parameters.json",
    "TemplateFilePath":"Infrastructure/DBX_CICD_Deployment/Main_DBX_CICD.bicep",
    "AZURE_DATABRICKS_APP_ID": "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d",
    "MANAGEMENT_RESOURCE_ENDPOINT": "https://management.core.windows.net/",
    "Clusters": [
        {
            "cluster_name": "dbx-sp-cluster",
            "spark_version": "10.4.x-scala2.12",
            "node_type_id": "Standard_D3_v2",
            "spark_conf": {},
            "autotermination_minutes": 30,
            "runtime_engine": "STANDARD",
            "autoscale": {
                "min_workers": 2,
                "max_workers": 4
            }
        },
        {
            "cluster_name": "dbx-sp-cluster2",
            "spark_version": "10.4.x-scala2.12",
            "node_type_id": "Standard_D3_v2",
            "spark_conf": {},
            "autotermination_minutes": 30,
            "runtime_engine": "STANDARD",
            "autoscale": {
                "min_workers": 2,
                "max_workers": 4
            }
        }
    ],
} 

# Serializing json  
#json_object = json.dumps(dictionary_json, indent = 4) 
#print(json_object)

import os

cur_path = os.path.dirname(__file__)
new_file = os.path.join(cur_path, 'Development.json')
print(new_file)



with open(new_file, "w") as outfile:
    json.dump(dictionary_json, outfile, indent=4)




# 1. Feature Merge To Main
# 2. Yaml Triggers
# 3. Bash Script points to ipynb file, and converts to py, using nb convert. Development.py is overwritten (Development.py will be destroyed in time)
# 4. Bash Script points to Development.py and runs it - which updates Development.json

# This is Visible Inside Synapse
