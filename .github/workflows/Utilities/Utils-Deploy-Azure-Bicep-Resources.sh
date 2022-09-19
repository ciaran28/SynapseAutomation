#!/bin/bash

# Ensure Carriage Return Is Set To LF For Linux Machines.. 

echo "environment variable: $pipeline_TemplateParamFilePath"
echo "environment variable: $pipeline_Location"
echo "environment variable: $pipeline_TemplateFilePath"
# Important to define unique deployment names as conflicts will occur
echo "Create Azure DBX Resource Environments...."

az deployment sub create \
    --location $pipeline_Location \
    --template-file $pipeline_TemplateFilePath \
    --parameters $pipeline_TemplateParamFilePath \
    --name "$environment"