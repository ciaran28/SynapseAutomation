echo "Performing Parameters Tests... All Values Should Be None Empty"

echo "Synapse Workspace Name: $bicep_parameters_workspaceName_value"

echo "Synapse Workspace Resource Group: $bicep_parameters_resourceGroupName_value"

echo "Specify Whether ADO or Github: $pipeline_Git_Configuration_0_git_provider"

echo "Repo Account Name for Synapse Dev: $pipeline_Git_Configuration_0_git_username"

echo "Repo Name for Synapse Dev: $pipeline_Git_Configuration_0_repository_name"

echo "Root Folder To Use For Synapse Artifacts: $pipeline_Git_Configuration_0_repo_root_folder"
echo "Tests Complete...."


echo "Link Repo With Synapse Artifacts To Synapse Development Workpace...."
az synapse workspace update \
                --name "$bicep_parameters_workspaceName_value" \
                --resource-group "$bicep_parameters_resourceGroupName_value" \
                --account-name "$pipeline_Git_Configuration_0_git_username" \
                --collaboration-branch "$pipeline_Git_Configuration_0_branch" \
                --repository-type "$pipeline_Git_Configuration_0_git_provider" \
                --repository-name "$pipeline_Git_Configuration_0_repository_name" \
                --root-folder "$pipeline_Git_Configuration_0_repo_root_folder"
