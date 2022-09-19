az synapse workspace update \
                --name "$bicep_parameters_workspaceName_value" \
                --resource-group "$bicep_parameters_resourceGroupName_value" \
                --account-name "$pipeline_Git_Configuration_0_git_username" \
                --collaboration-branch "$pipeline_Git_Configuration_0_branch" \
                --repository-type "$pipeline_Git_Configuration_0_git_provider" \
                --repository-name "$pipeline_Git_Configuration_0_repository_name" \
                --root-folder "$pipeline_Git_Configuration_0_repo_root_folder"
