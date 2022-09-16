az synapse workspace update \
                --name "synapsewsdeploychd" \
                --resource-group "synapse-dev-rg" \
                --account-name "ciaran28" \
                --collaboration-branch "main" \
                --repository-type "GitHub" \
                --repository-name "SynapseArtifacts" \
                --root-folder "/synapseresources" 
