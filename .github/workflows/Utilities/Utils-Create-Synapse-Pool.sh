
LIST_CLUSTERS=$(az synapse spark pool list \
                --resource-group "$bicep_parameters_resourceGroupName_value" \
                --workspace-name "$bicep_parameters_workspaceName_value")
echo "List Clusters: $LIST_CLUSTERS"

# Extract Existing Cluster Names
CLUSTER_NAMES=$( jq -r '[.[].name]' <<< "$LIST_CLUSTERS")
echo "Cluster Names: $CLUSTER_NAMES"

echo "Ingest JSON Environment File"
JSON=$( jq '.' .github/workflows/Pipeline_Param/$environment.json)
#echo "${JSON}" | jq


# The Az Synapse Pool Create Will OverWrite Pools Which Already Exist
echo "Configure All Synapse Spark Pools.... "
for row in $(echo "${JSON}" | jq -r '.Clusters[] | @base64'); do
    _jq() {
        echo ${row} | base64 --decode | jq -r ${1}
    }

    echo "Synapse Pool Does Not Exist: Create Synapse Pool... "
    
    az synapse spark pool create \
        --name "$(_jq '.cluster_name')" \
        --node-size "$(_jq '.node_size')" \
        --node-size-family "$(_jq '.node_size_family')" \
        --spark-version "$(_jq '.spark_version')" \
        --node-count "$(_jq '.node_count')" \
        --enable-auto-pause "$(_jq '.enable_auto_pause')" \
        --delay "$(_jq '.auto_pause_delay_in_minutes')" \
        --enable-auto-scale "$(_jq '.enable_auto_scale')" \
        --max-node-count "$(_jq '.autoscale_max_node_count')" \
        --min-node-count "$(_jq '.autoscale_min_node_count')" \
        --resource-group "$bicep_parameters_resourceGroupName_value" \
        --workspace-name "$bicep_parameters_workspaceName_value" \
        --no-wait

done

