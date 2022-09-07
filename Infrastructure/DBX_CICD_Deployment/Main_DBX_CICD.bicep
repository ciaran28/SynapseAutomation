targetScope = 'subscription'

param location string
param environment string
param storageConfig object
param containerNames array
param resourceGroupName string
param workspaceName string
param ShouldCreateContainers bool = true
param loganalyticswsname string 
param appInsightswsname string 
param storageAccountName string 





//var roleDefinitionAzureStorageBlobContributor = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')



// ################################################################################################################################################################//
//                                                                       Create Resource Group                                                                    
// ################################################################################################################################################################//
resource azResourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  dependsOn: []
  name: resourceGroupName
  // Location of the Resource Group Does Not Have To Match That of The Resouces Within. Metadate for all resources within groups can reside in 'uksouth' below
  location: location
}




// ################################################################################################################################################################//
//                                                                       Module for Creating Azure Databricks Workspace
// Outputs AzDatabricks Workspace ID, which is used when Assigning RBACs
// ################################################################################################################################################################//

module synapsewsDeploy '../Az_Resources/Az_Synapse/Az_Synapse.bicep' = {
  scope: resourceGroup('${azResourceGroup.name}')
  name: 'synapsewsdeploychd'
  dependsOn: [
    azResourceGroup
  ]
  params: {
    location: location
    environment: environment
    shouldCreateContainers: ShouldCreateContainers
    containerNames: containerNames
    storageConfig : storageConfig
  }
}

// ################################################################################################################################################################//
//                                                                  KEY VAULT - SELECT KV                                                                                //
// ################################################################################################################################################################//

module azKeyVault '../Az_Resources/Az_KeyVault/Az_KeyVault.bicep' = {
  dependsOn: [
    synapsewsDeploy
  ]
  scope: azResourceGroup
  name: 'azKeyVault'
  params: {
    environment: environment 
  }
}

// ################################################################################################################################################################//
//                                                                       Module for Create Azure Data Lake Storage
// RBAC is assigned -> azDatabricks given access to Storage 
// ################################################################################################################################################################//
module azDataLake '../Az_Resources/Az_DataLake/Az_DataLake.bicep' =  {
  dependsOn: [
    azResourceGroup
    synapsewsDeploy
  ]
  scope: resourceGroup(resourceGroupName)
  name: 'azDataLake' 
  params: {
    storageAccountName: storageAccountName
    storageConfig: storageConfig
    location: location
    containerNames: containerNames
    ShouldCreateContainers: ShouldCreateContainers
    
    // Arm Is Incredible Dumb and only takes outputs from one resource (The last Resource To Deploy). Therefore the parameters below are simply for outputting so we can grab them in a task in YAML 
    workspaceName: workspaceName
    resourceGroupName: resourceGroupName
    azKeyVaultName: azKeyVault.outputs.azKeyVaultName


  }
}

module logAnalytics '../Az_Resources/Az_AppInsights/Az_AppInsights.bicep' = {
  dependsOn: [
    azResourceGroup
    synapsewsDeploy
    azDataLake
  ]
  scope: resourceGroup(resourceGroupName)
  name: 'logAnalytics'
  params: {
    location: location
    logwsname: loganalyticswsname
    appinsightname: appInsightswsname
  }
}




