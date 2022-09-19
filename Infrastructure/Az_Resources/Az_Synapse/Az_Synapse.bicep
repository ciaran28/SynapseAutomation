// ################################################################################################################################################################//
//                                                                       Parameters                                                                                //
// ################################################################################################################################################################//
param location string 
param environment string 
param shouldCreateContainers bool
param containerNames array
param storageConfig object
param workspaceName string 
param workspaceNamesa string 

//@secure()
//param adminPassword string


// ################################################################################################################################################################//
//                                                                         Variables                                                                               //
// ################################################################################################################################################################//
var roleDefinitionAzureEventHubsDataOwner = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
var roleDefinitionUser = guid('${resourceGroup().id}/ba92f5b4-2d11-453d-a403-e96b0029c9fe/')




// ################################################################################################################################################################//
//                                                                       Deploy Storage Account                                                                    //
// Deploys a Data Lake Storage Account // Containers // The ADLS will be used as the Default SA for the Synapse Workspace provissioned afterwards.
// Params { Takes Boolean to determine whether to deploy containers. Containers are defined in an array.
// ################################################################################################################################################################//
resource storageDeploy 'Microsoft.Storage/storageAccounts@2021-08-01' =  {    
  name: workspaceNamesa
    location: location
    kind: storageConfig.kind
    sku: {
      name: storageConfig.sku_name
    }
    properties: {
      allowBlobPublicAccess: storageConfig.allowBlobPublicAccess
      isHnsEnabled: storageConfig.isHnsEnabled
      accessTier: storageConfig.accessTier
    }
    // Nested Resource Deployment
    resource blobServices 'blobServices' = {
      name: 'default'
      resource containersCreate 'containers' = [for ContainerName in containerNames: if (shouldCreateContainers) {
        name: ContainerName
        properties: {
          publicAccess: 'Blob'
        }
      }]
    }
}



// ################################################################################################################################################################//
//                                                                       Synapse Workspace                                                                         //
// ################################################################################################################################################################//
  resource synapseWorkspace 'Microsoft.Synapse/workspaces@2021-06-01' = {
  name: workspaceName
  location: location // Policy on subscription. Probably in the wrong region
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    defaultDataLakeStorage: {
      resourceId: storageDeploy.id
      filesystem: 'bronze'
      accountUrl: 'https://${storageDeploy.name}.dfs.core.windows.net'
    }
    sqlAdministratorLogin: 'sqlserver'
    sqlAdministratorLoginPassword: 'CoMplexSec23!'
    publicNetworkAccess: 'Enabled'
    
    
  }
}


// ################################################################################################################################################################//
//                                                             Configure FireWall Settings                                                                         //
// End Ip Address etc, set to '0.0.0.0' equates to Azure-Internal IP Addresses
// Must Also Name The Resource 'AllowAllWindowsAzureIps' if setting all IP addresseses to 0.
// ################################################################################################################################################################//
resource AllowAzTraffic 'Microsoft.Synapse/workspaces/firewallRules@2021-06-01' = {
  name: 'AllowAllWindowsAzureIps'
  parent: synapseWorkspace
  properties: {
    endIpAddress: '0.0.0.0'
    startIpAddress: '0.0.0.0'
  }
}



// ################################################################################################################################################################//
//                                                              Firewall Rules - Allow All Traffic                                                                 //
// ################################################################################################################################################################//

//Default Firewall Rules - Allow All Traffic
resource AllowAllTraffic 'Microsoft.Synapse/workspaces/firewallRules@2021-06-01' = {
  name: 'AllowAllNetworks'
  parent: synapseWorkspace
  properties:{
    startIpAddress: '0.0.0.0'
    endIpAddress: '255.255.255.255'
  }
}



// ################################################################################################################################################################//
//                                                              RBAC Assignment for Managed Identity on Synapse WS 
//  Potential Error: When you delete resources, the RBAC is not always removed, creating strange errors. Search all Scopes for role assignments, and you can see 
//  RBACs attached to
//  'User' Unknown. Delete this, and retry
//  https://stackoverflow.com/questions/61637124/azure-devops-pipeline-error-tenant-id-application-id-principal-id-and-scope                                        //
// ################################################################################################################################################################//
resource rbacDataCont 'Microsoft.Authorization/roleAssignments@2020-10-01-preview' = {
  name: guid(synapseWorkspace.id, roleDefinitionAzureEventHubsDataOwner)
  // As Scope is not defined, the role will sccope to the file. As it is not specified it defaults to Resource Group.
  properties: {
    principalId: synapseWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleDefinitionAzureEventHubsDataOwner
  }
}


// ################################################################################################################################################################//
//                                                              CREATE BIG DATA POOL/CLUSTER                                                                       //
// Points of Note:                                                                                                                                                 //
//   If Spark Cluster Config Parameters are Invalid It Will Silently Not Deploy Without Errors (e.g. MinNodeCount: > 3.. if you select 2 it will fail without error//
// ################################################################################################################################################################//


resource bigDataCluster 'Microsoft.Synapse/workspaces/bigDataPools@2021-06-01' = {
  name: 'bigDataCluster'
  location: location
  tags: {
    tagName1: 'tagValue1'
    tagName2: 'tagValue2'
  }
  parent: synapseWorkspace
  properties: {
    autoPause: {
      delayInMinutes: 15
      enabled: true
    }
    autoScale: {
      enabled: true
      maxNodeCount: 10
      minNodeCount: 3
    }
    cacheSize: 100
    nodeSizeFamily: 'MemoryOptimized'
    nodeSize: 'Medium'
    sparkVersion: '3.1'
  }
}


// This Should Be In JSON Parameter File In Bash
resource rbacUser 'Microsoft.Authorization/roleAssignments@2020-10-01-preview' = {
  name: roleDefinitionUser
  properties: { 
    principalId: '3fb6e2d3-7734-43fc-be9e-af8671acf605' // This is you personal user User ID in ADD
    principalType: 'User'
    roleDefinitionId: roleDefinitionAzureEventHubsDataOwner // It Does Not Like This To Be A Hard Coded String - Strange
  }
}


// ################################################################################################################################################################//
//                                                              SET SYNAPSE MSI AS SQL ADMIN                                                                      //
// ################################################################################################################################################################//

  resource grantsqlControl 'Microsoft.Synapse/workspaces/managedIdentitySqlControlSettings@2021-06-01' = {
    name: 'default'
    parent: synapseWorkspace
    properties:{
      grantSqlControlToManagedIdentity:{
        desiredState: 'Enabled'
      }
    }
  }

