
// I don't think Application ID is required. Test by taking it out. If not needed then all we need is tenant ID and Object. Tenant is the same
// We could then get the ID for the Databricks Workspace, and also configure it to retrieve access from key vault. It would only need 
// Get access. Therefore we can use the PAT Token in our notebooks. 

// I think it's better to abandon the accessPolicies, and use RBAC assignments instead, on account of us not being 
// able to get the Object ID of Dbx ws in BICEP. Gonna have to do it in YAML anyway -->
// I was having issues assigning serviceConenct1 RBAC in YAML --> could keep it here ....

param environment string


var keyVaultName = 'keyvault-${environment}-${substring(uniqueString(resourceGroup().id), 0, 4)}'


resource azKeyVault 'Microsoft.KeyVault/vaults@2021-10-01' = {
  name: keyVaultName
  location: 'uksouth'
  properties: {
    sku: {
      family: 'A'
      name: 'premium'
    }
    tenantId: '72f988bf-86f1-41af-91ab-2d7cd011db47' 
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
      ipRules: []
      virtualNetworkRules: []
    }
    enableRbacAuthorization: true // if this is false then you cannot use RBAC assignments, on acl (below). If true acl (below) is ignored
    enableSoftDelete: false
    enabledForTemplateDeployment: true
    accessPolicies: [
        {
          //applicationId: 'ce79c2ef-170d-4f1c-a706-7814efb94898' // Application ID of databricks SPN
          permissions: {
            // Give it the ability to set secrets // we can then get rid of the Key Vault Admin permission set in the main pipeline
              // Can we do this for the main spn , the equivalent of serviceConnect1
            secrets: [
            'set'
            'list'
            'get'
          ]
          }
          tenantId: '72f988bf-86f1-41af-91ab-2d7cd011db47' 
          objectId: 'ab926dd1-657d-4bb2-9987-c7857046d0dd'
        }
        
        {
        //applicationId: '5d57ca95-aca6-453d-9110-97f687d9dff6' // Application ID of serviceConnect1
        permissions: {
          secrets: [
            'set'
            'list'
            'get'
          ]
        }
        tenantId: '72f988bf-86f1-41af-91ab-2d7cd011db47' 
        objectId: '47527038-bf92-4422-8632-961c5851c21b'
      }
    ]
  }
  
}


output azKeyVaultName string = azKeyVault.name
