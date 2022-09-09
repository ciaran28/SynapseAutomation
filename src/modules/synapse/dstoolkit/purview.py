# from azureml.pipeline.core import Pipeline
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Union
from pyapacheatlas.auth import ServicePrincipalAuthentication
from pyapacheatlas.core import AtlasProcess
from pyapacheatlas.core.util import AtlasUnInit
from pyapacheatlas.core.client import PurviewClient
from pyapacheatlas.core.entity import AtlasEntity
from pyapacheatlas.core.typedef import EntityTypeDef
from pyapacheatlas.core.typedef import RelationshipTypeDef
from pyapacheatlas.core.util import GuidTracker


class Entities(str, Enum):
    NOTEBOOK = "notebook"
    DATASTEP = "datastep"
    REL_NOETBOOK_TO_DATASTEP = "notebook_to_datastep"
    REL_DATASTEP_TO_DATASTEP = "datastep_to_datastep"
    DATASTEPCOMPONENT = "datastepcomponent"


class EntityBase:

    def __init__(
        self,
        name: str,
        qualified_name: str,
        entity_type: str
    ) -> None:
        self._name = name
        self._qualified_name = qualified_name
        self._entity_type = entity_type

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def qualified_name(self) -> str:
        return self._qualified_name
    
    @property
    def entity_type(self) -> str:
        return self._entity_type

    @abstractmethod
    def entity_definition(self, guid: str) -> Union[AtlasEntity, AtlasProcess]:
        """
        Creates the atlas entity
        """
        pass

    @abstractmethod
    def type_def(self) -> Union[EntityTypeDef, None]:
        """
        Defines the atlas entity
        """
        pass



class EntityDatastepcomponent(EntityBase):

    def __init__(
        self,
        name: str,
        qualified_name: str,
        inputs: List[str] = None,
        outputs: List[str] = None,
        parameters: Dict[str, str] = None
        # guid: str = None
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._parameters = parameters
        super().__init__(
            name=name,
            qualified_name=qualified_name,
            entity_type=Entities.DATASTEPCOMPONENT
        )

    def type_def(self) -> EntityTypeDef:
        res = EntityTypeDef(
            name=self._entity_type,
            # attributeDefs=[
            #     AtlasAttributeDef(name="name", typename='string', cardinlity=Cardinality.SINGLE, isOptional=False)
            # ],
            superTypes=["DataSet"]
        )
        return res
   
    def entity_definition(self, guid: str) -> AtlasEntity:
        entity = AtlasEntity(
            name=self._name,
            typeName=self._entity_type,
            qualified_name=self.qualified_name,
            guid=guid
        )
        return entity


class EntityDatastep(EntityBase):

    def __init__(
        self,
        name: str,
        qualified_name: str,
        inputs: List[EntityDatastepcomponent] = None,
        outputs: List[EntityDatastepcomponent] = None

    ) -> None:
        self._inputs = inputs
        self._outputs = outputs

        super().__init__(
            qualified_name=qualified_name,
            name=name,
            entity_type=Entities.DATASTEP
        )
    
    @property
    def inputs(self):
        return self._inputs
        
    @property
    def outputs(self):
        return self._outputs

    def type_def(self) -> None:
        return None
   
    def entity_definition(
        self,
        guid: str,
        inputs_guids: List[Dict],
        outputs_guids: List[Dict]
    ) -> AtlasProcess:
        atlas_entity = AtlasProcess(
            name=self.name,
            typeName=self.entity_type,
            qualified_name=self.qualified_name,
            guid=guid,
            inputs=inputs_guids,
            outputs=outputs_guids
        )
        return atlas_entity
        

class PurviewHandler:

    def __init__(
        self,
        sp: ServicePrincipalAuthentication,
        name: str,
        entity_root: str = "pyapacheatlas"
    ) -> None:
        self.__sp = sp
        self._purview_client = PurviewClient(
            account_name=name,
            authentication=self.__sp
        )
        self._guid_tracker = GuidTracker()
        self.__entity_root = entity_root
    
    @property
    def entity_root(self) -> str:
        return self.__entity_root
    
    @entity_root.setter
    def entity_root(self, value):
        self.__entity_root = value
    
    def create_guid(self):
        return self._guid_tracker.get_guid()

    def build_qualified_name(
        self,
        name: str
    ) -> str:
        qualified_name = f"{self.__entity_root}://{name}"
        return qualified_name
    
    def build_atlasentity(
        self,
        name: str,
        typeName: str,
        qualified_name: str = None,
        attributes: Dict = {},
        relationshipAttributes: Dict = AtlasUnInit()
        # workspace: str = None
    ) -> AtlasEntity:

        if not qualified_name:
            qn = self.build_qualified_name(name=name)
        else:
            qn = qualified_name
        atlas_entity = AtlasEntity(
            name=name,
            typeName=typeName,
            qualified_name=qn,
            guid=self._guid_tracker.get_guid(),
            attributes=attributes,
            relationshipAttributes=relationshipAttributes
            # workspace=workspace
        )
        return atlas_entity
    
    def build_atlasprocess(
        self,
        name,
        typeName,
        inputs,
        outputs,
        qualified_name: str = None,
        attributes: Dict = None
    ) -> AtlasProcess:
        if not qualified_name:
            qn = self.build_qualified_name(name=name)
        else:
            qn = qualified_name
        process = AtlasProcess(
            name=name,
            typeName=typeName,
            qualified_name=qn,
            inputs=inputs,
            outputs=outputs,
            guid=self._guid_tracker.get_guid(),
            attributes=attributes
        )
        return process

    def upload_entity(
        self,
        entity: Union[AtlasEntity, AtlasProcess]
    ) -> Dict:
        res = self._purview_client.upload_entities(
            entity
        )
        return res
    
    def exists(self):
        return True if self.details() else False

    def upload_typedefs_entity(
        self,
        entityDefs: List[EntityTypeDef],
        force_update: bool = True
    ):
        typedef_results = None
        try:
            typedef_results = self._purview_client.upload_typedefs(
                entityDefs=entityDefs,
                force_update=force_update
            )
        except:
            print(f"Entities already exists: {', '.join([t.name for t in entityDefs])}")
        return typedef_results
    
    def upload_typedefs_relation(
        self,
        relationshipDefs: List[RelationshipTypeDef],
        entityDefs: List[EntityTypeDef] = None,
        force_update: bool = True
    ):
        typedef_results = None
        try:
            if entityDefs:
                typedef_results = self._purview_client.upload_typedefs(
                    entityDefs=entityDefs,
                    relationshipDefs=relationshipDefs,
                    force_update=force_update
                )
            else:
                typedef_results = self._purview_client.upload_typedefs(
                    relationshipDefs=relationshipDefs,
                    force_update=force_update
                )
        except:
            print(f"Relation already exists: {', '.join([t.name for t in relationshipDefs])}")
        return typedef_results
    
    def get_entity_details(
        self,
        qualifiedName,
        typeName
    ) -> Dict:
        entities = self._purview_client.get_entity(
            qualifiedName=[qualifiedName],
            typeName=typeName
        )
        for entity in entities.get("entities"):
            entity = entity
            break
        return entity
    #get_entity_details('https://sampledataadls.dfs.core.windows.net/masterdata/employees.csv','azure_datalake_gen2_path')

    def get_entity_guid(self, qualifiedName, typeName):
        entity = self.get_entity_details(
            qualifiedName=qualifiedName,
            typeName=typeName
        )
        return entity.get("guid")
    #get_entity_guid('https://sampledataadls.dfs.core.windows.net/creditriskdata/borrower.csv','azure_datalake_gen2_path')

    def entity_exists(self, qualifiedName, typeName):
        res = self.get_entity_details(
            qualifiedName=qualifiedName,
            typeName=typeName
        )
        return True if res else False

    def upload_Entity(
        self,
        entity: EntityBase
    ):
        if isinstance(entity, EntityDatastep):
            inputs = []
            for e in entity.inputs:
                ent_def = self.upload_typedefs_entity(
                    entityDefs=[e.type_def()]
                )
                ne = e.entity_definition(
                    guid=self.create_guid()
                )
                prv_entity = self.upload_entity(
                    entity=ne
                )
                guid = self.get_entity_guid(
                    qualifiedName=e.qualified_name,
                    typeName=e.entity_type
                )
                inputs.append({'guid': guid})
            
            outputs = []
            for e in entity.outputs:
                ent_def = self.upload_typedefs_entity(
                    entityDefs=[e.type_def()]
                )
                ne = e.entity_definition(
                    guid=self.create_guid()
                )
                prv_entity = self.upload_entity(
                    entity=ne
                )
                guid = self.get_entity_guid(
                    qualifiedName=e.qualified_name,
                    typeName=e.entity_type
                )
                outputs.append({'guid': guid})
            
            dataset_entity = entity.entity_definition(
                guid=self.create_guid(),
                inputs_guids=inputs,
                outputs_guids=outputs
            )
            prv_entity = self.upload_entity(entity=dataset_entity)

