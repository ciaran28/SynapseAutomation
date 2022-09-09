import os
from typing import Any
from typing import Dict
from dstoolkit.core import Log
from dstoolkit.core import Singleton
from dstoolkit.core import trace
from dstoolkit.dev import DevelopmentClient
from dstoolkit.dev import DevelopmentEngine
from dstoolkit.dev import IdeEnvironment
from pyspark.sql import SparkSession


class Engine(metaclass=Singleton):

    """
    This is the core of the framework.
    Configures the environment to interact with the remote Synapse cluster.
    """

    def __init__(self):
        """
        Instantiate the current object

        """
        self.__ide_environment = DevelopmentEngine().get_instance().ide_environment
        self.ai_cs = None

    def initialize_env(self):
        """
        Initializes the DevelopmentClient.
        That is, sets the mssparkutils and spark context accordingly if the code is runt on cluster or locally.
        """
        DevelopmentClient(
            mssparkutils=DevelopmentEngine().get_instance().mssparkutils,
            spark=DevelopmentEngine().get_instance().spark,
            ide_environment=self.__ide_environment
        )

    def initialize_logger(
        self,
        pipeline_name: str,
        aics_kv_secret: str = None,
        aics_kv_name: str = None,
        aics: str = None
    ):
        """
        Initializes the logger

        Parameters
        ----------
        pipeline_name : str
            Name to use with the logger. It will be the base name used for all the upcoming logs and tracing
        aics_kv_name : str, optional
            Name of the Azure Key Vault, by default None
        aics_kv_secret : str, optional
            Secret name where the Application Insight connection string is stored, by default None
        aics: str, optional
            Application Insight connection string, by default None

        Raises
        ------
        ValueError
            Unknown Ide Environment used
        """
        # Configuring application insight key
        if self.__ide_environment == IdeEnvironment.LOCAL:
            self.ai_cs = aics
        elif self.__ide_environment == IdeEnvironment.REMOTE:
            self.ai_cs = DevelopmentEngine().\
                get_instance().\
                mssparkutils.\
                credentials.\
                getSecret(aics_kv_name, aics_kv_secret)
        else:
            raise ValueError(f'ide_environment unknown: {self.__ide_environment}')
        # Instantiating logger
        Log(pipeline_name, self.ai_cs)

    def spark(self) -> SparkSession:
        """
        Current spark context

        Returns
        -------
        SparkSession
            Spark context
        """
        return DevelopmentClient.get_instance().spark

    def mssparkutils(self):
        """
        Current mssparkutils

        Returns
        -------
        mssparkutils
            The mssparkutils
        """
        return DevelopmentClient.get_instance().mssparkutils

    @classmethod
    def get_instance(cls):
        """
        Current singleton Engine

        Returns
        -------
        Engine
            The Engine
        """
        return Engine()

    @staticmethod
    def ide_environment() -> IdeEnvironment:
        """
        Current Ide Environment

        Returns
        -------
        IdeEnvironment
            The Ide Environment
        """
        return DevelopmentClient.get_instance().ide_environment

    @staticmethod
    def is_ide_remote() -> bool:
        """
        Checks if the current environment is remote

        Returns
        -------
        bool
            Check result
        """
        return DevelopmentClient.get_instance().ide_environment == IdeEnvironment.REMOTE

    @staticmethod
    def is_ide_local() -> bool:
        """
        Checks if the current environment is Local

        Returns
        -------
        bool
            Check result
        """
        return DevelopmentClient.get_instance().ide_environment == IdeEnvironment.LOCAL

    def run_notebook_with_retry(
        self,
        notebook: str,
        args: Dict,
        timeout=86400,
        max_retries=3
    ) -> Any:
        """
        Runs the specified notebook through mssparkutils

        Parameters
        ----------
        notebook : str
            Name or path of the notebook
        args : Dict
            Arguments passed to the notebook
        timeout : int, optional
            run timeout in seconds, by default 86400
        max_retries : int, optional
            [description], by default 3

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        e
            [description]
        """
        num_retries = 0
        while True:
            try:
                return DevelopmentClient.get_instance().mssparkutils.notebook.run(notebook, timeout, args)
            except Exception as e:
                if num_retries > max_retries:
                    raise e
                print("Retrying error"), e
                num_retries += 1

    @trace
    # TODO: rename check
    def run_notebook(self, notebook: str, args: Dict, timeout=86400, error_raise=True):
        try:
            res = DevelopmentClient.get_instance().mssparkutils.notebook.run(notebook, timeout, args)
        except Exception as e:
            res = f"Notebook {notebook} failed"
            Log.get_instance().log_error(res)
            if error_raise:
                raise e
        return res
