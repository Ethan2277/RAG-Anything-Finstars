"""
Multi-user, multi-session configuration functionality for RAGAnything

Modify the configuration to support multiple users and sessions, allowing for better organization and management of user interactions.
"""

import os
from typing import Any
from lightrag.utils import get_env_value, logger

class SessionConfigMixin:
    """SessionConfigMixin class containing multi-user, multi-session configuration functionality for RAGAnything"""

    def _check_multi_session_management(self):
        """Check if multi-session management is enabled"""
        if self.config.user_id is None or self.config.session_id is None:
            return False
        logger.info(
            f"Multi-session management enabled for user {self.config.user_id} "
            f"in session {self.config.session_id}"
        )
        return True
    
    async def _modify_lightrag_config_for_session(
        self,
        user_id: str = None,
        session_id: str = None,
        lightrag_params: dict[str, Any] = None,
        **kwargs,
    ):
        """
        Modify configuration for a specific user session

        Args:
            user_id: Unique identifier for the user session
            session_id: Unique identifier for the session
            lightrag_params: Existing LightRAG parameters to modify
            **kwargs: Additional configuration parameters
        """
        
        if user_id is None:
            user_id = self.config.user_id
        if session_id is None:
            session_id = self.config.session_id

        # Update configuration when using PostgreSQL as lightrag storage only, in the future we may support other storage types
        # TODO: Add support for other storage types
        if 'kv_storage' not in lightrag_params or lightrag_params['kv_storage'] != 'PGKVStorage':
            self.logger.warning(
                "SessionConfigMixin is only supported with PGKVStorage, "
                "please set lightrag.kv_storage to 'PGKVStorage' to use this feature."
            )
            return
        if 'vector_storage' not in lightrag_params or lightrag_params['vector_storage'] != 'PGVectorStorage':
            self.logger.warning(
                "SessionConfigMixin is only supported with PGVectorStorage, "
                "please set lightrag.vector_storage to 'PGVectorStorage' to use this feature."
            )
            return
        if 'graph_storage' not in lightrag_params or lightrag_params['graph_storage'] != 'PGGraphStorage':
            self.logger.warning(
                "SessionConfigMixin is only supported with PGGraphStorage, "
                "please set lightrag.graph_storage to 'PGGraphStorage' to use this feature."
            )
            return
        if 'doc_status_storage' not in lightrag_params or lightrag_params['doc_status_storage'] != 'PGDocStatusStorage':
            self.logger.warning(
                "SessionConfigMixin is only supported with PGDocStatusStorage, "
                "please set lightrag.doc_status_storage to 'PGDocStatusStorage' to use this feature."
            )
            return
        
        POSTGRES_DATABASE = get_env_value("POSTGRES_DATABASE", None, str)
        if POSTGRES_DATABASE is None:
            self.logger.warning(
                "SessionConfigMixin requires POSTGRES_DATABASE to be set, "
                "please set it in your environment variables."
            )
            return
        # Update the configuration with user and session IDs
        os.environ["POSTGRES_DATABASE"] = f"{user_id}_{session_id}_{POSTGRES_DATABASE}"