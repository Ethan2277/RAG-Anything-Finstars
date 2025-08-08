"""
Resource management for RAGAnything, including input files and parsed content.

This module provides functionality to manage resources such as input files, parsed content, and images.
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from pydantic import Field
from typing import Any

from lightrag.utils import compute_mdhash_id
from raganything.parser import MineruParser

class ResourceMixin:
    """ResourceMixin class containing resource management for RAGAnything"""
    resource_storage: str = "PGResourceStorage"
    """Default resource storage type for database management"""
    
    def _check_resource_management(self) -> bool:
        """
        Check if resource management is enabled in the configuration.
        
        Returns:
            bool: True if resource management is enabled, False otherwise.
        """
        return self.config.resource_management

    async def _initialize_resources_storage(self):
        """
        Initialize the resource storage for managing input files and parsed content.
        This method sets up the necessary storage for file resources, parsed content, and parsed images.
        """
        if not self._check_resource_management():
            self.logger.warning("Resource management is not enabled.")
            return
        
        # Initialize resource storage using LightRAG's KV storage
        self.file_resource = self.lightrag.key_string_value_json_storage_cls(
            namespace="file_resource",
            workspace=self.lightrag.workspace,
            global_config=self.lightrag.__dict__,
            embedding_func=self.embedding_func,
        )
        self.parsed_content = self.lightrag.key_string_value_json_storage_cls(
            namespace="parsed_content_list",
            workspace=self.lightrag.workspace,
            global_config=self.lightrag.__dict__,
            embedding_func=self.embedding_func,
        )
        self.parsed_images = self.lightrag.key_string_value_json_storage_cls(
            namespace="parsed_images",
            workspace=self.lightrag.workspace,
            global_config=self.lightrag.__dict__,
            embedding_func=self.embedding_func,
        )
        self.parsed_markdown = self.lightrag.key_string_value_json_storage_cls(
            namespace="parsed_markdown",
            workspace=self.lightrag.workspace,
            global_config=self.lightrag.__dict__,
            embedding_func=self.embedding_func,
        )
        tasks = []
        for storage in (
            self.file_resource,
            self.parsed_content,
            self.parsed_images,
            self.parsed_markdown,
        ):
            if storage:
                tasks.append(storage.initialize())

        await asyncio.gather(*tasks)

    async def _finalize_resources_storage(self):
        """
        Finalize the resource storage by performing any necessary cleanup or finalization steps.
        """
        if not self._check_resource_management():
            self.logger.warning("Resource management is not enabled.")
            return

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Finalize resource storage using LightRAG's KV storage
        await self.file_resource.finalize()
        await self.parsed_content.finalize()
        await self.parsed_images.finalize()
    
    def _check_resource_storage_initialized(self) -> bool:
        """
        Check if the resource storage is initialized.
        
        Returns:
            bool: True if the resource storage is initialized, False otherwise.
        """
        if not self._check_resource_management():
            return False
        
        return (
            hasattr(self, "file_resource") and self.file_resource.db is not None and
            hasattr(self, "parsed_content") and self.parsed_content.db is not None and
            hasattr(self, "parsed_images") and self.parsed_images.db is not None and
            hasattr(self, "parsed_markdown") and self.parsed_markdown.db is not None
        )

    async def store_input_file_resource(
        self, 
        file_path: str | Path,
    ):
        """
        Store an input file resource in the resource storage.
        
        Args:
            file_path: Path to the input file.
            file_name: Name of the input file.
            file_content: Content of the input file.
            file_type: Type of the input file (e.g., 'txt', 'pdf').
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        file_name = file_path.stem
        file_type = file_path.suffix[1:]  # Get file extension without the dot
        file_content = await self._read_file_content(file_path)
            
        inserting_file = {
            compute_mdhash_id(file_content, prefix=f'doc_'): {
                "file_name": file_name,
                "file_type": file_type,
                "file_path": str(file_path),
                "file_content": file_content,
            }
        }
        await self.file_resource.upsert(inserting_file)
        
    async def store_parsed_content_list_resource(
        self, 
        input_file_path: str | Path,
        parsed_content_list: list
    ):
        """
        Store a list of parsed content in the resource storage.
        
        Args:
            input_file_path: Path to the input file from which content was parsed.
            parsed_content_list: List of parsed content strings.
        """
        file_content = await self._read_file_content(input_file_path)
        inserting_contents= {
            compute_mdhash_id(json.dumps(content), prefix='parsed_content_'): {
                "content": content,
                "file_path": str(input_file_path),
                "doc_id": compute_mdhash_id(file_content, prefix='doc_'),
            }
            for content in parsed_content_list
        }
        await self.parsed_content.upsert(inserting_contents)

    async def store_parsed_images_resource(
        self, 
        input_file_path: str | Path,
        parsed_images_dir: str | Path
    ):
        """
        Store parsed images in the resource storage.
        
        Args:
            input_file_path: Path to the input file from which images were parsed.
            parsed_images_dir: Directory containing parsed images.
        """
        if not os.path.exists(parsed_images_dir):
            self.logger.warning(f"Parsed images directory {parsed_images_dir} does not exist.")
            return
        
        file_content = await self._read_file_content(input_file_path)
        
        inserting_images = {}
        for image_file in os.listdir(parsed_images_dir):
            image_path = os.path.join(parsed_images_dir, image_file)
            if os.path.isfile(image_path):
                with open(image_path, 'rb') as f:
                    image_content = base64.b64encode(f.read()).decode('utf-8')
                inserting_images[compute_mdhash_id(image_content, prefix='parsed_image_')] = {
                    "image_path": str(image_path),
                    "image_content": image_content,
                    "file_path": str(input_file_path),
                    "doc_id": compute_mdhash_id(file_content, prefix='doc_'),
                }

        await self.parsed_images.upsert(inserting_images)
        
    async def store_parsed_markdown_resource(
        self, 
        input_file_path: str,
        parsed_markdown_content: str
    ):
        """
        Store parsed markdown content in the resource storage.
        
        Args:
            input_file_path: Path to the input file from which markdown was parsed.
            parsed_markdown_content: Parsed markdown content as a string.
        """
        file_content = await self._read_file_content(input_file_path)
        
        inserting_markdown = {
            compute_mdhash_id(parsed_markdown_content, prefix='parsed_markdown_'): {
                "markdown_content": parsed_markdown_content,
                "file_path": str(input_file_path),
                "doc_id": compute_mdhash_id(file_content, prefix='doc_'),
            }
        }
        await self.parsed_markdown.upsert(inserting_markdown)

    async def store_parsed_result_resource(
        self, 
        input_file_path: str | Path,
        parser_output_dir: str | Path = None, 
        parse_method: str = None,
        **kwargs
    ):
        # TODO: User-defined doc id plugin
        if parser_output_dir is None:
            parser_output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        backend = kwargs.get("backend", "")
        if backend.startswith("vlm-"):
            parse_method = "vlm"
        if isinstance(input_file_path, str):
            input_file_path = Path(input_file_path)
        if isinstance(parser_output_dir, str):
            parser_output_dir = Path(parser_output_dir)
            
        if self.config.parser != 'mineru':
            self.logger.warning(
                "RAGAnything currently only supports 'mineru' parser for storing parsed results."
            )
            return
        file_stem=input_file_path.stem
        content_list, md_content = MineruParser()._read_output_files(
            output_dir=parser_output_dir,
            file_stem=file_stem,
            method=parse_method,
        )
        await self.store_parsed_content_list_resource(
            input_file_path=input_file_path,
            parsed_content_list=content_list
        )
        await self.store_parsed_images_resource(
            input_file_path=input_file_path,
            parsed_images_dir=os.path.join(parser_output_dir, file_stem, parse_method, 'images')
        )
        await self.store_parsed_markdown_resource(
            input_file_path=input_file_path,
            parsed_markdown_content=md_content
        )
        
    async def activate_resource_management(self):
        """
        Activate resource management by initializing the resource storage.
        This method should be called at the start of the application to ensure resources are managed properly.
        """
        if not self._check_resource_management():
            self.logger.warning("Resource management is not enabled in the configuration.")
            return
        
        await self._ensure_lightrag_initialized()
        await self._initialize_resources_storage()
        self.logger.info("Resource management initialized successfully.")
        
        
    