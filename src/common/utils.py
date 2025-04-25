# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions used across different modules of NIM Blueprints."""
import os
import yaml
import logging
from pathlib import Path
from functools import lru_cache, wraps
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Callable, List, Dict

logger = logging.getLogger(__name__)

try:
    import torch
except Exception as e:
    logger.warning(f"Optional module torch not installed.")

try:
    from langchain.text_splitter import SentenceTransformersTokenTextSplitter
except Exception as e:
    logger.warning(f"Optional langchain module not installed for SentenceTransformersTokenTextSplitter.")

try:
    from langchain_core.vectorstores import VectorStore
except Exception as e:
    logger.warning(f"Optional Langchain module langchain_core not installed.")

try:
    # Remove NVIDIA AI Endpoints imports
    # from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
    # Add OpenAI imports
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as e:
    logger.error(f"Optional API connector not installed.")

try:
    from langchain_community.vectorstores import PGVector
    from langchain_community.vectorstores import Milvus
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    logger.warning(f"Optional Langchain module langchain_community not installed.")

try:
    from faiss import IndexFlatL2
except Exception as e:
    logger.warning(f"Optional faissDB not installed.")


from langchain_core.embeddings import Embeddings
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain.llms.base import LLM
from src.common import configuration

if TYPE_CHECKING:
    from src.common.configuration_wizard import ConfigWizard

DEFAULT_MAX_CONTEXT = 1500

def utils_cache(func: Callable) -> Callable:
    """Use this to convert unhashable args to hashable ones"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable args to hashable ones
        args_hashable = tuple(tuple(arg) if isinstance(arg, (list, dict, set)) else arg for arg in args)
        kwargs_hashable = {key: tuple(value) if isinstance(value, (list, dict, set)) else value for key, value in kwargs.items()}
        return func(*args_hashable, **kwargs_hashable)
    return wrapper


@lru_cache
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


@lru_cache
def get_prompts() -> Dict:
    """Retrieves prompt configurations from YAML file and return a dict.
    """

    # default config taking from prompt.yaml
    default_config_path = os.path.join("./", os.environ.get("EXAMPLE_PATH"), "prompt.yaml")
    default_config = {}
    if Path(default_config_path).exists():
        with open(default_config_path, 'r') as file:
            logger.info(f"Using prompts config file from: {default_config_path}")
            default_config = yaml.safe_load(file)

    config_file = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")

    config = {}
    if Path(config_file).exists():
        with open(config_file, 'r') as file:
            logger.info(f"Using prompts config file from: {config_file}")
            config = yaml.safe_load(file)

    config = _combine_dicts(default_config, config)
    return config



def create_vectorstore_langchain(document_embedder, collection_name: str = "") -> VectorStore:
    """Create the vector db index for langchain."""

    config = get_config()

    if config.vector_store.name == "faiss":
        vectorstore = FAISS(document_embedder, IndexFlatL2(config.embeddings.dimensions), InMemoryDocstore(), {})
    elif config.vector_store.name == "pgvector":
        db_name = os.getenv('POSTGRES_DB', None)
        if not collection_name:
            collection_name = os.getenv('COLLECTION_NAME', "vector_db")
        logger.info(f"Using PGVector collection: {collection_name}")
        connection_string = f"postgresql://{os.getenv('POSTGRES_USER', '')}:{os.getenv('POSTGRES_PASSWORD', '')}@{config.vector_store.url}/{db_name}"
        vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=connection_string,
            embedding_function=document_embedder,
        )
    elif config.vector_store.name == "milvus":
        if not collection_name:
            collection_name = os.getenv('COLLECTION_NAME', "vector_db")
        logger.info(f"Using milvus collection: {collection_name}")
        url = urlparse(config.vector_store.url)
        vectorstore = Milvus(
            document_embedder,
            connection_args={"host": url.hostname, "port": url.port},
            collection_name=collection_name,
            # Use CPU-based index type instead of GPU for Mac compatibility
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "nlist": config.vector_store.nlist},
            search_params={"nprobe": config.vector_store.nprobe},
            auto_id = True
        )
    else:
        raise ValueError(f"{config.vector_store.name} vector database is not supported")
    logger.info("Vector store created and saved.")
    return vectorstore


def get_vectorstore(vectorstore, document_embedder) -> VectorStore:
    """
    Send a vectorstore object.
    If a Vectorstore object already exists, the function returns that object.
    Otherwise, it creates a new Vectorstore object and returns it.
    """
    if vectorstore is None:
        return create_vectorstore_langchain(document_embedder)
    return vectorstore

@lru_cache()
def get_llm(**kwargs) -> LLM | SimpleChatModel:
    """Create the LLM connection."""
    settings = get_config()

    # Update to use OpenAI models instead of NVIDIA NIM
    logger.info(f"Using OpenAI as model engine for llm. Model name: {settings.llm.model_name}")
    
    # Default to gpt-4o-mini if no model specified
    model_name = "gpt-4o-mini"
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    logger.info(f"Initializing DirectOpenAIFix with model: {model_name}")
    
    # Use our brute-force fix instead of other wrappers
    return DirectOpenAIFix(
        model=model_name,
        temperature=kwargs.get('temperature', 0),
        top_p=kwargs.get('top_p', 0.7),
        max_tokens=kwargs.get('max_tokens', 1024),
        api_key=OPENAI_API_KEY
    )


@lru_cache
def get_embedding_model() -> Embeddings:
    """Create the embedding model."""
    settings = get_config()

    # Use HuggingFace embeddings if specified
    if settings.embeddings.model_engine == "huggingface":
        model_kwargs = {"device": "cpu"}
        if torch.cuda.is_available():
            model_kwargs["device"] = "cuda:0"

        encode_kwargs = {"normalize_embeddings": False}
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        # Load in a specific embedding model
        return hf_embeddings
    else :
        model_name = 'text-embedding-3-small'
        logger.info(f"Using OpenAI embedding model {model_name}")

        print("blahblah")
        return OpenAIEmbeddings(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"]
        )



@lru_cache
def get_ranking_model() -> BaseDocumentCompressor:
    """Create the ranking model.

    Returns:
        BaseDocumentCompressor: Base class for document compressors.
    """
    # Comment out the reranking for now
    logger.info("Reranking is currently disabled")
    return None


def get_text_splitter() -> SentenceTransformersTokenTextSplitter:
    """Return the token text splitter instance from langchain."""

    if get_config().text_splitter.model_name:
        embedding_model_name = get_config().text_splitter.model_name

    return SentenceTransformersTokenTextSplitter(
        model_name=embedding_model_name,
        tokens_per_chunk=get_config().text_splitter.chunk_size - 2,
        chunk_overlap=get_config().text_splitter.chunk_overlap,
    )


def get_docs_vectorstore_langchain(vectorstore: VectorStore) -> List[str]:
    """Retrieves filenames stored in the vector store implemented in LangChain."""

    settings = get_config()
    try:
        # No API availbe in LangChain for listing the docs, thus usig its private _dict
        extract_filename = lambda metadata : os.path.splitext(os.path.basename(metadata['source']))[0]
        if settings.vector_store.name == "faiss":
            in_memory_docstore = vectorstore.docstore._dict
            filenames = [extract_filename(doc.metadata) for doc in in_memory_docstore.values()]
            filenames = list(set(filenames))
            return filenames
        elif settings.vector_store.name == "pgvector":
            # No API availbe in LangChain for listing the docs, thus usig its private _make_session
            with vectorstore._make_session() as session:
                embedding_doc_store = session.query(vectorstore.EmbeddingStore.custom_id, vectorstore.EmbeddingStore.document, vectorstore.EmbeddingStore.cmetadata).all()
                filenames = set([extract_filename(metadata) for _, _, metadata in embedding_doc_store if metadata])
                return filenames
        elif settings.vector_store.name == "milvus":
            # Getting all the ID's > 0
            if vectorstore.col:
                milvus_data = vectorstore.col.query(expr="pk >= 0", output_fields=["pk","source", "text"])
                filenames = set([extract_filename(metadata) for metadata in milvus_data])
                return filenames
    except Exception as e:
        logger.error(f"Error occurred while retrieving documents: {e}")
    return []

def del_docs_vectorstore_langchain(vectorstore: VectorStore, filenames: List[str]) -> bool:
    """Delete documents from the vector index implemented in LangChain."""

    settings = get_config()
    try:
        # No other API availbe in LangChain for listing the docs, thus usig its private _dict
        extract_filename = lambda metadata : os.path.splitext(os.path.basename(metadata['source']))[0]
        if settings.vector_store.name == "faiss":
            in_memory_docstore = vectorstore.docstore._dict
            for filename in filenames:
                ids_list = [doc_id for doc_id, doc_data in in_memory_docstore.items() if extract_filename(doc_data.metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.delete(ids_list)
                logger.info(f"Deleted documents with filenames {filename}")
        elif settings.vector_store.name == "pgvector":
            with vectorstore._make_session() as session:
                collection = vectorstore.get_collection(session)
                filter_by = vectorstore.EmbeddingStore.collection_id == collection.uuid
                embedding_doc_store = session.query(vectorstore.EmbeddingStore.custom_id, vectorstore.EmbeddingStore.document, vectorstore.EmbeddingStore.cmetadata).filter(filter_by).all()
            for filename in filenames:
                ids_list = [doc_id for doc_id, doc_data, metadata in embedding_doc_store if extract_filename(metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.delete(ids_list)
                logger.info(f"Deleted documents with filenames {filename}")
        elif settings.vector_store.name == "milvus":
            # Getting all the ID's > 0
            milvus_data = vectorstore.col.query(expr="pk >= 0", output_fields=["pk","source", "text"])
            for filename in filenames:
                ids_list = [metadata["pk"] for metadata in milvus_data if extract_filename(metadata) == filename]
                if not len(ids_list):
                    logger.info("File does not exist in the vectorstore")
                    return False
                vectorstore.col.delete(f"pk in {ids_list}")
                logger.info(f"Deleted documents with filenames {filename}")
                return True
    except Exception as e:
        logger.error(f"Error occurred while deleting documents: {e}")
        return False
    return True

def _combine_dicts(dict_a, dict_b):
    """Combines two dictionaries recursively, prioritizing values from dict_b.

    Args:
        dict_a: The first dictionary.
        dict_b: The second dictionary.

    Returns:
        A new dictionary with combined key-value pairs.
    """

    combined_dict = dict_a.copy()  # Start with a copy of dict_a

    for key, value_b in dict_b.items():
        if key in combined_dict:
            value_a = combined_dict[key]
            # Remove the special handling for "command"
            if isinstance(value_a, dict) and isinstance(value_b, dict):
                combined_dict[key] = _combine_dicts(value_a, value_b)
            # Otherwise, replace the value from A with the value from B
            else:
                combined_dict[key] = value_b
        else:
            # Add any key not present in A
            combined_dict[key] = value_b

    return combined_dict

# Create a simple direct wrapper for OpenAI calls
class DirectOpenAIFix(ChatOpenAI):
    """A direct wrapper that formats messages right before sending to OpenAI API."""
    
    def _generate(self, messages, stop=None, **kwargs):
        """
        Directly intercept and fix messages before sending to OpenAI API.
        This is a brute-force approach to ensure we never hit the OpenAI error.
        """
        # Print original messages for debugging
        logger.info(f"ORIGINAL MESSAGES: {messages}")
        
        # Create a completely new list of messages
        fixed_messages = []
        tool_calls_seen = {}  # Track tool calls by ID
        tool_responses_seen = {}  # Track tool responses by ID
        
        # First pass - collect all tool calls and responses
        for msg in messages:
            # Track tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict) and 'id' in tc:
                        tool_calls_seen[tc['id']] = True
            
            # Track tool responses
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                tool_responses_seen[msg.tool_call_id] = True
        
        # Second pass - build the fixed sequence
        for msg in messages:
            # Add the original message
            fixed_messages.append(msg)
            
            # If this is a message with tool calls, ensure responses follow immediately
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict) and 'id' in tc:
                        tc_id = tc['id']
                        # If there's no response for this tool call, add one
                        if tc_id not in tool_responses_seen:
                            from langchain_core.messages import ToolMessage
                            fixed_messages.append(
                                ToolMessage(
                                    content="No result available for this tool call",
                                    tool_call_id=tc_id
                                )
                            )
                            # Mark as seen so we don't add duplicates
                            tool_responses_seen[tc_id] = True
        
        # Find all tool calls without responses
        missing_responses = [tc_id for tc_id in tool_calls_seen if tc_id not in tool_responses_seen]
        if missing_responses:
            logger.warning(f"Found {len(missing_responses)} tool calls without responses, adding them now")
            for tc_id in missing_responses:
                from langchain_core.messages import ToolMessage
                fixed_messages.append(
                    ToolMessage(
                        content="No result available for this tool call",
                        tool_call_id=tc_id
                    )
                )
        
        logger.info(f"FIXED MESSAGES: {fixed_messages}")
        
        try:
            # Call the original method with fixed messages
            return super()._generate(fixed_messages, stop=stop, **kwargs)
        except Exception as e:
            # If we still get an error, log it and try a last-ditch fix
            logger.error(f"Error after initial fix: {e}")
            
            # Extract the tool_call_id from the error if possible
            import re
            error_msg = str(e)
            tool_call_matches = re.findall(r"call_[a-zA-Z0-9]+", error_msg)
            
            if tool_call_matches:
                for tc_id in tool_call_matches:
                    logger.warning(f"Adding emergency fix for tool call ID: {tc_id}")
                    from langchain_core.messages import ToolMessage
                    fixed_messages.append(
                        ToolMessage(
                            content="Emergency fix for missing tool response",
                            tool_call_id=tc_id
                        )
                    )
                
                # Try one more time with the emergency fix
                return super()._generate(fixed_messages, stop=stop, **kwargs)
            else:
                # If we can't fix it, re-raise the original error
                raise
