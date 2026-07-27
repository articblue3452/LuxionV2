"""ChromaDB-backed storage for durable semantic user preferences."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb


class SemanticMemoryError(RuntimeError):
    """Raised when semantic memory cannot be read or persisted."""


class SemanticMemory:
    """Persist stable user preferences in a local ChromaDB collection.

    The public methods mirror the former JSON-backed implementation so callers
    do not need to know which persistence backend is in use. Documents contain
    the remembered value, while metadata identifies the preference key and its
    semantic-memory namespace.
    """

    _COLLECTION_NAME = "semantic_memory"
    _MEMORY_TYPE = "semantic"
    _ID_PREFIX = "semantic_"
    _VALUE_TYPE_FIELD = "value_type"

    def __init__(self) -> None:
        database_path = Path(__file__).with_name("chroma_db")

        try:
            self._client = chromadb.PersistentClient(path=str(database_path))
            # get_or_create_collection preserves an existing collection and
            # creates it only on the first run. Embeddings are intentionally
            # disabled for today's key/value API; a future RAG method can call
            # collection.query with explicitly supplied embeddings.
            self._collection = self._client.get_or_create_collection(
                name=self._COLLECTION_NAME,
                embedding_function=None,
            )
            self._migrate_legacy_json_if_needed()
        except Exception as error:
            raise SemanticMemoryError(
                f"Unable to initialise semantic memory at {database_path}."
            ) from error

    def load(self) -> dict[str, Any]:
        """Return every remembered value, matching the former JSON API."""
        return self.all()

    def save(self, data: dict[str, Any]) -> None:
        """Replace stored semantic memory with ``data``.

        This preserves the legacy JSON store's ``save`` semantics for any
        external caller while using ChromaDB as the only persistence backend.
        """
        if not isinstance(data, Mapping):
            raise ValueError("Semantic memory must contain a mapping.")

        desired = {self._validate_key(key): value for key, value in data.items()}
        existing_keys = set(self.all())

        for key, value in desired.items():
            self.set(key, value)

        for key in existing_keys - set(desired):
            self.delete(key)

    def get(self, key: str) -> Any | None:
        """Return the value for ``key``, or ``None`` when it is not stored."""
        normalized_key = self._validate_key(key)

        try:
            result = self._collection.get(
                where={"key": normalized_key},
                include=["documents", "metadatas"],
            )
        except Exception as error:
            raise SemanticMemoryError(
                f"Unable to retrieve semantic memory for key {normalized_key!r}."
            ) from error

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents:
            return None

        return self._deserialize_value(documents[0], metadatas[0])

    def set(self, key: str, value: Any) -> None:
        """Create or update the deterministic document for ``key``."""
        normalized_key = self._validate_key(key)
        document, value_type = self._serialize_value(value)
        memory_id = self._memory_id(normalized_key)
        metadata = {
            "key": normalized_key,
            "memory_type": self._MEMORY_TYPE,
            self._VALUE_TYPE_FIELD: value_type,
        }

        try:
            existing = self._collection.get(ids=[memory_id], include=[])
            if existing.get("ids"):
                self._collection.update(
                    ids=[memory_id], documents=[document], metadatas=[metadata]
                )
            else:
                self._collection.add(
                    ids=[memory_id], documents=[document], metadatas=[metadata]
                )
        except Exception as error:
            raise SemanticMemoryError(
                f"Unable to persist semantic memory for key {normalized_key!r}."
            ) from error

    def delete(self, key: str) -> None:
        """Delete the deterministic document for ``key`` if it exists."""
        normalized_key = self._validate_key(key)

        try:
            self._collection.delete(ids=[self._memory_id(normalized_key)])
        except Exception as error:
            raise SemanticMemoryError(
                f"Unable to delete semantic memory for key {normalized_key!r}."
            ) from error

    def all(self) -> dict[str, Any]:
        """Return all semantic memories as a ``{key: value}`` mapping."""
        try:
            result = self._collection.get(
                where={"memory_type": self._MEMORY_TYPE},
                include=["documents", "metadatas"],
            )
        except Exception as error:
            raise SemanticMemoryError("Unable to retrieve semantic memory.") from error

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return {
            metadata["key"]: self._deserialize_value(document, metadata)
            for document, metadata in zip(documents, metadatas, strict=True)
        }

    @classmethod
    def _memory_id(cls, key: str) -> str:
        """Build the deterministic ChromaDB identifier for a memory key."""
        return f"{cls._ID_PREFIX}{key}"

    def _migrate_legacy_json_if_needed(self) -> None:
        """Import the former JSON store once, without overwriting Chroma data."""
        legacy_path = Path(__file__).with_name("semantic.json")
        if not legacy_path.exists() or self._collection.count() != 0:
            return

        try:
            legacy_data = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SemanticMemoryError(
                f"Unable to read legacy semantic memory at {legacy_path}."
            ) from error

        if not isinstance(legacy_data, dict):
            raise SemanticMemoryError("Legacy semantic memory must contain a JSON object.")

        for key, value in legacy_data.items():
            self.set(key, value)

    @staticmethod
    def _validate_key(key: str) -> str:
        """Ensure a key can be used in Chroma metadata and deterministic IDs."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Semantic memory key must be a non-empty string.")
        return key.strip()

    @staticmethod
    def _serialize_value(value: Any) -> tuple[str, str]:
        """Convert a scalar value to a Chroma document without losing its type."""
        if isinstance(value, str):
            return value, "str"
        if isinstance(value, bool):
            return str(value).lower(), "bool"
        if isinstance(value, (int, float)):
            return str(value), type(value).__name__
        if value is None:
            return "null", "none"
        raise ValueError("Semantic memory value must be a scalar value.")

    @staticmethod
    def _deserialize_value(document: str, metadata: Mapping[str, Any]) -> Any:
        """Restore the original scalar type from a stored Chroma document."""
        value_type = metadata.get(SemanticMemory._VALUE_TYPE_FIELD, "str")
        if value_type == "bool":
            return document.lower() == "true"
        if value_type == "int":
            return int(document)
        if value_type == "float":
            return float(document)
        if value_type == "none":
            return None
        return document
