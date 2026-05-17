import hashlib
from dataclasses import dataclass, field
from enum import Enum

from ceba.utils.json import json_dumps_deterministic, normalize_json_safe
from ceba.utils.typing import (
    copy_string_keyed_dict,
    ensure_not_forbidden_key,
    ensure_string_keyed_dict,
)

__all__ = [
    "HashAlgorithm",
    "StructuralHash",
    "canonical_json_bytes",
    "sha256_hex",
    "hash_json_value",
    "create_structural_hash",
]


class HashAlgorithm(Enum):
    SHA256 = "SHA256"


@dataclass(frozen=True)
class StructuralHash:
    algorithm: HashAlgorithm
    digest: str
    source_kind: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, HashAlgorithm):
            raise TypeError("algorithm must be HashAlgorithm")
        if not isinstance(self.digest, str):
            raise TypeError("digest must be a string")
        checked_digest: str = self.digest.strip()
        if len(checked_digest) != 64 or checked_digest.lower() != checked_digest:
            raise ValueError("digest must be lowercase SHA-256 hex")
        if any(character not in "0123456789abcdef" for character in checked_digest):
            raise ValueError("digest must be lowercase SHA-256 hex")
        object.__setattr__(self, "digest", checked_digest)
        object.__setattr__(self, "source_kind", ensure_not_forbidden_key(self.source_kind, "source_kind"))
        object.__setattr__(self, "metadata", copy_string_keyed_dict(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm.value,
            "digest": self.digest,
            "source_kind": self.source_kind,
            "metadata": dict(self.metadata),
        }


def canonical_json_bytes(value: object) -> bytes:
    normalize_json_safe(value)
    return json_dumps_deterministic(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def hash_json_value(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def create_structural_hash(
    *,
    value: object,
    source_kind: str,
    metadata: dict[str, object] | None = None,
) -> StructuralHash:
    checked_source_kind: str = ensure_not_forbidden_key(source_kind, "source_kind")
    checked_metadata: dict[str, object] = copy_string_keyed_dict(metadata, "metadata")
    base_metadata: dict[str, object] = {
        "source": "structural_hash",
        "source_kind": checked_source_kind,
    }
    base_metadata.update({key: item for key, item in checked_metadata.items() if key != "source"})
    ensure_string_keyed_dict(base_metadata, "metadata")
    return StructuralHash(
        algorithm=HashAlgorithm.SHA256,
        digest=hash_json_value(value),
        source_kind=checked_source_kind,
        metadata=base_metadata,
    )
