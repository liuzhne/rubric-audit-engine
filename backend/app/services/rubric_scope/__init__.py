"""评分细则的适用范围（docs/评分细则适用范围方案.md）。"""

from backend.app.services.rubric_scope.vocabulary import (
    CROSS_SECTION,
    DOCUMENT,
    MAX_CROSS_SECTIONS,
    SCOPE_SOURCES,
    SCOPE_TYPES,
    SECTION,
    VOCABULARY_VERSION,
    ScopeError,
    describe_scope,
    normalize_scope,
    read_scope,
    role_names,
    scope_schema,
    stored_scope,
    vocabulary,
    vocabulary_payload,
)

__all__ = [
    "CROSS_SECTION",
    "DOCUMENT",
    "MAX_CROSS_SECTIONS",
    "SCOPE_SOURCES",
    "SCOPE_TYPES",
    "SECTION",
    "VOCABULARY_VERSION",
    "ScopeError",
    "describe_scope",
    "normalize_scope",
    "read_scope",
    "role_names",
    "scope_schema",
    "stored_scope",
    "vocabulary",
    "vocabulary_payload",
]
