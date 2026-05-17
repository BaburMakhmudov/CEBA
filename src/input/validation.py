"""CEBA Input validation utilities.

Validation checks the technical form of input-layer structures before
downstream processing. It builds validation reports without computing meaning.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ois.input.errors import SchemaViolationError, ValidationFailedError
from ois.input.schema import (
    CanonicalContent,
    CanonicalContentKind,
    CanonicalObservation,
    ErrorSeverity,
    InputEnvelope,
    InputSource,
    ObservationMetadata,
    ObservationType,
    PayloadRef,
    PayloadStorageKind,
    RawObservation,
    SecurityReport,
    SecurityStatus,
    ValidationErrorItem,
    ValidationReport,
    ValidationStatus,
    ValidationWarningItem,
)
from ois.input.source import validate_source_boundary


_MAX_ERROR_ITEMS = 128
_MAX_WARNING_ITEMS = 128


def passed_report(*, schema_version: str = "1.0.0") -> ValidationReport:
    return ValidationReport(
        status=ValidationStatus.PASSED,
        is_valid=True,
        errors=[],
        warnings=[],
        schema_version=schema_version,
    )


def failed_report(
    *,
    errors: list[ValidationErrorItem],
    warnings: list[ValidationWarningItem] | None = None,
    schema_version: str = "1.0.0",
) -> ValidationReport:
    if not errors:
        raise SchemaViolationError(
            "failed_report requires at least one error",
            field_path="errors",
        )
    return ValidationReport(
        status=ValidationStatus.FAILED,
        is_valid=False,
        errors=errors[:_MAX_ERROR_ITEMS],
        warnings=(warnings or [])[:_MAX_WARNING_ITEMS],
        schema_version=schema_version,
    )


def warning_report(
    *,
    warnings: list[ValidationWarningItem],
    schema_version: str = "1.0.0",
) -> ValidationReport:
    if not warnings:
        raise SchemaViolationError(
            "warning_report requires at least one warning",
            field_path="warnings",
        )
    return ValidationReport(
        status=ValidationStatus.WARNING,
        is_valid=True,
        errors=[],
        warnings=warnings[:_MAX_WARNING_ITEMS],
        schema_version=schema_version,
    )


def make_error_item(
    *,
    code: str,
    message: str,
    field_path: str | None = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
) -> ValidationErrorItem:
    _validate_issue_text(code, field_path="code")
    _validate_issue_text(message, field_path="message")
    return ValidationErrorItem(
        code=code,
        message=message,
        field_path=field_path,
        severity=severity,
    )


def make_warning_item(
    *,
    code: str,
    message: str,
    field_path: str | None = None,
) -> ValidationWarningItem:
    _validate_issue_text(code, field_path="code")
    _validate_issue_text(message, field_path="message")
    return ValidationWarningItem(
        code=code,
        message=message,
        field_path=field_path,
    )


def validation_error_to_report(error: ValidationError) -> ValidationReport:
    error_items: list[ValidationErrorItem] = []
    pydantic_errors = error.errors()
    error_limit = (
        _MAX_ERROR_ITEMS - 1
        if len(pydantic_errors) > _MAX_ERROR_ITEMS
        else _MAX_ERROR_ITEMS
    )
    for item in pydantic_errors[:error_limit]:
        error_items.append(
            make_error_item(
                code="schema.validation_error",
                message=str(item.get("msg", "validation error")),
                field_path=_field_path_from_loc(tuple(item.get("loc", ()))),
            )
        )
    if len(pydantic_errors) > _MAX_ERROR_ITEMS:
        error_items.append(
            make_error_item(
                code="schema.validation_error.truncated",
                message="validation error list truncated",
                field_path=None,
            )
        )
    return failed_report(errors=error_items)


def validate_raw_observation(raw: RawObservation) -> ValidationReport:
    try:
        if not isinstance(raw, RawObservation):
            return _single_error_report(
                code="schema.invalid_type",
                message="raw must be RawObservation",
                field_path="raw",
            )
        _require_non_empty(raw.source_name, field_path="source_name")
        _require_non_empty(raw.adapter_name, field_path="adapter_name")
        _require_non_empty(raw.adapter_version, field_path="adapter_version")
        if raw.raw_payload is None:
            return _single_error_report(
                code="schema.required",
                message="raw_payload must not be None",
                field_path="raw_payload",
            )
        if raw.received_at is None:
            return _single_error_report(
                code="schema.required",
                message="received_at is required",
                field_path="received_at",
            )
        validate_source_boundary(
            InputSource(
                source_type=raw.source_type,
                source_name=raw.source_name,
                source_id=None,
                trust_level=raw.trust_level,
                metadata=raw.source_metadata,
            )
        )
        return passed_report()
    except ValidationError as error:
        return validation_error_to_report(error)
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_input_source(source: InputSource) -> ValidationReport:
    try:
        if not isinstance(source, InputSource):
            return _single_error_report(
                code="schema.invalid_type",
                message="source must be InputSource",
                field_path="source",
            )
        validate_source_boundary(source)
        return passed_report()
    except ValidationError as error:
        return validation_error_to_report(error)
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_payload_ref(payload_ref: PayloadRef) -> ValidationReport:
    try:
        if not isinstance(payload_ref, PayloadRef):
            return _single_error_report(
                code="schema.invalid_type",
                message="payload_ref must be PayloadRef",
                field_path="payload_ref",
            )
        if payload_ref.storage_kind in {
            PayloadStorageKind.REFERENCE,
            PayloadStorageKind.EXTERNAL,
        } and not payload_ref.uri:
            return _single_error_report(
                code="schema.required",
                message="uri is required for reference or external payloads",
                field_path="uri",
            )
        if payload_ref.size_bytes is not None and payload_ref.size_bytes < 0:
            return _single_error_report(
                code="schema.invalid_value",
                message="size_bytes must be greater than or equal to 0",
                field_path="size_bytes",
            )
        return passed_report()
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_canonical_content(content: CanonicalContent) -> ValidationReport:
    try:
        if not isinstance(content, CanonicalContent):
            return _single_error_report(
                code="schema.invalid_type",
                message="content must be CanonicalContent",
                field_path="content",
            )
        reports: list[ValidationReport] = []
        if content.kind == CanonicalContentKind.TEXT and content.text is None:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="text is required for TEXT content",
                    field_path="text",
                )
            )
        if content.kind in {
            CanonicalContentKind.JSON,
            CanonicalContentKind.EVENT,
        } and content.data is None:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="data is required for JSON or EVENT content",
                    field_path="data",
                )
            )
        if (
            content.kind == CanonicalContentKind.BINARY_REF
            and content.payload_ref is None
        ):
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="payload_ref is required for BINARY_REF content",
                    field_path="payload_ref",
                )
            )
        if content.kind == CanonicalContentKind.COMMAND and content.text is None:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="text is required for COMMAND content",
                    field_path="text",
                )
            )
        if (
            content.kind == CanonicalContentKind.ERROR
            and content.text is None
            and content.data is None
        ):
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="data or text is required for ERROR content",
                    field_path="content",
                )
            )
        if content.payload_ref is not None:
            reports.append(validate_payload_ref(content.payload_ref))
        return _merge_reports(reports)
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_observation_metadata(metadata: ObservationMetadata) -> ValidationReport:
    try:
        if not isinstance(metadata, ObservationMetadata):
            return _single_error_report(
                code="schema.invalid_type",
                message="metadata must be ObservationMetadata",
                field_path="metadata",
            )
        if metadata.length_chars is not None and metadata.length_chars < 0:
            return _single_error_report(
                code="schema.invalid_value",
                message="length_chars must be greater than or equal to 0",
                field_path="length_chars",
            )
        if metadata.size_bytes is not None and metadata.size_bytes < 0:
            return _single_error_report(
                code="schema.invalid_value",
                message="size_bytes must be greater than or equal to 0",
                field_path="size_bytes",
            )
        if not metadata.encoding:
            return _single_error_report(
                code="schema.required",
                message="encoding must not be empty",
                field_path="encoding",
            )
        return passed_report()
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_security_report(security_report: SecurityReport) -> ValidationReport:
    try:
        if not isinstance(security_report, SecurityReport):
            return _single_error_report(
                code="schema.invalid_type",
                message="security_report must be SecurityReport",
                field_path="security_report",
            )
        if (
            security_report.status == SecurityStatus.BLOCKED
            and not security_report.blocked_reason
        ):
            return _single_error_report(
                code="schema.required",
                message="blocked_reason is required when status is BLOCKED",
                field_path="blocked_reason",
            )
        return passed_report()
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_input_envelope(envelope: InputEnvelope) -> ValidationReport:
    try:
        if not isinstance(envelope, InputEnvelope):
            return _single_error_report(
                code="schema.invalid_type",
                message="envelope must be InputEnvelope",
                field_path="envelope",
            )
        reports: list[ValidationReport] = []
        if envelope.payload_ref is None and envelope.inline_payload is None:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="payload_ref or inline_payload is required",
                    field_path="payload",
                )
            )
        reports.append(validate_input_source(envelope.source))
        if envelope.payload_ref is not None:
            reports.append(validate_payload_ref(envelope.payload_ref))
        if envelope.received_at is None:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="received_at is required",
                    field_path="received_at",
                )
            )
        return _merge_reports(reports)
    except SchemaViolationError as error:
        return _error_to_report(error)


def validate_canonical_observation(
    observation: CanonicalObservation,
) -> ValidationReport:
    try:
        if not isinstance(observation, CanonicalObservation):
            return _single_error_report(
                code="schema.invalid_type",
                message="observation must be CanonicalObservation",
                field_path="observation",
            )

        reports = [
            validate_security_report(observation.security_report),
            validate_canonical_content(observation.content),
            validate_observation_metadata(observation.metadata),
        ]
        if not observation.validation_report.is_valid:
            reports.append(
                _single_error_report(
                    code="schema.invalid_value",
                    message="validation_report.is_valid must be True",
                    field_path="validation_report.is_valid",
                )
            )
        if not observation.content_hash:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="content_hash must not be empty",
                    field_path="content_hash",
                )
            )
        if not observation.structural_hash:
            reports.append(
                _single_error_report(
                    code="schema.required",
                    message="structural_hash must not be empty",
                    field_path="structural_hash",
                )
            )
        if not _is_observation_content_compatible(observation):
            reports.append(
                _single_error_report(
                    code="schema.incompatible_content",
                    message="observation_type is not compatible with content.kind",
                    field_path="content.kind",
                )
            )
        return _merge_reports(reports)
    except SchemaViolationError as error:
        return _error_to_report(error)


def require_valid_raw_observation(raw: RawObservation) -> None:
    report = validate_raw_observation(raw)
    if not report.is_valid:
        raise ValidationFailedError(
            "raw observation validation failed",
            details={"report": report.model_dump(mode="json")},
        )


def require_valid_input_envelope(envelope: InputEnvelope) -> None:
    report = validate_input_envelope(envelope)
    if not report.is_valid:
        raise ValidationFailedError(
            "input envelope validation failed",
            details={"report": report.model_dump(mode="json")},
        )


def require_valid_canonical_observation(observation: CanonicalObservation) -> None:
    report = validate_canonical_observation(observation)
    if not report.is_valid:
        raise ValidationFailedError(
            "canonical observation validation failed",
            details={"report": report.model_dump(mode="json")},
        )


def _single_error_report(
    *,
    code: str,
    message: str,
    field_path: str | None = None,
) -> ValidationReport:
    return failed_report(
        errors=[
            make_error_item(
                code=code,
                message=message,
                field_path=field_path,
            )
        ]
    )


def _merge_reports(reports: list[ValidationReport]) -> ValidationReport:
    errors: list[ValidationErrorItem] = []
    warnings: list[ValidationWarningItem] = []
    for report in reports:
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    errors = errors[:_MAX_ERROR_ITEMS]
    warnings = warnings[:_MAX_WARNING_ITEMS]
    if errors:
        return failed_report(errors=errors, warnings=warnings)
    if warnings:
        return warning_report(warnings=warnings)
    return passed_report()


def _field_path_from_loc(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _validate_issue_text(value: str, *, field_path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaViolationError(
            "validation issue text must be a non-empty string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )


def _require_non_empty(value: str, *, field_path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaViolationError(
            "field must be a non-empty string",
            field_path=field_path,
            details={"type": type(value).__name__},
        )


def _error_to_report(error: SchemaViolationError) -> ValidationReport:
    return _single_error_report(
        code=error.code,
        message=error.payload.message,
        field_path=error.field_path,
    )


def _is_observation_content_compatible(observation: CanonicalObservation) -> bool:
    compatible_content_kinds = {
        ObservationType.TEXT_MESSAGE: {CanonicalContentKind.TEXT},
        ObservationType.FILE_TEXT_EXTRACT: {CanonicalContentKind.TEXT},
        ObservationType.COMMAND_INPUT: {
            CanonicalContentKind.COMMAND,
            CanonicalContentKind.TEXT,
        },
        ObservationType.COMMAND_OUTPUT: {
            CanonicalContentKind.COMMAND,
            CanonicalContentKind.TEXT,
        },
        ObservationType.FILE_UPLOAD: {CanonicalContentKind.BINARY_REF},
        ObservationType.API_EVENT: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
        },
        ObservationType.WEBHOOK_EVENT: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
        },
        ObservationType.SYSTEM_EVENT: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
        },
        ObservationType.LOG_EVENT: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
        },
        ObservationType.DATABASE_ROW_CHANGE: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
        },
        ObservationType.USER_ACTION: {
            CanonicalContentKind.EVENT,
            CanonicalContentKind.JSON,
            CanonicalContentKind.TEXT,
        },
        ObservationType.ERROR_EVENT: {CanonicalContentKind.ERROR},
    }
    return (
        observation.content.kind
        in compatible_content_kinds[observation.observation_type]
    )


__all__ = [
    "failed_report",
    "make_error_item",
    "make_warning_item",
    "passed_report",
    "require_valid_canonical_observation",
    "require_valid_input_envelope",
    "require_valid_raw_observation",
    "validate_canonical_content",
    "validate_canonical_observation",
    "validate_input_envelope",
    "validate_input_source",
    "validate_observation_metadata",
    "validate_payload_ref",
    "validate_raw_observation",
    "validate_security_report",
    "validation_error_to_report",
    "warning_report",
]
