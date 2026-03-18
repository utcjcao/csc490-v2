from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalVerificationRequest:
    run_id: str
    model_storage_uri: str
    model_sha256: str
    input_region: str
    output_constraint: str
    verifier_name: str
    timeout_seconds: int
    memory_mb: int


@dataclass(slots=True)
class CanonicalVerificationMetrics:
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    reused_artifact_count: int = 0
    recomputed_step_count: int = 0


@dataclass(slots=True)
class CanonicalVerificationFailure:
    code: str
    message: str


@dataclass(slots=True)
class CanonicalVerificationResponse:
    run_id: str
    status: str
    outcome: str
    verifier_name: str
    summary: str
    metrics: CanonicalVerificationMetrics = field(default_factory=CanonicalVerificationMetrics)
    failure: CanonicalVerificationFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelReference:
    model_id: str
    storage_uri: str
    sha256: str


@dataclass(slots=True)
class PropertyPayload:
    property_id: str
    property_type: str
    input_region: str
    output_constraint: str
    normalization: str | None = None


@dataclass(slots=True)
class ArtifactReference:
    artifact_id: str
    artifact_type: str
    storage_uri: str
    artifact_hash: str


@dataclass(slots=True)
class BaselineContext:
    run_id: str
    artifact_refs: list[ArtifactReference] = field(default_factory=list)


@dataclass(slots=True)
class RecomputeStepPayload:
    name: str
    reason: str


@dataclass(slots=True)
class ReusePlanPayload:
    reuse_plan_id: str
    selected_artifact_ids: list[str] = field(default_factory=list)
    recompute_steps: list[RecomputeStepPayload] = field(default_factory=list)


@dataclass(slots=True)
class VerifierProfilePayload:
    verifier_profile_id: str
    name: str
    version: str


@dataclass(slots=True)
class ExecutionLimits:
    timeout_seconds: int
    memory_mb: int


@dataclass(slots=True)
class RunManifest:
    run_id: str
    mode: str
    model: ModelReference
    property: PropertyPayload
    verifier_profile: VerifierProfilePayload
    limits: ExecutionLimits
    baseline: BaselineContext | None = None
    reuse_plan: ReusePlanPayload | None = None


@dataclass(slots=True)
class RunMetricsPayload:
    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    reused_artifact_count: int = 0
    recomputed_step_count: int = 0


@dataclass(slots=True)
class ArtifactManifest:
    artifact_type: str
    storage_uri: str
    artifact_hash: str
    validity_scope: str


@dataclass(slots=True)
class CounterexamplePayload:
    input_blob_uri: str
    observed_output: str
    expected_constraint: str


@dataclass(slots=True)
class WorkerFailure:
    code: str
    message: str


@dataclass(slots=True)
class ResultManifest:
    run_id: str
    status: str
    outcome: str
    metrics: RunMetricsPayload = field(default_factory=RunMetricsPayload)
    artifacts: list[ArtifactManifest] = field(default_factory=list)
    counterexample: CounterexamplePayload | None = None
    failure: WorkerFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_request_from_dict(data: dict[str, Any]) -> CanonicalVerificationRequest:
    return CanonicalVerificationRequest(**data)


def run_manifest_from_dict(data: dict[str, Any]) -> RunManifest:
    baseline_data = data.get("baseline")
    reuse_plan_data = data.get("reuse_plan")

    baseline = None
    if baseline_data is not None:
        baseline = BaselineContext(
            run_id=baseline_data["run_id"],
            artifact_refs=[
                ArtifactReference(**artifact_ref)
                for artifact_ref in baseline_data.get("artifact_refs", [])
            ],
        )

    reuse_plan = None
    if reuse_plan_data is not None:
        reuse_plan = ReusePlanPayload(
            reuse_plan_id=reuse_plan_data["reuse_plan_id"],
            selected_artifact_ids=reuse_plan_data.get("selected_artifact_ids", []),
            recompute_steps=[
                RecomputeStepPayload(**step) for step in reuse_plan_data.get("recompute_steps", [])
            ],
        )

    return RunManifest(
        run_id=data["run_id"],
        mode=data["mode"],
        model=ModelReference(**data["model"]),
        property=PropertyPayload(**data["property"]),
        baseline=baseline,
        reuse_plan=reuse_plan,
        verifier_profile=VerifierProfilePayload(**data["verifier_profile"]),
        limits=ExecutionLimits(**data["limits"]),
    )
