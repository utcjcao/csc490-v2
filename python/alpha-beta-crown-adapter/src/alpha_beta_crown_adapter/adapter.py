from __future__ import annotations

from adapter_common import (
    CanonicalVerificationFailure,
    CanonicalVerificationMetrics,
    CanonicalVerificationRequest,
    CanonicalVerificationResponse,
)


class AlphaBetaCrownAdapter:
    def run(self, request: CanonicalVerificationRequest) -> CanonicalVerificationResponse:
        if "adapter-error" in request.model_storage_uri:
            return CanonicalVerificationResponse(
                run_id=request.run_id,
                status="failed",
                outcome="error",
                verifier_name=request.verifier_name,
                summary="Stub adapter returned a deterministic error for the requested model URI.",
                metrics=CanonicalVerificationMetrics(),
                failure=CanonicalVerificationFailure(
                    code="stub_adapter_error",
                    message=(
                        "The Phase 1 stub intentionally returns an adapter error when "
                        "model_storage_uri contains 'adapter-error'."
                    ),
                ),
            )

        return CanonicalVerificationResponse(
            run_id=request.run_id,
            status="completed",
            outcome="proved",
            verifier_name=request.verifier_name,
            summary=(
                "Stub alpha-beta-CROWN run completed successfully for the canonical Phase 1 demo."
            ),
            metrics=CanonicalVerificationMetrics(
                wall_time_ms=25,
                cpu_time_ms=11,
                reused_artifact_count=0,
                recomputed_step_count=1,
            ),
            failure=None,
        )
