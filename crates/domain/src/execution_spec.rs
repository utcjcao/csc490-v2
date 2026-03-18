use serde::{Deserialize, Serialize};

use crate::{
    DomainError, ModelFormat, ModelVersion, PropertySpec, PropertyType, VerificationRun,
    VerifierProfile,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub timeout_seconds: u64,
    pub memory_mb: u64,
}

impl ExecutionLimits {
    pub fn new(timeout_seconds: u64, memory_mb: u64) -> Result<Self, DomainError> {
        if timeout_seconds == 0 {
            return Err(DomainError::invariant_violation(
                "execution timeout must be greater than zero",
            ));
        }

        if memory_mb == 0 {
            return Err(DomainError::invariant_violation(
                "execution memory limit must be greater than zero",
            ));
        }

        Ok(Self { timeout_seconds, memory_mb })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LinfInputRegion {
    eps_micros: u64,
}

impl LinfInputRegion {
    pub fn new(eps_micros: u64) -> Result<Self, DomainError> {
        if eps_micros == 0 {
            return Err(DomainError::invariant_violation("linf epsilon must be greater than zero"));
        }

        Ok(Self { eps_micros })
    }

    pub fn eps_micros(&self) -> u64 {
        self.eps_micros
    }

    pub fn to_normalized_json(&self) -> String {
        format!(r#"{{"eps":{},"norm":"linf"}}"#, format_decimal_micros(self.eps_micros))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabelConstraint {
    label: u32,
}

impl LabelConstraint {
    pub fn new(label: u32) -> Result<Self, DomainError> {
        Ok(Self { label })
    }

    pub fn label(&self) -> u32 {
        self.label
    }

    pub fn to_normalized_json(&self) -> String {
        format!(r#"{{"label":{}}}"#, self.label)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedExecutionSpec {
    pub run: VerificationRun,
    pub model: ModelVersion,
    pub property: PropertySpec,
    pub verifier: VerifierProfile,
    pub limits: ExecutionLimits,
    pub input_region: LinfInputRegion,
    pub output_constraint: LabelConstraint,
}

impl NormalizedExecutionSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run: VerificationRun,
        model: ModelVersion,
        property: PropertySpec,
        verifier: VerifierProfile,
        limits: ExecutionLimits,
        input_region: LinfInputRegion,
        output_constraint: LabelConstraint,
    ) -> Result<Self, DomainError> {
        if run.model_id != model.id {
            return Err(DomainError::invariant_violation(
                "verification run model_id must match the normalized model",
            ));
        }

        if run.property_id != property.id {
            return Err(DomainError::invariant_violation(
                "verification run property_id must match the normalized property",
            ));
        }

        if run.verifier_profile_id != verifier.id {
            return Err(DomainError::invariant_violation(
                "verification run verifier_profile_id must match the verifier profile",
            ));
        }

        if property.project_id != run.project_id {
            return Err(DomainError::invariant_violation(
                "normalized property must belong to the verification run project",
            ));
        }

        if model.format != ModelFormat::Onnx {
            return Err(DomainError::invariant_violation(
                "normalized execution spec only supports ONNX models",
            ));
        }

        if property.property_type != PropertyType::LocalRobustnessLinf {
            return Err(DomainError::invariant_violation(
                "normalized execution spec only supports local linf robustness properties",
            ));
        }

        if !verifier.supported_formats.contains(&model.format) {
            return Err(DomainError::invariant_violation(
                "verifier profile does not support the normalized model format",
            ));
        }

        if !verifier.supported_property_types.contains(&property.property_type) {
            return Err(DomainError::invariant_violation(
                "verifier profile does not support the normalized property type",
            ));
        }

        Ok(Self { run, model, property, verifier, limits, input_region, output_constraint })
    }
}

fn format_decimal_micros(value: u64) -> String {
    let whole = value / 1_000_000;
    let fractional = value % 1_000_000;

    if fractional == 0 {
        return whole.to_string();
    }

    let mut rendered = format!("{whole}.{fractional:06}");
    while rendered.ends_with('0') {
        rendered.pop();
    }

    rendered
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{
        ExecutionLimits, LinfInputRegion, ModelFormat, ModelVersion, NormalizedExecutionSpec,
        PropertySpec, PropertyType, RunMetrics, RunStatus, TransformType, VerificationMode,
        VerificationRun, VerifierProfile,
    };

    use super::LabelConstraint;

    fn demo_model(project_id: Uuid) -> ModelVersion {
        ModelVersion::new(
            project_id,
            None,
            ModelFormat::Onnx,
            "demo-model-sha256".to_string(),
            "demo-architecture".to_string(),
            "demo-model-sha256".to_string(),
            TransformType::Root,
            Some("demo".to_string()),
            "demo://models/example.onnx".to_string(),
        )
        .expect("model should be valid")
    }

    fn demo_property(project_id: Uuid) -> PropertySpec {
        PropertySpec::new(
            project_id,
            PropertyType::LocalRobustnessLinf,
            r#"{"eps":0.01,"norm":"linf"}"#.to_string(),
            r#"{"label":1}"#.to_string(),
            None,
            "property:10000:1".to_string(),
        )
        .expect("property should be valid")
    }

    fn demo_verifier() -> VerifierProfile {
        VerifierProfile::new(
            "alpha-beta-crown".to_string(),
            "stub-0.1.0".to_string(),
            "python -m alpha_beta_crown_adapter".to_string(),
            vec![ModelFormat::Onnx],
            vec![PropertyType::LocalRobustnessLinf],
            vec!["stub-proof-summary".to_string()],
        )
        .expect("verifier should be valid")
    }

    #[test]
    fn rejects_zero_execution_timeout() {
        let error = ExecutionLimits::new(0, 512).expect_err("zero timeout should fail");
        assert_eq!(
            error.to_string(),
            "invariant violation: execution timeout must be greater than zero"
        );
    }

    #[test]
    fn rejects_zero_linf_epsilon() {
        let error = LinfInputRegion::new(0).expect_err("zero epsilon should fail");
        assert_eq!(
            error.to_string(),
            "invariant violation: linf epsilon must be greater than zero"
        );
    }

    #[test]
    fn rejects_inconsistent_normalized_execution_spec() {
        let project_id = Uuid::new_v4();
        let model = demo_model(project_id);
        let property = demo_property(project_id);
        let verifier = demo_verifier();
        let run = VerificationRun {
            id: Uuid::new_v4(),
            project_id,
            model_id: Uuid::new_v4(),
            property_id: property.id,
            verifier_profile_id: verifier.id,
            mode: VerificationMode::Full,
            status: RunStatus::Pending,
            outcome: None,
            started_at: None,
            ended_at: None,
            metrics: RunMetrics::default(),
            reuse_plan_id: None,
        };

        let error = NormalizedExecutionSpec::new(
            run,
            model,
            property,
            verifier,
            ExecutionLimits::new(60, 1024).expect("limits should be valid"),
            LinfInputRegion::new(10_000).expect("eps should be valid"),
            LabelConstraint::new(1).expect("label should be valid"),
        )
        .expect_err("inconsistent run should fail");

        assert_eq!(
            error.to_string(),
            "invariant violation: verification run model_id must match the normalized model"
        );
    }

    #[test]
    fn renders_normalized_json_payloads_deterministically() {
        let region = LinfInputRegion::new(10_000).expect("eps should be valid");
        let constraint = LabelConstraint::new(3).expect("label should be valid");

        assert_eq!(region.to_normalized_json(), r#"{"eps":0.01,"norm":"linf"}"#);
        assert_eq!(constraint.to_normalized_json(), r#"{"label":3}"#);
    }
}
