use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable,
};
use ark_bls12_381::Fr as F;
use rand::{rngs::StdRng, SeedableRng};

use zk_snark::snark::{prove, verify};

// -----------------------------------------------------------------------
// Circuit 1: a + b + c = output  (one add constraint)
// -----------------------------------------------------------------------
struct AddThreeCircuit {
    output: F,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
}

impl ConstraintSynthesizer<F> for AddThreeCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        cs.enforce_constraint(
            ark_relations::lc!() + a + b + c,
            ark_relations::lc!() + Variable::One,
            ark_relations::lc!() + out,
        )
    }
}

// -----------------------------------------------------------------------
// Circuit 2: a * b = output  (one multiply constraint)
// -----------------------------------------------------------------------
struct MulCircuit {
    output: F,
    a: Option<F>,
    b: Option<F>,
}

impl ConstraintSynthesizer<F> for MulCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        cs.enforce_constraint(
            ark_relations::lc!() + a,
            ark_relations::lc!() + b,
            ark_relations::lc!() + out,
        )
    }
}

// -----------------------------------------------------------------------
// Circuit 3: t = a*b;  t + c + d = output  (two constraints)
// -----------------------------------------------------------------------
struct MulAddCircuit {
    output: F,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

impl ConstraintSynthesizer<F> for MulAddCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        let d = cs.new_witness_variable(|| Ok(self.d.unwrap_or_default()))?;
        let t = cs.new_witness_variable(|| {
            Ok(self.a.unwrap_or_default() * self.b.unwrap_or_default())
        })?;
        cs.enforce_constraint(
            ark_relations::lc!() + a,
            ark_relations::lc!() + b,
            ark_relations::lc!() + t,
        )?;
        cs.enforce_constraint(
            ark_relations::lc!() + t + c + d,
            ark_relations::lc!() + Variable::One,
            ark_relations::lc!() + out,
        )
    }
}

// -----------------------------------------------------------------------
// Circuit 4: t2 = c*d;  (a+b)*t2 = output  (two constraints)
// -----------------------------------------------------------------------
struct MulSumCircuit {
    output: F,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

impl ConstraintSynthesizer<F> for MulSumCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        let d = cs.new_witness_variable(|| Ok(self.d.unwrap_or_default()))?;
        let t2 = cs.new_witness_variable(|| {
            Ok(self.c.unwrap_or_default() * self.d.unwrap_or_default())
        })?;
        cs.enforce_constraint(
            ark_relations::lc!() + c,
            ark_relations::lc!() + d,
            ark_relations::lc!() + t2,
        )?;
        cs.enforce_constraint(
            ark_relations::lc!() + a + b,
            ark_relations::lc!() + t2,
            ark_relations::lc!() + out,
        )
    }
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// ---- Non-ZK: correct proof and soundness --------------------------------

#[test]
fn test_mult_correct_proof() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let proof = prove(MulCircuit { output, a: Some(a), b: Some(b) }, false, &mut rng())
        .expect("failed to prove");
    assert!(
        verify(MulCircuit { output, a: None, b: None }, &proof).expect("failed to verify"),
        "valid mult proof rejected"
    );
}

#[test]
fn test_additive_wrong_output_rejected() {
    let (a, b, c) = (F::from(3u64), F::from(5u64), F::from(7u64));
    let output = a + b + c;
    let proof = prove(AddThreeCircuit { output, a: Some(a), b: Some(b), c: Some(c) }, false, &mut rng())
        .expect("failed to prove");
    assert!(
        !verify(AddThreeCircuit { output: F::from(999u64), a: None, b: None, c: None }, &proof)
            .expect("failed to verify"),
        "wrong output accepted"
    );
}

// ---- ZK: correct proof and soundness ------------------------------------

#[test]
fn test_mult_zk_correct_proof() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let proof = prove(MulCircuit { output, a: Some(a), b: Some(b) }, true, &mut rng())
        .expect("failed to prove");
    assert!(
        verify(MulCircuit { output, a: None, b: None }, &proof).expect("failed to verify"),
        "valid ZK mult proof rejected"
    );
}

#[test]
fn test_mult_zk_wrong_output_rejected() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let proof = prove(MulCircuit { output, a: Some(a), b: Some(b) }, true, &mut rng())
        .expect("failed to prove");
    assert!(
        !verify(MulCircuit { output: F::from(41u64), a: None, b: None }, &proof)
            .expect("failed to verify"),
        "wrong ZK output accepted"
    );
}

// ---- Multi-constraint ZK circuits ---------------------------------------

#[test]
fn test_mixed_zk_correct_proof() {
    let (a, b, c, d) = (F::from(3u64), F::from(4u64), F::from(5u64), F::from(6u64));
    let output = a * b + c + d; // 23
    let proof = prove(
        MulAddCircuit { output, a: Some(a), b: Some(b), c: Some(c), d: Some(d) },
        true,
        &mut rng(),
    ).expect("failed to prove");
    assert!(
        verify(MulAddCircuit { output, a: None, b: None, c: None, d: None }, &proof)
            .expect("failed to verify"),
        "valid ZK mixed proof rejected"
    );
}

#[test]
fn test_deep_mixed_zk_correct_proof() {
    let (a, b, c, d) = (F::from(2u64), F::from(3u64), F::from(4u64), F::from(5u64));
    let output = (a + b) * (c * d); // 100
    let proof = prove(
        MulSumCircuit { output, a: Some(a), b: Some(b), c: Some(c), d: Some(d) },
        true,
        &mut rng(),
    ).expect("failed to prove");
    assert!(
        verify(MulSumCircuit { output, a: None, b: None, c: None, d: None }, &proof)
            .expect("failed to verify"),
        "valid ZK deep-mixed proof rejected"
    );
}
