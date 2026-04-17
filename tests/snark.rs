use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable,
};
use ark_bls12_381::Fr as F;
use ark_ff::One;
use rand::{rngs::StdRng, SeedableRng};

use zk_snark::snark::{prove, setup, verify};

struct AddThreeCircuit {
    output: Option<F>,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
}

impl ConstraintSynthesizer<F> for AddThreeCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output.unwrap_or_default()))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a + b + c,
            || ark_relations::lc!() + Variable::One,
            || ark_relations::lc!() + out,
        )
    }
}

struct MulCircuit {
    output: Option<F>,
    a: Option<F>,
    b: Option<F>,
}

impl ConstraintSynthesizer<F> for MulCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output.unwrap_or_default()))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a,
            || ark_relations::lc!() + b,
            || ark_relations::lc!() + out,
        )
    }
}

struct MulAddCircuit {
    output: Option<F>,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

impl ConstraintSynthesizer<F> for MulAddCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output.unwrap_or_default()))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        let d = cs.new_witness_variable(|| Ok(self.d.unwrap_or_default()))?;
        let t = cs.new_witness_variable(|| {
            Ok(self.a.unwrap_or_default() * self.b.unwrap_or_default())
        })?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a,
            || ark_relations::lc!() + b,
            || ark_relations::lc!() + t,
        )?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + t + c + d,
            || ark_relations::lc!() + Variable::One,
            || ark_relations::lc!() + out,
        )
    }
}

struct MulSumCircuit {
    output: Option<F>,
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

impl ConstraintSynthesizer<F> for MulSumCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output.unwrap_or_default()))?;
        let a = cs.new_witness_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_witness_variable(|| Ok(self.b.unwrap_or_default()))?;
        let c = cs.new_witness_variable(|| Ok(self.c.unwrap_or_default()))?;
        let d = cs.new_witness_variable(|| Ok(self.d.unwrap_or_default()))?;
        let t2 = cs.new_witness_variable(|| {
            Ok(self.c.unwrap_or_default() * self.d.unwrap_or_default())
        })?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + c,
            || ark_relations::lc!() + d,
            || ark_relations::lc!() + t2,
        )?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a + b,
            || ark_relations::lc!() + t2,
            || ark_relations::lc!() + out,
        )
    }
}

struct PubMulCircuit {
    a: Option<F>,
    b: Option<F>,
    w: Option<F>,
}

impl ConstraintSynthesizer<F> for PubMulCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let a = cs.new_input_variable(|| Ok(self.a.unwrap_or_default()))?;
        let b = cs.new_input_variable(|| Ok(self.b.unwrap_or_default()))?;
        let w = cs.new_witness_variable(|| Ok(self.w.unwrap_or_default()))?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a,
            || ark_relations::lc!() + w,
            || ark_relations::lc!() + b,
        )
    }
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

#[test]
fn test_mult_correct_proof() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, false);
    let proof = prove(&pp, MulCircuit { output: Some(output), a: Some(a), b: Some(b) }, false, &mut rng())
        .expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(
        verify(&vp, &public, &proof).expect("failed to verify"),
        "valid mult proof rejected"
    );
}

#[test]
fn test_additive_wrong_output_rejected() {
    let (a, b, c) = (F::from(3u64), F::from(5u64), F::from(7u64));
    let output = a + b + c;
    let (pp, vp) = setup::<F, _>(AddThreeCircuit { output: None, a: None, b: None, c: None }, false);
    let proof = prove(&pp, AddThreeCircuit { output: Some(output), a: Some(a), b: Some(b), c: Some(c) }, false, &mut rng())
        .expect("failed to prove");
    let public_wrong = vec![F::one(), F::from(999u64)];
    assert!(
        !verify(&vp, &public_wrong, &proof).expect("failed to verify"),
        "wrong output accepted"
    );
}

#[test]
fn test_mult_zk_correct_proof() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let proof = prove(&pp, MulCircuit { output: Some(output), a: Some(a), b: Some(b) }, true, &mut rng())
        .expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(
        verify(&vp, &public, &proof).expect("failed to verify"),
        "valid ZK mult proof rejected"
    );
}

#[test]
fn test_mult_zk_wrong_output_rejected() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let proof = prove(&pp, MulCircuit { output: Some(output), a: Some(a), b: Some(b) }, true, &mut rng())
        .expect("failed to prove");
    let public_wrong = vec![F::one(), F::from(41u64)];
    assert!(
        !verify(&vp, &public_wrong, &proof).expect("failed to verify"),
        "wrong ZK output accepted"
    );
}

#[test]
fn test_zk_tampered_mask_sum_rejected() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let mut proof = prove(
        &pp,
        MulCircuit { output: Some(output), a: Some(a), b: Some(b) },
        true,
        &mut rng(),
    )
    .expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(verify(&vp, &public, &proof).expect("verify"), "valid proof rejected");

    let z_out_sum = proof.piop_proof.z_out_sum.unwrap();
    proof.piop_proof.z_out_sum = Some(z_out_sum + F::one());
    assert!(
        !verify(&vp, &public, &proof).expect("verify"),
        "tampered outer mask sum accepted"
    );

    proof.piop_proof.z_out_sum = Some(z_out_sum);
    proof.piop_proof.z_in_sum = Some(proof.piop_proof.z_in_sum.unwrap() + F::one());
    assert!(
        !verify(&vp, &public, &proof).expect("verify"),
        "tampered inner mask sum accepted"
    );
}

#[test]
fn test_zk_tampered_mask_eval_rejected() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let mut proof = prove(
        &pp,
        MulCircuit { output: Some(output), a: Some(a), b: Some(b) },
        true,
        &mut rng(),
    )
    .expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(verify(&vp, &public, &proof).expect("verify"), "valid proof rejected");

    proof.piop_proof.z_in_eval = Some(proof.piop_proof.z_in_eval.unwrap() + F::one());
    assert!(
        !verify(&vp, &public, &proof).expect("verify"),
        "tampered inner mask evaluation accepted"
    );
}

#[test]
fn test_mixed_zk_correct_proof() {
    let (a, b, c, d) = (F::from(3u64), F::from(4u64), F::from(5u64), F::from(6u64));
    let output = a * b + c + d;
    let (pp, vp) = setup::<F, _>(MulAddCircuit { output: None, a: None, b: None, c: None, d: None }, true);
    let proof = prove(
        &pp,
        MulAddCircuit { output: Some(output), a: Some(a), b: Some(b), c: Some(c), d: Some(d) },
        true,
        &mut rng(),
    ).expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(
        verify(&vp, &public, &proof).expect("failed to verify"),
        "valid ZK mixed proof rejected"
    );
}

#[test]
fn test_public_input_binding() {
    let (a, w) = (F::from(3u64), F::from(5u64));
    let b = a * w;
    let (pp, vp) = setup::<F, _>(PubMulCircuit { a: None, b: None, w: None }, false);
    let proof = prove(
        &pp,
        PubMulCircuit { a: Some(a), b: Some(b), w: Some(w) },
        false,
        &mut rng(),
    )
    .expect("failed to prove");

    assert!(
        verify(&vp, &[F::one(), a, b], &proof).expect("verify"),
        "valid multi-input proof rejected"
    );
    assert!(
        !verify(&vp, &[F::one(), a, b + F::one()], &proof).expect("verify"),
        "wrong public output b accepted"
    );
    assert!(
        !verify(&vp, &[F::one(), a + F::one(), b], &proof).expect("verify"),
        "wrong public input a accepted"
    );
    assert!(
        !verify(&vp, &[F::from(0u64), a, b], &proof).expect("verify"),
        "instance block with constant wire != 1 accepted"
    );
}

#[test]
fn test_zk_shape_mismatch_rejected() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let output = a * b;
    let (pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let mut proof = prove(
        &pp,
        MulCircuit { output: Some(output), a: Some(a), b: Some(b) },
        true,
        &mut rng(),
    )
    .expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(verify(&vp, &public, &proof).expect("verify"), "valid proof rejected");

    proof.blinding_commitment = None;
    proof.w_combined_proof = None;
    proof.z_out_mask_commitment = None;
    proof.z_in_mask_commitment = None;
    proof.z_out_mask_proof = None;
    proof.z_in_mask_proof = None;
    assert!(
        !verify(&vp, &public, &proof).expect("verify"),
        "malformed mixed ZK/non-ZK proof accepted"
    );
}

#[test]
#[should_panic(expected = "ZK proving requires ZK parameters")]
fn test_zk_prove_requires_zk_setup() {
    let (a, b) = (F::from(6u64), F::from(7u64));
    let (pp, _vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, false);
    let _ = prove(
        &pp,
        MulCircuit { output: Some(a * b), a: Some(a), b: Some(b) },
        true,
        &mut rng(),
    );
}

#[test]
fn test_zk_setup_floors_witness_cube() {
    use zk_snark::{merkle::num_queries, pcs::opened_symbols_bound};

    let (_pp, vp) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, true);
    let s = vp.ell_col - 1;
    let opened = opened_symbols_bound(s) + 2 * num_queries();
    assert!(
        (1usize << (s - 1)) + (1usize << s) >= 2 * opened,
        "witness cube 2^{s} is below the ZK leak budget ({opened} opened symbols)"
    );

    let (_pp2, vp2) = setup::<F, _>(MulCircuit { output: None, a: None, b: None }, false);
    assert!(vp2.ell_col < vp.ell_col, "non-ZK setup should not pay the ZK floor");
}

#[test]
fn test_deep_mixed_zk_correct_proof() {
    let (a, b, c, d) = (F::from(2u64), F::from(3u64), F::from(4u64), F::from(5u64));
    let output = (a + b) * (c * d);
    let (pp, vp) = setup::<F, _>(MulSumCircuit { output: None, a: None, b: None, c: None, d: None }, true);
    let proof = prove(
        &pp,
        MulSumCircuit { output: Some(output), a: Some(a), b: Some(b), c: Some(c), d: Some(d) },
        true,
        &mut rng(),
    ).expect("failed to prove");
    let public = vec![F::one(), output];
    assert!(
        verify(&vp, &public, &proof).expect("failed to verify"),
        "valid ZK deep-mixed proof rejected"
    );
}
