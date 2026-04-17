//! The chapter's compiled argument: R1CS, Spartan's outer/inner sumchecks,
//! preprocessed sparse-matrix commitments, the fold-and-commit PCS, and a SHA-256
//! Fiat--Shamir transcript.
//!
//! The assignment uses Spartan's half-split layout: private variables occupy the
//! low half of the Boolean cube and `(1, public_inputs)` the high half. Only the
//! private half is committed; verification reconstructs the public half's MLE.
//! ZK mode adds the chapter's additive sumcheck masks and commits to
//! `ŵ = w̃_lo + B`, opening `ŵ` and `B` through a combined fold chain.
use ark_ff::{FftField, PrimeField};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystem, Matrix};
use ark_serialize::CanonicalSerialize;
use ark_std::{rand::Rng, UniformRand};
use rayon::prelude::*;

use crate::{
    mask::{pack_mask_coeffs, prove_mask_eval, verify_mask_eval, MaskEvalProof},
    matrix_eval::{
        encode_matrix, prove_matrix_eval, verify_matrix_eval, MatrixCommitments, MatrixEncoding,
        MatrixEvalProof,
    },
    merkle::num_queries,
    pcs::{
        commit, opened_symbols_bound, prove_eval, prove_eval_combined, verify_eval,
        verify_eval_combined, CombinedProof, Commitment, PcsError, Proof,
    },
    piop::{ell_for, piop_prove, piop_verify, PiopProof},
    r1cs::ConstraintMatrices,
    sumcheck::sample_additive_mask,
    transcript::Transcript,
};

const OUTER_MASK_DEG: usize = 3;
const INNER_MASK_DEG: usize = 2;

/// Prover parameters: the R1CS matrices and the PCS-committed sparse encodings.
pub struct ProverParams<F: PrimeField> {
    pub matrices: ConstraintMatrices<F>,
    pub enc_a: MatrixEncoding<F>,
    pub enc_b: MatrixEncoding<F>,
    pub enc_c: MatrixEncoding<F>,
    pub ell_row: usize,
    pub ell_col: usize,
    /// Whether the witness cube includes the randomized-encoding padding used in ZK mode.
    pub zk: bool,
}

/// Verifier parameters: sparse-encoding roots and R1CS dimensions.
pub struct VerifierParams {
    pub comm_a: MatrixCommitments,
    pub comm_b: MatrixCommitments,
    pub comm_c: MatrixCommitments,
    pub ell_row: usize,
    pub ell_col: usize,
    pub num_constraints: usize,
    pub num_vars: usize,
    pub num_instance_variables: usize,
}

#[derive(CanonicalSerialize)]
pub struct SNARKProof<F: PrimeField> {
    /// PCS commitment to the witness half (ŵ = w̃_lo + B in ZK mode, w̃_lo otherwise).
    pub w_commitment: Commitment,
    /// PIOP transcript (outer + inner sumcheck).
    pub piop_proof: PiopProof<F>,
    /// Succinct evaluation proofs certifying Ã,B̃,C̃ at (x*,y*) against the
    /// preprocessing commitments (A, B, C in order).
    pub matrix_eval_proofs: (MatrixEvalProof<F>, MatrixEvalProof<F>, MatrixEvalProof<F>),
    /// Non-ZK: PCS evaluation proof for w̃_lo(y′). `None` in ZK mode (see `w_combined_proof`).
    pub w_proof: Option<Proof<F>>,
    /// ZK: PCS commitment to the blinding polynomial B (the second codeword of the
    /// batched opening).
    pub blinding_commitment: Option<Commitment>,
    /// ZK: combined opening of `(ŵ, B)` at `y′`.
    pub w_combined_proof: Option<CombinedProof<F>>,
    /// ZK: hiding PCS commitment to the packed coefficients of the outer additive
    /// mask Z_out(x) = Σ_i Σ_{j≤3} c[i][j]·x_i^j.
    pub z_out_mask_commitment: Option<Commitment>,
    /// ZK: hiding PCS commitment to the packed coefficients of the inner mask Z_in.
    pub z_in_mask_commitment: Option<Commitment>,
    /// ZK: inner-product proof binding Z_out(x*) to the committed coefficients.
    pub z_out_mask_proof: Option<MaskEvalProof<F>>,
    /// ZK: inner-product proof binding Z_in(y*) to the committed coefficients.
    pub z_in_mask_proof: Option<MaskEvalProof<F>>,
}

fn matrices_of<F: PrimeField, C: ConstraintSynthesizer<F>>(circuit: C) -> ConstraintMatrices<F> {
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    let inner = cs.borrow().unwrap();
    let mut raw = cs.to_matrices().unwrap();
    let mut abc = raw.remove("R1CS").expect("no R1CS predicate");
    ConstraintMatrices {
        a: abc.remove(0),
        b: abc.remove(0),
        c: abc.remove(0),
        num_instance_variables: inner.num_instance_variables,
        num_witness_variables: inner.num_witness_variables,
        num_constraints: inner.get_predicate_num_constraints("R1CS").unwrap_or(0),
    }
}

/// Remap each matrix column to the Spartan half-split layout (see `setup`):
/// witness columns `[num_instance, ·)` move to the low half `c − num_instance`, and
/// public columns `[0, num_instance)` move to the high half `half + c`. The relation
/// is invariant under this permutation of variables, so `A,B,C` and the assignment
/// `Z` are simply relabeled consistently.
fn remap_columns<F: PrimeField>(m: &mut Matrix<F>, num_instance: usize, half: usize) {
    for row in m.iter_mut() {
        for entry in row.iter_mut() {
            let c = entry.1;
            entry.1 = if c < num_instance { half + c } else { c - num_instance };
        }
    }
}

/// Synthesize the circuit and commit the public sparse encodings of `A`, `B`, and `C`.
///
/// Public-input binding (Spartan layout). The assignment vector `Z` is laid out on
/// the Boolean cube so that the private witness occupies the *low* half and the
/// public block `(1, io)` the *high* half: with `ell_col = s + 1`, column `c` is
/// remapped to
///   - witness `c ∈ [ninst, nvars)`  ↦  `c − ninst`            (low half `[0, 2^s)`),
///   - public  `c ∈ [0, ninst)`      ↦  `2^s + c`              (high half).
/// The prover then commits *only* the witness half; the verifier reconstructs the
/// public half's MLE from the stated `public_inputs` itself (see `verify`). This is
/// what ties a proof to its public inputs: the committed polynomial has no public
/// slots to forge.
///
/// In ZK mode the witness half is enlarged for the randomized padding used at
/// base-layer openings; `setup(circuit, true)` is therefore required for ZK proofs.
pub fn setup<F, C>(circuit: C, zk: bool) -> (ProverParams<F>, VerifierParams)
where
    F: PrimeField + FftField + UniformRand,
    C: ConstraintSynthesizer<F>,
{
    let mut matrices = matrices_of(circuit);
    let num_instance = matrices.num_instance_variables;
    let num_witness = matrices.num_witness_variables;

    let mut s = ell_for(num_witness.max(1)).max(ell_for(num_instance));
    if zk {
        s = s.max(ell_for(num_witness.max(1)) + 1);
        let two_t = 2 * num_queries();
        while (1usize << (s - 1)) + (1usize << s) < 2 * (opened_symbols_bound(s) + two_t) {
            s += 1;
        }
    }
    let ell_col = s + 1;
    let half = 1usize << s;
    remap_columns(&mut matrices.a, num_instance, half);
    remap_columns(&mut matrices.b, num_instance, half);
    remap_columns(&mut matrices.c, num_instance, half);

    let ell_row = ell_for(matrices.num_constraints);

    let mut encs: Vec<MatrixEncoding<F>> = [&matrices.a, &matrices.b, &matrices.c]
        .into_par_iter()
        .map(|m| encode_matrix(m, ell_row, ell_col))
        .collect();
    let enc_c = encs.pop().unwrap();
    let enc_b = encs.pop().unwrap();
    let enc_a = encs.pop().unwrap();

    let vp = VerifierParams {
        comm_a: enc_a.commitments.clone(),
        comm_b: enc_b.commitments.clone(),
        comm_c: enc_c.commitments.clone(),
        ell_row,
        ell_col,
        num_constraints: matrices.num_constraints,
        num_vars: num_instance + num_witness,
        num_instance_variables: matrices.num_instance_variables,
    };
    let pp = ProverParams { matrices, enc_a, enc_b, enc_c, ell_row, ell_col, zk };
    (pp, vp)
}

pub fn prove<F, C, R>(
    pp: &ProverParams<F>,
    circuit: C,
    zk: bool,
    rng: &mut R,
) -> Result<SNARKProof<F>, PcsError>
where
    F: PrimeField + FftField + UniformRand,
    C: ConstraintSynthesizer<F>,
    R: Rng,
{
    assert!(
        !zk || pp.zk,
        "ZK proving requires ZK parameters: call setup(circuit, true); \
         without the witness-cube floor and random padding the opening leaks the witness"
    );

    let mut transcript = Transcript::new(b"zk-snark");

    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    let (instance, witness) = {
        let inner = cs.borrow().unwrap();
        (
            inner.assignments.instance_assignment.clone(),
            inner.assignments.witness_assignment.clone(),
        )
    };
    let matrices = &pp.matrices;

    for &v in &instance[1..] {
        transcript.absorb_field(v);
    }

    pp.enc_a.commitments.absorb_into(&mut transcript);
    pp.enc_b.commitments.absorb_into(&mut transcript);
    pp.enc_c.commitments.absorb_into(&mut transcript);

    // Build the half-split assignment used by the chapter's Spartan reduction.
    let ell_col = pp.ell_col;
    let s = ell_col - 1;
    let half = 1usize << s;
    let num_instance = instance.len();

    let mut w_lo = witness;
    let num_witness = w_lo.len();
    w_lo.resize(half, F::zero());
    if zk {
        // These slots have no matrix entries and randomize the combined opening's base layer.
        for slot in w_lo[num_witness..].iter_mut() {
            *slot = F::rand(rng);
        }
    }

    let mut w_pad = w_lo.clone();
    w_pad.resize(1usize << ell_col, F::zero());
    for (c, &v) in instance.iter().enumerate().take(num_instance) {
        w_pad[half + c] = v;
    }

    // In ZK mode commit to `ŵ = w̃_lo + B` and `B` over the witness half.
    let (w_commit, blinding_commitment, blinding_witness_opt) = if zk {
        let b_vec: Vec<F> = (0..half).map(|_| F::rand(rng)).collect();
        let w_hat = w_lo.iter().zip(b_vec.iter()).map(|(&w, &b)| w + b).collect();
        let (b_commit, b_wit) = commit(b_vec, zk, rng);
        (w_hat, Some(b_commit), Some(b_wit))
    } else {
        (w_lo.clone(), None, None)
    };

    let (w_commitment, w_witness) = commit(w_commit, zk, rng);
    transcript.absorb(&w_commitment.root);
    if let Some(ref bc) = blinding_commitment {
        transcript.absorb(&bc.root);
    }

    // Mask roots are bound before the PIOP derives any Fiat--Shamir challenge.
    let ell_row = pp.ell_row;
    let (
        z_out_mask_commitment,
        z_in_mask_commitment,
        z_out_coeffs_opt,
        z_in_coeffs_opt,
        z_out_mask_wit_opt,
        z_in_mask_wit_opt,
        z_out_packed_opt,
        z_in_packed_opt,
    ) = if zk {
        let z_out_coeffs = sample_additive_mask::<F, R>(ell_row, OUTER_MASK_DEG, rng);
        let z_in_coeffs = sample_additive_mask::<F, R>(ell_col, INNER_MASK_DEG, rng);
        let z_out_packed = pack_mask_coeffs(&z_out_coeffs, OUTER_MASK_DEG, rng);
        let z_in_packed = pack_mask_coeffs(&z_in_coeffs, INNER_MASK_DEG, rng);

        let (zo_comm, zo_wit) = commit(z_out_packed.clone(), zk, rng);
        let (zi_comm, zi_wit) = commit(z_in_packed.clone(), zk, rng);
        transcript.absorb(&zo_comm.root);
        transcript.absorb(&zi_comm.root);

        (
            Some(zo_comm),
            Some(zi_comm),
            Some(z_out_coeffs),
            Some(z_in_coeffs),
            Some(zo_wit),
            Some(zi_wit),
            Some(z_out_packed),
            Some(z_in_packed),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    let (piop_proof, x_star, y_star, _u) = piop_prove(
        matrices, &w_pad, ell_row, ell_col, &mut transcript, zk, z_out_coeffs_opt, z_in_coeffs_opt, rng,
    );

    let y_prime = &y_star[..s];

    let (v_a, mep_a) = prove_matrix_eval(&pp.enc_a, &x_star, &y_star, &mut transcript)?;
    let (v_b, mep_b) = prove_matrix_eval(&pp.enc_b, &x_star, &y_star, &mut transcript)?;
    let (v_c, mep_c) = prove_matrix_eval(&pp.enc_c, &x_star, &y_star, &mut transcript)?;
    debug_assert_eq!(v_a, piop_proof.a_eval, "A matrix-eval claim mismatch");
    debug_assert_eq!(v_b, piop_proof.b_eval, "B matrix-eval claim mismatch");
    debug_assert_eq!(v_c, piop_proof.c_eval, "C matrix-eval claim mismatch");

    let (z_out_mask_proof, z_in_mask_proof) =
        match (z_out_mask_wit_opt, z_out_packed_opt, z_in_mask_wit_opt, z_in_packed_opt) {
            (Some(zo_wit), Some(zo_packed), Some(zi_wit), Some(zi_packed)) => {
                let (zo_v, zo_pf) =
                    prove_mask_eval(&zo_wit, &zo_packed, &x_star, OUTER_MASK_DEG, &mut transcript, zk, rng)?;
                let (zi_v, zi_pf) =
                    prove_mask_eval(&zi_wit, &zi_packed, &y_star, INNER_MASK_DEG, &mut transcript, zk, rng)?;
                debug_assert_eq!(Some(zo_v), piop_proof.z_out_eval, "Z_out(x*) mismatch");
                debug_assert_eq!(Some(zi_v), piop_proof.z_in_eval, "Z_in(y*) mismatch");
                (Some(zo_pf), Some(zi_pf))
            }
            _ => (None, None),
        };

    // ZK uses the combined `(ŵ, B)` fold chain; non-ZK opens `w̃_lo` directly.
    let (w_proof, w_combined_proof) = match blinding_witness_opt {
        Some(b_wit) => {
            let (_e0, _e1, combined) =
                prove_eval_combined(&w_witness, &b_wit, y_prime, &mut transcript, rng)?;
            (None, Some(combined))
        }
        None => {
            let (_v, p) = prove_eval(&w_witness, y_prime, &mut transcript, zk, rng)?;
            (Some(p), None)
        }
    };

    Ok(SNARKProof {
        w_commitment,
        piop_proof,
        matrix_eval_proofs: (mep_a, mep_b, mep_c),
        w_proof,
        blinding_commitment,
        w_combined_proof,
        z_out_mask_commitment,
        z_in_mask_commitment,
        z_out_mask_proof,
        z_in_mask_proof,
    })
}

/// `public_inputs` is the instance-assignment vector `[1, v_1, …]` (index 0 is the
/// constant one); only `public_inputs[1..]` are bound to the transcript.
pub fn verify<F>(
    vp: &VerifierParams,
    public_inputs: &[F],
    proof: &SNARKProof<F>,
) -> Result<bool, PcsError>
where
    F: PrimeField + FftField,
{
    let mut transcript = Transcript::new(b"zk-snark");

    if public_inputs.len() != vp.num_instance_variables {
        return Ok(false);
    }
    if public_inputs[0] != F::one() {
        return Ok(false);
    }

    // Every ZK component must be present together or absent together.
    let is_zk = proof.blinding_commitment.is_some();
    let zk_fields = [
        proof.w_combined_proof.is_some(),
        proof.z_out_mask_commitment.is_some(),
        proof.z_in_mask_commitment.is_some(),
        proof.z_out_mask_proof.is_some(),
        proof.z_in_mask_proof.is_some(),
        proof.piop_proof.z_out_eval.is_some(),
        proof.piop_proof.z_in_eval.is_some(),
        proof.piop_proof.z_out_sum.is_some(),
        proof.piop_proof.z_in_sum.is_some(),
    ];
    if zk_fields.iter().any(|&present| present != is_zk) {
        return Ok(false);
    }
    if proof.w_proof.is_some() == is_zk {
        return Ok(false);
    }

    for &v in public_inputs.iter().skip(1) {
        transcript.absorb_field(v);
    }

    vp.comm_a.absorb_into(&mut transcript);
    vp.comm_b.absorb_into(&mut transcript);
    vp.comm_c.absorb_into(&mut transcript);

    transcript.absorb(&proof.w_commitment.root);
    if let Some(ref bc) = proof.blinding_commitment {
        transcript.absorb(&bc.root);
    }
    if let Some(ref zo_c) = proof.z_out_mask_commitment {
        transcript.absorb(&zo_c.root);
    }
    if let Some(ref zi_c) = proof.z_in_mask_commitment {
        transcript.absorb(&zi_c.root);
    }

    let Some((x_star, y_star, u)) =
        piop_verify(vp.ell_row, vp.ell_col, &proof.piop_proof, &mut transcript)
    else {
        return Ok(false);
    };

    let (ref mep_a, ref mep_b, ref mep_c) = proof.matrix_eval_proofs;
    if !verify_matrix_eval(&vp.comm_a, &x_star, &y_star, proof.piop_proof.a_eval, mep_a, &mut transcript)? {
        return Ok(false);
    }
    if !verify_matrix_eval(&vp.comm_b, &x_star, &y_star, proof.piop_proof.b_eval, mep_b, &mut transcript)? {
        return Ok(false);
    }
    if !verify_matrix_eval(&vp.comm_c, &x_star, &y_star, proof.piop_proof.c_eval, mep_c, &mut transcript)? {
        return Ok(false);
    }

    // In the half-split layout the full assignment satisfies
    //      Z̃(y*) = (1 − y_t)·w̃_lo(y′) + y_t·ĩo(y′),
    // where the verifier computes `ĩo(y′)` from the stated public inputs.
    let s = vp.ell_col - 1;
    let y_prime = &y_star[..s];
    let y_t = y_star[s];

    let io_eval: F = public_inputs
        .iter()
        .enumerate()
        .map(|(c, &v)| {
            let eq: F = y_prime
                .iter()
                .enumerate()
                .map(|(j, &yj)| if (c >> j) & 1 == 1 { yj } else { F::one() - yj })
                .product();
            eq * v
        })
        .sum();

    if is_zk {
        let zo_c = proof.z_out_mask_commitment.as_ref().unwrap();
        let zo_p = proof.z_out_mask_proof.as_ref().unwrap();
        let zi_c = proof.z_in_mask_commitment.as_ref().unwrap();
        let zi_p = proof.z_in_mask_proof.as_ref().unwrap();
        let z_out_claim = proof.piop_proof.z_out_eval.unwrap();
        let z_in_claim = proof.piop_proof.z_in_eval.unwrap();
        if !verify_mask_eval(zo_c, &x_star, OUTER_MASK_DEG, z_out_claim, zo_p, &mut transcript)? {
            return Ok(false);
        }
        if !verify_mask_eval(zi_c, &y_star, INNER_MASK_DEG, z_in_claim, zi_p, &mut transcript)? {
            return Ok(false);
        }

        let combined = proof.w_combined_proof.as_ref().unwrap();
        let bc = proof.blinding_commitment.as_ref().unwrap();
        if !verify_eval_combined(&proof.w_commitment, bc, y_prime, combined, &mut transcript)? {
            return Ok(false);
        }
        let witness_eval = combined.eval0 - combined.eval1;
        let reconstructed = (F::one() - y_t) * witness_eval + y_t * io_eval;
        Ok(reconstructed == u)
    } else {
        let w_proof = proof.w_proof.as_ref().unwrap();
        let witness_eval = w_proof.final_value;
        if !verify_eval(&proof.w_commitment, y_prime, witness_eval, w_proof, &mut transcript)? {
            return Ok(false);
        }
        let reconstructed = (F::one() - y_t) * witness_eval + y_t * io_eval;
        Ok(reconstructed == u)
    }
}
