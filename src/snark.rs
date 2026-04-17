/// Non-interactive ZK-SNARK via the BCS transformation.
///
/// Three-phase compiler (book §Compiled SNARK):
///   1. Any circuit implementing `ConstraintSynthesizer<F>` is arithmetized as
///      R1CS by calling `generate_constraints`, producing `ConstraintMatrices<F>`
///      and a flat witness vector.
///   2. The witness MLE is committed with WHIR (fold-and-commit).
///   3. A two-phase PIOP (outer + inner sumcheck) proves the R1CS is satisfied.
///   4. Fiat-Shamir via SHA-256 transcript makes everything non-interactive.
///
/// Public inputs are bound to the transcript before any challenges are
/// squeezed, ensuring soundness: a proof for public input `v` cannot verify
/// under a different stated public input `v'`.
///
/// ZK mode: sumchecks masked with degree-matched zero-sum polynomials + witness MLE
/// blinded (commit to ŵ = w + b; reveal b = B̃(y*) for reconciliation).
///
/// Degree-matched masking: Z_out = ZE·(ZA·ZB − ZC) (degree 3 per variable) and
/// Z_in = Zα·Zβ (degree 2 per variable). Each factor is a multilinear oracle
/// committed and opened separately so the verifier can reconstruct z_out/z_in
/// evaluations while the prover is bound by the WHIR commitments.
use ark_ff::{FftField, PrimeField};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
use ark_std::{rand::Rng, UniformRand};

use crate::{
    pcs::{whir_commit, whir_prove_eval, whir_verify_eval, PcsError, WhirCommitment, WhirProof},
    piop::{ell_for, piop_prove, piop_verify, PiopProof},
    r1cs::mle_of_vector,
    sumcheck::{sample_degree2_zero_sum_masking, sample_degree3_zero_sum_masking},
    transcript::Transcript,
};

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

pub struct SNARKProof<F: PrimeField> {
    /// WHIR commitment to the (blinded) witness ŵ = w̃ + B.
    pub whir_commitment: WhirCommitment,
    /// PIOP transcript (outer + inner sumcheck).
    pub piop_proof: PiopProof<F>,
    /// WHIR evaluation proof for ŵ(y*).
    pub whir_proof: WhirProof<F>,
    /// ZK: claimed B(y*); verifier recovers w̃(y*) = ŵ(y*) − B(y*).
    pub blinding_eval: Option<F>,
    /// ZK: separate WHIR commitment to the blinding polynomial B.
    pub blinding_commitment: Option<WhirCommitment>,
    /// ZK: WHIR evaluation proof for B(y*), binding blinding_eval to the committed B.
    pub blinding_whir_proof: Option<WhirProof<F>>,
    /// ZK: WHIR commitments to the four Z_out factors (ZE, ZA, ZB, ZC).
    /// Z_out(x) = ZE(x)·(ZA(x)·ZB(x) − ZC(x)); each factor is a multilinear oracle.
    pub z_out_factor_commitments: Option<(WhirCommitment, WhirCommitment, WhirCommitment, WhirCommitment)>,
    /// ZK: WHIR commitments to the two Z_in factors (Zα, Zβ).
    /// Z_in(y) = Zα(y)·Zβ(y).
    pub z_in_factor_commitments: Option<(WhirCommitment, WhirCommitment)>,
    /// ZK: WHIR evaluation proofs for Z_out factors at x*.
    pub z_out_factor_proofs: Option<(WhirProof<F>, WhirProof<F>, WhirProof<F>, WhirProof<F>)>,
    /// ZK: WHIR evaluation proofs for Z_in factors at y*.
    pub z_in_factor_proofs: Option<(WhirProof<F>, WhirProof<F>)>,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove<F, C, R>(circuit: C, zk: bool, rng: &mut R) -> Result<SNARKProof<F>, PcsError>
where
    F: PrimeField + FftField + UniformRand,
    C: ConstraintSynthesizer<F>,
    R: Rng,
{
    let mut transcript = Transcript::new(b"zk-snark");

    // 1. Arithmetize the circuit as R1CS.
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();

    let matrices = cs.to_matrices().unwrap();
    let (instance, witness) = {
        let inner = cs.borrow().unwrap();
        (inner.instance_assignment.clone(), inner.witness_assignment.clone())
    };

    // 2. Bind public inputs to the Fiat-Shamir transcript (skip index 0 = constant-1).
    for &v in &instance[1..] {
        transcript.absorb_field(v);
    }

    // 3. Build the flat witness: [instance || witness] padded to 2^ell_col.
    let num_vars = matrices.num_instance_variables + matrices.num_witness_variables;
    let ell_col = ell_for(num_vars);
    let ell_row = ell_for(matrices.num_constraints);
    let padded_len = 1usize << ell_col;

    let mut w_pad: Vec<F> = instance;
    w_pad.extend_from_slice(&witness);
    w_pad.resize(padded_len, F::zero());

    // 4. ZK blinding: commit to ŵ = w̃ + B and to B separately.
    let (w_commit, blinding_commitment, blinding_whir_witness_opt, blinding_vec) = if zk {
        let b_vec: Vec<F> = (0..padded_len).map(|_| F::rand(rng)).collect();
        let w_hat = w_pad.iter().zip(b_vec.iter()).map(|(&w, &b)| w + b).collect();
        let (b_commit, b_wit) = whir_commit(b_vec.clone(), zk, rng);
        (w_hat, Some(b_commit), Some(b_wit), Some(b_vec))
    } else {
        (w_pad.clone(), None, None, None)
    };

    // 5. WHIR commitment to ŵ; absorb ŵ root and B root (if ZK).
    let (whir_commitment, mut whir_witness) = whir_commit(w_commit, zk, rng);
    transcript.absorb(&whir_commitment.root);
    if let Some(ref bc) = blinding_commitment {
        transcript.absorb(&bc.root);
    }

    // 5b. ZK: sample degree-matched masking factors, commit each as a multilinear
    //     oracle, and absorb all roots before any PIOP challenges are squeezed.
    //
    //     Z_out = ZE·(ZA·ZB − ZC): degree-3-per-variable outer masking.
    //     Z_in  = Zα·Zβ:           degree-2-per-variable inner masking.
    //
    //     Because each factor is multilinear, its WHIR-committed MLE at x* (or y*)
    //     equals its bookkeeping-folded value — ensuring consistency.
    let (
        z_out_factor_commitments_opt,
        z_in_factor_commitments_opt,
        z_out_wits_opt,
        z_in_wits_opt,
        z_out_factored_opt,
        z_in_factored_opt,
    ) = if zk {
        let (ze_o, za_o, zb_o, zc_o) = sample_degree3_zero_sum_masking::<F, R>(ell_row, rng);
        let (ze_comm, ze_wit) = whir_commit(ze_o.clone(), zk, rng);
        let (za_comm, za_wit) = whir_commit(za_o.clone(), zk, rng);
        let (zb_comm, zb_wit) = whir_commit(zb_o.clone(), zk, rng);
        let (zc_comm, zc_wit) = whir_commit(zc_o.clone(), zk, rng);
        transcript.absorb(&ze_comm.root);
        transcript.absorb(&za_comm.root);
        transcript.absorb(&zb_comm.root);
        transcript.absorb(&zc_comm.root);

        let (za_i, zb_i) = sample_degree2_zero_sum_masking::<F, R>(ell_col, rng);
        let (za_in_comm, za_in_wit) = whir_commit(za_i.clone(), zk, rng);
        let (zb_in_comm, zb_in_wit) = whir_commit(zb_i.clone(), zk, rng);
        transcript.absorb(&za_in_comm.root);
        transcript.absorb(&zb_in_comm.root);

        (
            Some((ze_comm, za_comm, zb_comm, zc_comm)),
            Some((za_in_comm, zb_in_comm)),
            Some((ze_wit, za_wit, zb_wit, zc_wit)),
            Some((za_in_wit, zb_in_wit)),
            Some((ze_o, za_o, zb_o, zc_o)),
            Some((za_i, zb_i)),
        )
    } else {
        (None, None, None, None, None, None)
    };

    // 6. PIOP on the true witness w_pad.
    let (piop_proof, x_star, y_star, _u) =
        piop_prove(&matrices, &w_pad, &mut transcript, zk, z_out_factored_opt, z_in_factored_opt, rng);

    // 7. ZK: absorb b = B̃(y*); generate WHIR eval proof for B(y*).
    let (blinding_eval, blinding_whir_proof) = match (blinding_vec, blinding_whir_witness_opt) {
        (Some(b_vec), Some(mut b_wit)) => {
            let b = mle_of_vector(&b_vec, ell_col, &y_star);
            transcript.absorb_field(b);
            let (_b_val, b_proof) = whir_prove_eval(&mut b_wit, &y_star, &mut transcript, zk, rng)?;
            (Some(b), Some(b_proof))
        }
        _ => (None, None),
    };

    // 7b. ZK: generate WHIR eval proofs for each Z_out factor at x* and each
    //     Z_in factor at y*. The verifier will reconstruct z_out_eval and
    //     z_in_eval from these openings, binding them to the committed factors.
    let (z_out_factor_proofs, z_in_factor_proofs) = match (z_out_wits_opt, z_in_wits_opt) {
        (Some((mut ze_wit, mut za_wit, mut zb_wit, mut zc_wit)),
         Some((mut za_in_wit, mut zb_in_wit))) => {
            let (_, ze_pf) = whir_prove_eval(&mut ze_wit, &x_star, &mut transcript, zk, rng)?;
            let (_, za_pf) = whir_prove_eval(&mut za_wit, &x_star, &mut transcript, zk, rng)?;
            let (_, zb_pf) = whir_prove_eval(&mut zb_wit, &x_star, &mut transcript, zk, rng)?;
            let (_, zc_pf) = whir_prove_eval(&mut zc_wit, &x_star, &mut transcript, zk, rng)?;
            let (_, za_in_pf) = whir_prove_eval(&mut za_in_wit, &y_star, &mut transcript, zk, rng)?;
            let (_, zb_in_pf) = whir_prove_eval(&mut zb_in_wit, &y_star, &mut transcript, zk, rng)?;
            (
                Some((ze_pf, za_pf, zb_pf, zc_pf)),
                Some((za_in_pf, zb_in_pf)),
            )
        }
        _ => (None, None),
    };

    // 8. WHIR evaluation proof for ŵ(y*).
    let (_v, whir_proof) = whir_prove_eval(&mut whir_witness, &y_star, &mut transcript, zk, rng)?;

    Ok(SNARKProof {
        whir_commitment,
        piop_proof,
        whir_proof,
        blinding_eval,
        blinding_commitment,
        blinding_whir_proof,
        z_out_factor_commitments: z_out_factor_commitments_opt,
        z_in_factor_commitments: z_in_factor_commitments_opt,
        z_out_factor_proofs,
        z_in_factor_proofs,
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify<F, C>(circuit: C, proof: &SNARKProof<F>) -> Result<bool, PcsError>
where
    F: PrimeField + FftField,
    C: ConstraintSynthesizer<F>,
{
    let mut transcript = Transcript::new(b"zk-snark");

    // 1. Rebuild R1CS structure from the circuit (witness values irrelevant here).
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone()).unwrap();
    cs.finalize();

    let matrices = cs.to_matrices().unwrap();
    let instance = cs.borrow().unwrap().instance_assignment.clone();

    // 2. Bind the stated public inputs to the transcript.
    for &v in &instance[1..] {
        transcript.absorb_field(v);
    }

    // 3. Absorb commitment roots (ŵ, B, ZE/ZA/ZB/ZC, Zα/Zβ — whichever are present).
    transcript.absorb(&proof.whir_commitment.root);
    if let Some(ref bc) = proof.blinding_commitment {
        transcript.absorb(&bc.root);
    }
    if let Some((ref ze_c, ref za_c, ref zb_c, ref zc_c)) = proof.z_out_factor_commitments {
        transcript.absorb(&ze_c.root);
        transcript.absorb(&za_c.root);
        transcript.absorb(&zb_c.root);
        transcript.absorb(&zc_c.root);
    }
    if let Some((ref za_in_c, ref zb_in_c)) = proof.z_in_factor_commitments {
        transcript.absorb(&za_in_c.root);
        transcript.absorb(&zb_in_c.root);
    }

    // 4. Verify PIOP → yields (x*, y*, u) where u = w̃(y*).
    let Some((x_star, y_star, u)) = piop_verify(&matrices, &proof.piop_proof, &mut transcript)
    else {
        return Ok(false);
    };

    // 5. ZK: verify B(y*) via its own WHIR proof, then check ŵ(y*) = w̃(y*) + B(y*).
    match (proof.blinding_eval, &proof.blinding_commitment, &proof.blinding_whir_proof) {
        (Some(b), Some(bc), Some(bp)) => {
            transcript.absorb_field(b);
            if !whir_verify_eval(bc, &y_star, b, bp, &mut transcript)? {
                return Ok(false);
            }

            // 5b. ZK: verify the Z_out factor openings at x* and reconstruct z_out_eval.
            //     Then verify the Z_in factor openings at y* and reconstruct z_in_eval.
            //     Check both against the values piop_verify already used.
            match (
                &proof.z_out_factor_commitments,
                &proof.z_out_factor_proofs,
                proof.piop_proof.z_out_eval,
                &proof.z_in_factor_commitments,
                &proof.z_in_factor_proofs,
                proof.piop_proof.z_in_eval,
            ) {
                (
                    Some((ze_c, za_c, zb_c, zc_c)),
                    Some((ze_p, za_p, zb_p, zc_p)),
                    Some(z_out_claim),
                    Some((za_in_c, zb_in_c)),
                    Some((za_in_p, zb_in_p)),
                    Some(z_in_claim),
                ) => {
                    // Open each Z_out factor at x* and collect the proved values.
                    let ze_v = ze_p.final_value;
                    if !whir_verify_eval(ze_c, &x_star, ze_v, ze_p, &mut transcript)? {
                        return Ok(false);
                    }
                    let za_v = za_p.final_value;
                    if !whir_verify_eval(za_c, &x_star, za_v, za_p, &mut transcript)? {
                        return Ok(false);
                    }
                    let zb_v = zb_p.final_value;
                    if !whir_verify_eval(zb_c, &x_star, zb_v, zb_p, &mut transcript)? {
                        return Ok(false);
                    }
                    let zc_v = zc_p.final_value;
                    if !whir_verify_eval(zc_c, &x_star, zc_v, zc_p, &mut transcript)? {
                        return Ok(false);
                    }

                    // Reconstruct z_out_eval and compare with what the PIOP trusted.
                    let z_out_reconstructed = ze_v * (za_v * zb_v - zc_v);
                    if z_out_reconstructed != z_out_claim {
                        return Ok(false);
                    }

                    // Open each Z_in factor at y*.
                    let za_in_v = za_in_p.final_value;
                    if !whir_verify_eval(za_in_c, &y_star, za_in_v, za_in_p, &mut transcript)? {
                        return Ok(false);
                    }
                    let zb_in_v = zb_in_p.final_value;
                    if !whir_verify_eval(zb_in_c, &y_star, zb_in_v, zb_in_p, &mut transcript)? {
                        return Ok(false);
                    }

                    let z_in_reconstructed = za_in_v * zb_in_v;
                    if z_in_reconstructed != z_in_claim {
                        return Ok(false);
                    }
                }
                _ => return Ok(false), // ZK mode requires all factor fields
            }

            // 6. Verify WHIR: ŵ(y*) = w̃(y*) + B(y*).
            whir_verify_eval(&proof.whir_commitment, &y_star, u + b, &proof.whir_proof, &mut transcript)
        }
        _ => {
            // 6. Non-ZK: verify WHIR directly.
            whir_verify_eval(&proof.whir_commitment, &y_star, u, &proof.whir_proof, &mut transcript)
        }
    }
}
