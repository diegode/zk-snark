//! Spartan with public sparse-matrix proofs and native masked ZK openings.
//! See README.md for the key corrections relative to the published chapter.
use crate::{
    matrix_eval::{
        MatrixCommitments, MatrixEncoding, MatrixEvalProof, encode_matrix, prove_matrix_eval,
        verify_matrix_eval,
    },
    pcs::{Commitment, FoldProver, PcsError, Proof, commit_public, verify_queries},
    piop::{PiopProof, ell_for, piop_prove, piop_verify},
    r1cs::ConstraintMatrices,
    transcript::Transcript,
    zk_pcs::{self, RelationProof},
    zk_piop::{self, ZkPiopProof},
};
use ark_ff::{FftField, PrimeField};
use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, Matrix, SynthesisError,
    predicate::{Predicate, PredicateConstraintSystem},
};
use ark_serialize::CanonicalSerialize;
use rand::{CryptoRng, Rng};
use rayon::prelude::*;

pub struct ProverParams<F: PrimeField> {
    pub matrices: ConstraintMatrices<F>,
    pub enc_a: MatrixEncoding<F>,
    pub enc_b: MatrixEncoding<F>,
    pub enc_c: MatrixEncoding<F>,
    pub ell_row: usize,
    pub ell_col: usize,
    /// Whether native ZK proving is enabled.
    pub zk: bool,
}

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

#[derive(Debug)]
pub enum SnarkError {
    Synthesis(SynthesisError),
    Pcs(PcsError),
    UnsupportedPredicate(String),
    InvalidR1csPredicate,
    InvalidR1csShape,
    CircuitMismatch,
}

impl From<SynthesisError> for SnarkError {
    fn from(error: SynthesisError) -> Self {
        Self::Synthesis(error)
    }
}

impl From<PcsError> for SnarkError {
    fn from(error: PcsError) -> Self {
        Self::Pcs(error)
    }
}

type MatrixProofs<F> = (MatrixEvalProof<F>, MatrixEvalProof<F>, MatrixEvalProof<F>);

#[derive(CanonicalSerialize)]
pub struct PlainProof<F: PrimeField> {
    pub w_commitment: Commitment,
    pub piop_proof: PiopProof<F>,
    pub matrix_eval_proofs: MatrixProofs<F>,
    pub w_proof: Proof<F>,
}

#[derive(CanonicalSerialize)]
pub struct ZkProof<F: PrimeField> {
    pub commitment: Vec<u8>,
    pub piop: ZkPiopProof<F>,
    pub matrix_eval_proofs: MatrixProofs<F>,
    pub opening: RelationProof<F>,
}

/// Exactly one variant must be present. The ZK variant has no raw witness or
/// inner-mask evaluation fields.
#[derive(CanonicalSerialize)]
pub struct SNARKProof<F: PrimeField> {
    pub plain: Option<PlainProof<F>>,
    pub zk: Option<ZkProof<F>>,
}

fn matrices_of<F: PrimeField>(
    cs: &ConstraintSystemRef<F>,
) -> Result<ConstraintMatrices<F>, SnarkError> {
    for (label, count) in cs.get_all_predicates_num_constraints() {
        if label != "R1CS" && count != 0 {
            return Err(SnarkError::UnsupportedPredicate(label));
        }
    }

    // Arkworks permits replacing a predicate under an existing label. The
    // Spartan reduction requires the actual relation a*b-c=0, not just three
    // matrices that happen to be labelled "R1CS".
    let standard = PredicateConstraintSystem::<F>::new_r1cs()?
        .get_predicate()
        .clone();
    let actual = cs
        .get_predicate_type("R1CS")
        .ok_or(SnarkError::InvalidR1csPredicate)?;
    let canonical = match (actual, standard) {
        (Predicate::Polynomial(a), Predicate::Polynomial(b)) => {
            a.polynomial.num_vars == b.polynomial.num_vars
                && a.polynomial.terms == b.polynomial.terms
        }
        _ => false,
    };
    if !canonical {
        return Err(SnarkError::InvalidR1csPredicate);
    }

    let inner = cs.borrow().ok_or(SnarkError::InvalidR1csShape)?;
    let mut raw = cs.to_matrices()?;
    let abc: [Matrix<F>; 3] = raw
        .remove("R1CS")
        .ok_or(SnarkError::InvalidR1csShape)?
        .try_into()
        .map_err(|_| SnarkError::InvalidR1csShape)?;
    let num_constraints = inner
        .get_predicate_num_constraints("R1CS")
        .ok_or(SnarkError::InvalidR1csShape)?;
    if abc.iter().any(|matrix| matrix.len() != num_constraints) {
        return Err(SnarkError::InvalidR1csShape);
    }
    let [a, b, c] = abc;
    Ok(ConstraintMatrices {
        a,
        b,
        c,
        num_instance_variables: inner.num_instance_variables,
        num_witness_variables: inner.num_witness_variables,
        num_constraints,
    })
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
            entry.1 = if c < num_instance {
                half + c
            } else {
                c - num_instance
            };
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
/// ZK encoding randomness lives outside the R1CS assignment. Both modes use the
/// same matrix dimensions; `zk` records whether ZK proving is enabled. Setup
/// rejects predicates outside the canonical R1CS relation.
pub fn setup<F, C>(circuit: C, zk: bool) -> Result<(ProverParams<F>, VerifierParams), SnarkError>
where
    F: PrimeField + FftField,
    C: ConstraintSynthesizer<F>,
{
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let mut matrices = matrices_of(&cs)?;
    let num_instance = matrices.num_instance_variables;
    let num_witness = matrices.num_witness_variables;

    let s = ell_for(num_witness.max(1)).max(ell_for(num_instance));
    let ell_col = s + 1;
    let half = 1usize << s;
    remap_columns(&mut matrices.a, num_instance, half);
    remap_columns(&mut matrices.b, num_instance, half);
    remap_columns(&mut matrices.c, num_instance, half);

    let ell_row = ell_for(matrices.num_constraints.max(1));

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
    let pp = ProverParams {
        matrices,
        enc_a,
        enc_b,
        enc_c,
        ell_row,
        ell_col,
        zk,
    };
    Ok((pp, vp))
}

/// Prove for the relation fixed at setup; a differently synthesized circuit is rejected.
pub fn prove<F, C, R>(
    pp: &ProverParams<F>,
    circuit: C,
    zk: bool,
    rng: &mut R,
) -> Result<SNARKProof<F>, SnarkError>
where
    F: PrimeField + FftField,
    C: ConstraintSynthesizer<F>,
    R: Rng + CryptoRng,
{
    assert!(
        !zk || pp.zk,
        "ZK proving requires ZK parameters: call setup(circuit, true)"
    );
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let mut matrices = matrices_of(&cs)?;
    let half = 1usize << (pp.ell_col - 1);
    remap_columns(&mut matrices.a, matrices.num_instance_variables, half);
    remap_columns(&mut matrices.b, matrices.num_instance_variables, half);
    remap_columns(&mut matrices.c, matrices.num_instance_variables, half);
    if matrices != pp.matrices {
        return Err(SnarkError::CircuitMismatch);
    }
    let (instance, mut witness) = {
        let inner = cs.borrow().unwrap();
        (
            inner.assignments.instance_assignment.clone(),
            inner.assignments.witness_assignment.clone(),
        )
    };
    if instance.len() != pp.matrices.num_instance_variables
        || witness.len() != pp.matrices.num_witness_variables
    {
        return Err(SnarkError::CircuitMismatch);
    }
    let mut t = Transcript::new(if zk {
        b"zk-snark/native-zk-v1"
    } else {
        b"zk-snark/interleaved-v1"
    });
    for &v in &instance[1..] {
        t.absorb_field(v);
    }
    pp.enc_a.commitments.absorb_into(&mut t);
    pp.enc_b.commitments.absorb_into(&mut t);
    pp.enc_c.commitments.absorb_into(&mut t);
    let s = pp.ell_col - 1;
    let half = 1usize << s;
    witness.resize(half, F::zero());
    let mut assignment = witness.clone();
    assignment.resize(2 * half, F::zero());
    assignment[half..half + instance.len()].copy_from_slice(&instance);
    if zk {
        let secrets = zk_piop::Secrets::sample(pp.ell_row, pp.ell_col, rng);
        let (commitment, opening_witness) = zk_pcs::commit(secrets.pack(&witness), rng)?;
        t.absorb(&commitment);
        let (piop, context) = zk_piop::prove(
            &pp.matrices,
            &assignment,
            pp.ell_row,
            pp.ell_col,
            &secrets,
            &mut t,
        );
        let matrix_eval_proofs = prove_matrices(pp, &context.x, &context.y, &mut t)?;
        let (form, target) = zk_piop::relation(&piop, &context, &instance, &mut t);
        let opening = zk_pcs::prove(opening_witness, form, target, &mut t, rng)?;
        return Ok(SNARKProof {
            plain: None,
            zk: Some(ZkProof {
                commitment,
                piop,
                matrix_eval_proofs,
                opening,
            }),
        });
    }
    let (w_commitment, w_witness) = commit_public(witness);
    t.absorb(&w_commitment.root);
    let mut folds = FoldProver::new(&w_witness.codeword)?;
    let (piop_proof, x, y, _) = piop_prove(
        &pp.matrices,
        &assignment,
        pp.ell_row,
        pp.ell_col,
        &mut t,
        |i, a, t| {
            if i < s {
                folds.advance_public(a);
                folds.absorb_latest(t);
            }
        },
    );
    let matrix_eval_proofs = prove_matrices(pp, &x, &y, &mut t)?;
    let w_proof = folds.finish(&w_witness, &mut t);
    Ok(SNARKProof {
        plain: Some(PlainProof {
            w_commitment,
            piop_proof,
            matrix_eval_proofs,
            w_proof,
        }),
        zk: None,
    })
}

fn prove_matrices<F: PrimeField + FftField>(
    pp: &ProverParams<F>,
    x: &[F],
    y: &[F],
    t: &mut Transcript,
) -> Result<MatrixProofs<F>, PcsError> {
    let (_, a) = prove_matrix_eval(&pp.enc_a, x, y, t)?;
    let (_, b) = prove_matrix_eval(&pp.enc_b, x, y, t)?;
    let (_, c) = prove_matrix_eval(&pp.enc_c, x, y, t)?;
    Ok((a, b, c))
}

fn verify_matrices<F: PrimeField + FftField>(
    vp: &VerifierParams,
    x: &[F],
    y: &[F],
    values: [F; 3],
    p: &MatrixProofs<F>,
    t: &mut Transcript,
) -> Result<bool, PcsError> {
    Ok(verify_matrix_eval(&vp.comm_a, x, y, values[0], &p.0, t)?
        && verify_matrix_eval(&vp.comm_b, x, y, values[1], &p.1, t)?
        && verify_matrix_eval(&vp.comm_c, x, y, values[2], &p.2, t)?)
}

/// Public inputs include the leading constant one.
pub fn verify<F: PrimeField + FftField>(
    vp: &VerifierParams,
    public: &[F],
    proof: &SNARKProof<F>,
) -> Result<bool, PcsError> {
    if public.len() != vp.num_instance_variables
        || public.first() != Some(&F::one())
        || proof.plain.is_some() == proof.zk.is_some()
    {
        return Ok(false);
    }
    let mut t = Transcript::new(if proof.zk.is_some() {
        b"zk-snark/native-zk-v1"
    } else {
        b"zk-snark/interleaved-v1"
    });
    for &v in &public[1..] {
        t.absorb_field(v);
    }
    vp.comm_a.absorb_into(&mut t);
    vp.comm_b.absorb_into(&mut t);
    vp.comm_c.absorb_into(&mut t);
    if let Some(p) = &proof.zk {
        if p.commitment.len() != 32 {
            return Ok(false);
        }
        t.absorb(&p.commitment);
        let Some(context) = zk_piop::verify(vp.ell_row, vp.ell_col, &p.piop, &mut t) else {
            return Ok(false);
        };
        if !verify_matrices(
            vp,
            &context.x,
            &context.y,
            p.piop.matrix_evals,
            &p.matrix_eval_proofs,
            &mut t,
        )? {
            return Ok(false);
        }
        let (form, target) = zk_piop::relation(&p.piop, &context, public, &mut t);
        return zk_pcs::verify(&p.commitment, form, target, &p.opening, &mut t);
    }
    let p = proof.plain.as_ref().unwrap();
    t.absorb(&p.w_commitment.root);
    let s = vp.ell_col - 1;
    let Some((x, y, u)) = piop_verify(vp.ell_row, vp.ell_col, &p.piop_proof, &mut t, |i, _, t| {
        if i < s {
            p.w_proof.absorb_round(i, s, t)?;
        }
        Some(())
    }) else {
        return Ok(false);
    };
    if !verify_matrices(
        vp,
        &x,
        &y,
        [
            p.piop_proof.a_eval,
            p.piop_proof.b_eval,
            p.piop_proof.c_eval,
        ],
        &p.matrix_eval_proofs,
        &mut t,
    )? {
        return Ok(false);
    }
    let y_prime = &y[..s];
    let io: F = public
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            v * y_prime
                .iter()
                .enumerate()
                .map(|(j, &y)| if (i >> j) & 1 == 1 { y } else { F::one() - y })
                .product::<F>()
        })
        .sum();
    if !verify_queries(
        &p.w_commitment,
        y_prime,
        p.w_proof.final_value,
        &p.w_proof,
        &mut t,
    )? {
        return Ok(false);
    }
    Ok((F::one() - y[s]) * p.w_proof.final_value + y[s] * io == u)
}
