//! Fold-and-commit multilinear PCS, following the chapter's BaseFold variant.
//!
//! The Boolean evaluation table is Möbius-transformed to multilinear
//! coefficients, Reed--Solomon encoded, and Merkle committed. Folding the
//! codeword with coordinate `r[i]` partially evaluates variable `i`; after `ell`
//! folds the remaining constant is `f̃(r)`. The proximity test opens the `±` pair
//! at each layer on random full-domain paths. Each folded root is absorbed
//! before the next folding challenge. Compiled openings reuse the surrounding
//! sumcheck; standalone openings supply an interleaved evaluation sumcheck.
//! See README.md for the correction to the chapter and remaining security limits.
use ark_crypto_primitives::merkle_tree::{MerkleTree, Path};
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;

use crate::{
    merkle::{
        BLOWUP, Hash, MerkleConfig, build_tree, field_to_bytes, make_leaf_bytes_public, num_queries,
    },
    r1cs::build_eq_table,
    sumcheck::{SumcheckProof, sumcheck_verify_interleaved},
    transcript::Transcript,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcsError {
    InvalidDomain,
    DivisionByZero,
    InvalidShape,
}

/// Multilinear Möbius transform: evaluations on the Boolean cube → multilinear
/// coefficients.  `coeffs[a] = Σ_{x ⊆ a} (−1)^{|a|−|x|} evals[x]`, computed in place by
/// the standard subtract-butterfly (`O(2^ell · ell)`).
fn evals_to_coeffs<F: Field>(mut a: Vec<F>) -> Vec<F> {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut step = 1;
    while step < n {
        let mut start = 0;
        while start < n {
            for i in start..start + step {
                let lo = a[i];
                a[i + step] -= lo;
            }
            start += step * 2;
        }
        step *= 2;
    }
    a
}

/// Encode an evaluation table into the rate-`1/BLOWUP` RS codeword used for commitment.
///
/// `c_0[j] = P(ω_0^j)` where `P(X) = Σ_b coeffs[b]·X^b`, `coeffs = evals_to_coeffs(evals)`,
/// and `ω_0` is the generator of the size-`BLOWUP·evals.len()` evaluation domain.  Unlike a
/// systematic encoding, `c_0[BLOWUP·k] ≠ evals[k]`: the FRI fold relates codewords across
/// layers at *all* positions, so no systematic structure is needed (or wanted).
pub fn rs_encode<F: FftField>(evals: &[F]) -> Vec<F> {
    debug_assert!(evals.len().is_power_of_two());
    let target = BLOWUP * evals.len();
    let domain =
        Radix2EvaluationDomain::<F>::new(target).expect("field supports NTT of target size");
    let mut buf = evals_to_coeffs(evals.to_vec());
    buf.resize(target, F::zero());
    domain.fft_in_place(&mut buf);
    buf
}

/// One FRI fold of a codeword `cw` (evals of `P` on `D`, |D| = cw.len()) by `alpha`,
/// returning the codeword of `P^e + alpha·P^o` on `D² = {x² : x ∈ D}` (half the size).
///
/// For `x = ω^j` (so `−x = ω^{j+half}`):
///   P^e(x²) = (P(x)+P(−x))/2,  P^o(x²) = (P(x)−P(−x))/(2x),
///   out[j]  = P^e(x²) + alpha·P^o(x²).
fn fold_codeword<F: FftField>(cw: &[F], alpha: F) -> Vec<F> {
    let n = cw.len();
    let half = n / 2;
    let omega = Radix2EvaluationDomain::<F>::new(n)
        .expect("field supports NTT of this size")
        .group_gen;
    let inv2 = F::from(2u64).inverse().expect("2 is invertible");
    let omega_inv = omega.inverse().expect("omega is invertible");

    let mut out = Vec::with_capacity(half);
    let mut wpow_inv = F::one();
    for j in 0..half {
        let a = cw[j];
        let b = cw[j + half];
        let inv2x = inv2 * wpow_inv;
        out.push((a + b) * inv2 + alpha * (a - b) * inv2x);
        wpow_inv *= omega_inv;
    }
    out
}

/// Verifier-side single fold of one ± pair: same formula as `fold_codeword` at the
/// index `low`, with `omega` the generator of this layer's domain (`x = omega^low`).
fn fold_pair<F: FftField>(a: F, b: F, alpha: F, omega: F, low: usize) -> F {
    let inv2 = F::from(2u64).inverse().expect("2 is invertible");
    let x = omega.pow([low as u64]);
    let inv2x = inv2 * x.inverse().expect("domain element is invertible");
    (a + b) * inv2 + alpha * (a - b) * inv2x
}

/// `[ (size, generator) ]` for the `ell` fold layers, sizes `BLOWUP·2^ell, …, BLOWUP·2`.
fn layer_domains<F: FftField>(ell: usize) -> Result<Vec<(usize, F)>, PcsError> {
    let mut n = 1usize
        .checked_shl(ell.try_into().map_err(|_| PcsError::InvalidShape)?)
        .and_then(|n| n.checked_mul(BLOWUP))
        .ok_or(PcsError::InvalidShape)?;
    let mut out = Vec::with_capacity(ell);
    for _ in 0..ell {
        let omega = Radix2EvaluationDomain::<F>::new(n)
            .ok_or(PcsError::InvalidDomain)?
            .group_gen;
        out.push((n, omega));
        n /= 2;
    }
    Ok(out)
}

#[derive(Clone, CanonicalSerialize)]
pub struct Commitment {
    pub root: Hash,
}

/// Prover data for the Boolean table, its RS codeword, and its Merkle tree.
pub struct Witness<F: PrimeField> {
    pub evals: Vec<F>,
    pub codeword: Vec<F>,
    pub tree: MerkleTree<MerkleConfig>,
}

/// Commit to public (non-hiding) data with no randomness.
pub fn commit_public<F: PrimeField + FftField>(evals: Vec<F>) -> (Commitment, Witness<F>) {
    let cw = rs_encode(&evals);
    let leaf_bytes = make_leaf_bytes_public(&cw);
    finish_commit(evals, cw, leaf_bytes)
}

fn finish_commit<F: PrimeField>(
    evals: Vec<F>,
    codeword: Vec<F>,
    leaf_bytes: Vec<Vec<u8>>,
) -> (Commitment, Witness<F>) {
    let tree = build_tree(&leaf_bytes);
    let root = tree.root();
    (
        Commitment { root },
        Witness {
            evals,
            codeword,
            tree,
        },
    )
}

/// One query: the ± pair opened at each of the `ell` layers (layer 0 against the
/// commitment h_0, layer i ≥ 1 against the intermediate root h_i), with Merkle paths.
#[derive(CanonicalSerialize)]
pub struct QueryProof<F: PrimeField> {
    /// `a_vals[i]` = c_i[low_i], `b_vals[i]` = c_i[low_i + |D_i|/2]  (length ell).
    pub a_vals: Vec<F>,
    pub b_vals: Vec<F>,
    pub a_paths: Vec<Path<MerkleConfig>>,
    pub b_paths: Vec<Path<MerkleConfig>>,
}

#[derive(CanonicalSerialize)]
pub struct Proof<F: PrimeField> {
    /// Root h_i is absorbed immediately after challenge r_{i-1}, before r_i.
    pub intermediate_roots: Vec<Hash>,
    /// The collapsed constant f̃(r), absorbed immediately after the last fold.
    pub final_value: F,
    /// `num_queries()` query proofs.
    pub queries: Vec<QueryProof<F>>,
}

impl<F: PrimeField> Proof<F> {
    /// This is only transcript replay, not an evaluation proof verifier. The
    /// caller must also check the sumcheck terminal identity and Merkle queries.
    pub(crate) fn absorb_round(
        &self,
        round: usize,
        ell: usize,
        transcript: &mut Transcript,
    ) -> Option<()> {
        if ell == 0 || round >= ell || self.intermediate_roots.len() != ell - 1 {
            return None;
        }
        if round + 1 < ell {
            transcript.absorb(&self.intermediate_roots[round]);
        } else {
            transcript.absorb_field(self.final_value);
        }
        Some(())
    }
}

/// Intermediate codewords, commitments, and the final folded constant.
pub(crate) struct EvalData<F: PrimeField> {
    owned_codewords: Vec<Vec<F>>,
    owned_trees: Vec<MerkleTree<MerkleConfig>>,
    intermediate_roots: Vec<Hash>,
    final_value: F,
}

/// Incremental fold state. Never construct the whole chain from an already
/// known point: callers absorb each round before drawing the next challenge.
pub(crate) struct FoldProver<F: PrimeField> {
    current: Vec<F>,
    ell: usize,
    round: usize,
    data: EvalData<F>,
}

impl<F: PrimeField + FftField> FoldProver<F> {
    pub(crate) fn new(codeword: &[F]) -> Result<Self, PcsError> {
        if !codeword.len().is_power_of_two() || codeword.len() < 2 * BLOWUP {
            return Err(PcsError::InvalidShape);
        }
        let ell = (codeword.len() / BLOWUP).ilog2() as usize;
        layer_domains::<F>(ell)?;
        Ok(Self {
            current: codeword.to_vec(),
            ell,
            round: 0,
            data: EvalData {
                owned_codewords: Vec::new(),
                owned_trees: Vec::new(),
                intermediate_roots: Vec::new(),
                final_value: F::zero(),
            },
        })
    }

    pub(crate) fn advance_public(&mut self, alpha: F) {
        assert!(self.round < self.ell, "too many PCS folds");
        let next = fold_codeword(&self.current, alpha);
        if self.round + 1 < self.ell {
            let leaves = make_leaf_bytes_public(&next);
            let tree = build_tree(&leaves);
            self.data.intermediate_roots.push(tree.root());
            self.data.owned_codewords.push(next.clone());
            self.data.owned_trees.push(tree);
        } else {
            self.data.final_value = next[0];
        }
        self.current = next;
        self.round += 1;
    }

    pub(crate) fn absorb_latest(&self, transcript: &mut Transcript) {
        assert!(self.round > 0);
        if self.round < self.ell {
            transcript.absorb(self.data.intermediate_roots.last().unwrap());
        } else {
            transcript.absorb_field(self.data.final_value);
        }
    }

    fn into_data(self) -> EvalData<F> {
        assert_eq!(self.round, self.ell, "unfinished PCS folds");
        self.data
    }

    pub(crate) fn finish(self, witness: &Witness<F>, transcript: &mut Transcript) -> Proof<F> {
        let ell = self.ell;
        finalize_eval(witness, ell, self.into_data(), transcript)
    }
}

/// Open queries after all fold messages have already been absorbed in their rounds.
fn finalize_eval<F: PrimeField + FftField>(
    witness: &Witness<F>,
    ell: usize,
    data: EvalData<F>,
    transcript: &mut Transcript,
) -> Proof<F> {
    let n0 = witness.codeword.len();
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    let codeword = |i: usize| -> &[F] {
        if i == 0 {
            &witness.codeword
        } else {
            &data.owned_codewords[i - 1]
        }
    };
    let tree = |i: usize| -> &MerkleTree<MerkleConfig> {
        if i == 0 {
            &witness.tree
        } else {
            &data.owned_trees[i - 1]
        }
    };
    let mut queries = Vec::with_capacity(num_queries());
    for &s in &query_lows {
        let mut a_vals = Vec::with_capacity(ell);
        let mut b_vals = Vec::with_capacity(ell);
        let mut a_paths = Vec::with_capacity(ell);
        let mut b_paths = Vec::with_capacity(ell);

        let mut low = s;
        let mut n_i = n0;
        for i in 0..ell {
            let half = n_i / 2;
            let low_i = low % half;
            let cw = codeword(i);
            a_vals.push(cw[low_i]);
            b_vals.push(cw[low_i + half]);
            a_paths.push(tree(i).generate_proof(low_i).unwrap());
            b_paths.push(tree(i).generate_proof(low_i + half).unwrap());
            low = low_i;
            n_i = half;
        }
        queries.push(QueryProof {
            a_vals,
            b_vals,
            a_paths,
            b_paths,
        });
    }

    Proof {
        intermediate_roots: data.intermediate_roots,
        final_value: data.final_value,
        queries,
    }
}

/// Standalone evaluation proof for a point already known to the prover.
/// `folds.final_value` is f(alpha), not the claimed f(r).
#[derive(CanonicalSerialize)]
pub struct EvaluationProof<F: PrimeField> {
    pub sc: SumcheckProof<F>,
    pub folds: Proof<F>,
}

fn absorb_point<F: PrimeField>(r: &[F], transcript: &mut Transcript) {
    transcript.absorb(&(r.len() as u64).to_le_bytes());
    for &ri in r {
        transcript.absorb_field(ri);
    }
}

fn check_witness_shape<F: PrimeField>(witness: &Witness<F>, ell: usize) -> Result<(), PcsError> {
    if ell == 0
        || ell >= usize::BITS as usize
        || witness.evals.len() != 1usize << ell
        || witness.evals.len().checked_mul(BLOWUP) != Some(witness.codeword.len())
    {
        return Err(PcsError::InvalidShape);
    }
    Ok(())
}

/// Sumcheck of f(b) * eq(b, r), sharing its fresh challenges with the folds.
fn prove_evaluation_sumcheck<F: PrimeField>(
    evals: &[F],
    r: &[F],
    transcript: &mut Transcript,
    mut after_challenge: impl FnMut(F, &mut Transcript),
) -> (F, SumcheckProof<F>) {
    let mut a = evals.to_vec();
    let mut b = build_eq_table(r);
    let v: F = a.iter().zip(&b).map(|(&a, &b)| a * b).sum();
    transcript.absorb_field(v);
    let mut round_polys = Vec::with_capacity(r.len());
    for _ in r {
        let half = a.len() / 2;
        let mut g = vec![F::zero(); 3];
        for k in 0..half {
            for (t, gt) in g.iter_mut().enumerate() {
                let t = F::from(t as u64);
                *gt += (a[2 * k] + t * (a[2 * k + 1] - a[2 * k]))
                    * (b[2 * k] + t * (b[2 * k + 1] - b[2 * k]));
            }
        }
        for &gi in &g {
            transcript.absorb_field(gi);
        }
        let alpha = transcript.squeeze_field();
        after_challenge(alpha, transcript);
        round_polys.push(g);
        for k in 0..half {
            a[k] = a[2 * k] + alpha * (a[2 * k + 1] - a[2 * k]);
            b[k] = b[2 * k] + alpha * (b[2 * k + 1] - b[2 * k]);
        }
        a.truncate(half);
        b.truncate(half);
    }
    (v, SumcheckProof { round_polys })
}

fn eq_points<F: PrimeField>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| a * b + (F::one() - a) * (F::one() - b))
        .product()
}

pub fn prove_eval<F: PrimeField + FftField>(
    witness: &Witness<F>,
    r: &[F],
    transcript: &mut Transcript,
) -> Result<(F, EvaluationProof<F>), PcsError> {
    check_witness_shape(witness, r.len())?;
    transcript.absorb(b"pcs-eval/interleaved-v1");
    transcript.absorb(&witness.tree.root());
    absorb_point(r, transcript);
    let mut folds = FoldProver::new(&witness.codeword)?;
    let (v, sc) = prove_evaluation_sumcheck(&witness.evals, r, transcript, |alpha, transcript| {
        folds.advance_public(alpha);
        folds.absorb_latest(transcript);
    });
    Ok((
        v,
        EvaluationProof {
            sc,
            folds: folds.finish(witness, transcript),
        },
    ))
}

pub fn verify_eval<F: PrimeField + FftField>(
    commitment: &Commitment,
    r: &[F],
    v: F,
    proof: &EvaluationProof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    if r.is_empty() {
        return Ok(false);
    }
    transcript.absorb(b"pcs-eval/interleaved-v1");
    transcript.absorb(&commitment.root);
    absorb_point(r, transcript);
    let Some((alpha, final_claim)) =
        sumcheck_verify_interleaved(&proof.sc, r.len(), 2, v, transcript, |i, _, t| {
            proof.folds.absorb_round(i, r.len(), t)
        })
    else {
        return Ok(false);
    };
    if final_claim != proof.folds.final_value * eq_points(&alpha, r) {
        return Ok(false);
    }
    verify_queries(
        commitment,
        &alpha,
        proof.folds.final_value,
        &proof.folds,
        transcript,
    )
}

/// Only checks authentication and fold consistency. The caller must have
/// replayed `Proof::absorb_round` between the challenges in `r`.
pub(crate) fn verify_queries<F: PrimeField + FftField>(
    commitment: &Commitment,
    r: &[F],
    v: F,
    proof: &Proof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    let ell = r.len();
    if ell == 0 {
        return Ok(false);
    }
    if proof.intermediate_roots.len() != ell - 1 {
        return Ok(false);
    }
    if proof.queries.len() != num_queries() {
        return Ok(false);
    }

    let domains = layer_domains::<F>(ell)?;
    let n0 = domains[0].0;
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    for (qi, &s) in query_lows.iter().enumerate() {
        let qp = &proof.queries[qi];
        if qp.a_vals.len() != ell
            || qp.b_vals.len() != ell
            || qp.a_paths.len() != ell
            || qp.b_paths.len() != ell
        {
            return Ok(false);
        }

        let mut low = s;
        let mut folded: Option<F> = None;
        for i in 0..ell {
            let (n_i, omega_i) = domains[i];
            let half = n_i / 2;
            let low_i = low % half;
            let a = qp.a_vals[i];
            let b = qp.b_vals[i];

            let root_i = if i == 0 {
                &commitment.root
            } else {
                &proof.intermediate_roots[i - 1]
            };
            if !verify_leaf(a, low_i, n_i, &qp.a_paths[i], root_i) {
                return Ok(false);
            }
            if !verify_leaf(b, low_i + half, n_i, &qp.b_paths[i], root_i) {
                return Ok(false);
            }

            // Match the previous fold to its committed symbol in this layer.
            if let Some(fv) = folded {
                let matched = if low < half { a } else { b };
                if matched != fv {
                    return Ok(false);
                }
            }

            folded = Some(fold_pair(a, b, r[i], omega_i, low_i));
            low = low_i;
        }

        if folded != Some(proof.final_value) {
            return Ok(false);
        }
    }

    Ok(proof.final_value == v)
}

fn verify_leaf<F: PrimeField>(
    val: F,
    expected_idx: usize,
    expected_size: usize,
    path: &Path<MerkleConfig>,
    root: &Hash,
) -> bool {
    if !expected_size.is_power_of_two()
        || expected_idx >= expected_size
        || path.leaf_index != expected_idx
        || path.auth_path.len().checked_add(1) != Some(expected_size.ilog2() as usize)
        || root.len() != 32
        || path.leaf_sibling_hash.len() != 32
        || path.auth_path.iter().any(|digest| digest.len() != 32)
    {
        return false;
    }
    let bytes = field_to_bytes(val);
    path.verify(&(), &(), root, bytes.as_slice())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::mle_of_vector;
    use crate::transcript::Transcript;
    use ark_bls12_381::Fr as F;
    use ark_std::UniformRand;
    use rand::{SeedableRng, rngs::StdRng};

    fn rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    #[test]
    fn test_fold_collapses_to_mle_eval() {
        let mut rng = rng();
        for ell in 1..=5usize {
            let evals: Vec<F> = (0..(1u64 << ell)).map(|_| F::rand(&mut rng)).collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
            let (_c, witness) = commit_public(evals.clone());
            let mut folds = FoldProver::new(&witness.codeword).unwrap();
            for &ri in &r {
                folds.advance_public(ri);
            }
            let data = folds.into_data();
            assert_eq!(
                data.final_value,
                mle_of_vector(&evals, ell, &r),
                "ell={ell}"
            );
        }
    }

    #[test]
    fn test_round_trip_non_zk() {
        let mut rng = rng();
        for ell in 1..=5 {
            let evals: Vec<F> = (0..1usize << ell).map(|_| F::rand(&mut rng)).collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
            let expected = mle_of_vector(&evals, ell, &r);
            let (commitment, witness) = commit_public(evals);
            let mut pt = Transcript::new(b"pcs-test");
            let mut pv = Transcript::new(b"pcs-test");
            let (v, proof) = prove_eval(&witness, &r, &mut pt).unwrap();
            assert_eq!(v, expected);
            assert!(verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
            assert_eq!(pt.squeeze_field::<F>(), pv.squeeze_field::<F>());
        }
    }

    #[test]
    fn test_wrong_claimed_value_rejected() {
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];

        let (commitment, witness) = commit_public(evals);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, proof) = prove_eval(&witness, &r, &mut pt).unwrap();

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v + F::from(1u64), &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_forged_value_rejected() {
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let (commitment, witness) = commit_public(evals);

        let mut pt = Transcript::new(b"forge");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        proof.folds.final_value += F::from(1u64);

        let mut vt = Transcript::new(b"forge");
        assert!(!verify_eval(&commitment, &r, v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_deferred_fold_commitments_rejected() {
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let (comm, witness) = commit_public(evals);
        let mut pt = Transcript::new(b"deferred");
        pt.absorb(b"pcs-eval/interleaved-v1");
        pt.absorb(&comm.root);
        absorb_point(&r, &mut pt);
        let mut folds = FoldProver::new(&witness.codeword).unwrap();
        // Omitting interleaved roots must invalidate the transcript even when
        // the fold computations themselves are correct.
        let (v, sc) = prove_evaluation_sumcheck(&witness.evals, &r, &mut pt, |alpha, _| {
            folds.advance_public(alpha);
        });
        let data = folds.into_data();
        for root in &data.intermediate_roots {
            pt.absorb(root);
        }
        pt.absorb_field(data.final_value);
        let proof = EvaluationProof {
            sc,
            folds: finalize_eval(&witness, r.len(), data, &mut pt),
        };
        let mut vt = Transcript::new(b"deferred");
        assert!(!verify_eval(&comm, &r, v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_fold_root_binds_next_challenge() {
        let (comm, witness) = commit_public((1..=8u64).map(F::from).collect());
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let mut pt = Transcript::new(b"schedule");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        let replay = |proof: &EvaluationProof<F>| {
            let mut t = Transcript::new(b"schedule");
            t.absorb(b"pcs-eval/interleaved-v1");
            t.absorb(&comm.root);
            absorb_point(&r, &mut t);
            let mut challenges = Vec::new();
            let _ = sumcheck_verify_interleaved(&proof.sc, r.len(), 2, v, &mut t, |i, alpha, t| {
                challenges.push(alpha);
                proof.folds.absorb_round(i, r.len(), t)
            });
            challenges
        };
        let original = replay(&proof);
        proof.folds.intermediate_roots[0][0] ^= 1;
        let altered = replay(&proof);
        assert_eq!(original[0], altered[0]);
        assert_ne!(original[1], altered[1]);
        let mut vt = Transcript::new(b"schedule");
        assert!(!verify_eval(&comm, &r, v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_evaluation_statement_and_shape_binding() {
        let (comm, witness) = commit_public((1..=8u64).map(F::from).collect());
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let mut pt = Transcript::new(b"binding");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        let accepts = |c: &Commitment, point: &[F], p: &EvaluationProof<F>| {
            verify_eval(c, point, v, p, &mut Transcript::new(b"binding")).unwrap()
        };
        assert!(accepts(&comm, &r, &proof));
        let mut wrong_point = r.clone();
        wrong_point[0] += F::from(1u64);
        assert!(!accepts(&comm, &wrong_point, &proof));
        let mut wrong_comm = comm.clone();
        wrong_comm.root[0] ^= 1;
        assert!(!accepts(&wrong_comm, &r, &proof));
        let root = proof.folds.intermediate_roots.pop().unwrap();
        assert!(!accepts(&comm, &r, &proof));
        proof.folds.intermediate_roots.push(root.clone());
        proof.folds.intermediate_roots.push(root);
        assert!(!accepts(&comm, &r, &proof));
        proof.folds.intermediate_roots.pop();
        let round = proof.sc.round_polys.pop().unwrap();
        assert!(!accepts(&comm, &r, &proof));
        proof.sc.round_polys.push(round.clone());
        proof.sc.round_polys.push(round);
        assert!(!accepts(&comm, &r, &proof));
    }

    #[test]
    fn test_tampered_leaf_rejected() {
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];
        let (commitment, witness) = commit_public(evals);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();

        proof.folds.queries[0].a_vals[0] += F::from(1u64);

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }
}
