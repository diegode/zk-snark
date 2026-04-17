//! Fold-and-commit multilinear PCS, following the chapter's BaseFold variant.
//!
//! The Boolean evaluation table is Möbius-transformed to multilinear
//! coefficients, Reed--Solomon encoded, and Merkle committed. Folding the
//! codeword with coordinate `r[i]` partially evaluates variable `i`; after `ell`
//! folds the remaining constant is `f̃(r)`. The proximity test opens the `±` pair
//! at each layer on random full-domain paths. This variant uses the chapter's
//! unique-decoding analysis rather than BaseFold's interleaved sumcheck.
use ark_crypto_primitives::merkle_tree::{MerkleTree, Path};
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;
use rand::RngCore;

use crate::{
    merkle::{
        build_tree, field_to_bytes, make_leaf_bytes, make_leaf_bytes_public, num_queries, Hash,
        MerkleConfig, BLOWUP,
    },
    transcript::Transcript,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcsError {
    InvalidDomain,
    DivisionByZero,
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

/// Upper bound on the codeword symbols opened across all fold layers.
///
/// The ZK path uses this count to size the randomized padding described in the
/// chapter's Aurora-style encoding discussion.
pub fn opened_symbols_bound(ell: usize) -> usize {
    let t = num_queries();
    let mut total = 1;
    let mut n = BLOWUP << ell;
    for _ in 0..ell {
        total += (2 * t).min(n);
        n /= 2;
    }
    total
}

/// `[ (size, generator) ]` for the `ell` fold layers, sizes `BLOWUP·2^ell, …, BLOWUP·2`.
fn layer_domains<F: FftField>(ell: usize) -> Result<Vec<(usize, F)>, PcsError> {
    let mut out = Vec::with_capacity(ell);
    let mut n = BLOWUP * (1usize << ell);
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
    pub salts: Vec<Vec<u8>>,
}

pub fn commit<F: PrimeField + FftField, R: RngCore>(
    evals: Vec<F>,
    zk: bool,
    rng: &mut R,
) -> (Commitment, Witness<F>) {
    if !zk {
        return commit_public(evals);
    }
    let cw = rs_encode(&evals);
    let (leaf_bytes, salts) = make_leaf_bytes(&cw, zk, rng);
    finish_commit(evals, cw, leaf_bytes, salts)
}

/// Commit to public (non-hiding) data with no randomness.
pub fn commit_public<F: PrimeField + FftField>(
    evals: Vec<F>,
) -> (Commitment, Witness<F>) {
    let cw = rs_encode(&evals);
    let (leaf_bytes, salts) = make_leaf_bytes_public(&cw);
    finish_commit(evals, cw, leaf_bytes, salts)
}

fn finish_commit<F: PrimeField>(
    evals: Vec<F>,
    codeword: Vec<F>,
    leaf_bytes: Vec<Vec<u8>>,
    salts: Vec<Vec<u8>>,
) -> (Commitment, Witness<F>) {
    let tree = build_tree(&leaf_bytes);
    let root = tree.root();
    (Commitment { root }, Witness { evals, codeword, tree, salts })
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
    pub a_salts: Vec<Vec<u8>>,
    pub b_salts: Vec<Vec<u8>>,
}

#[derive(CanonicalSerialize)]
pub struct Proof<F: PrimeField> {
    /// Merkle roots h_1, …, h_{ell-1} (length ell-1), absorbed before query indices.
    pub intermediate_roots: Vec<Hash>,
    /// The collapsed constant f̃(r); absorbed after all intermediate roots.
    pub final_value: F,
    /// `num_queries()` query proofs.
    pub queries: Vec<QueryProof<F>>,
}

/// Intermediate codewords, commitments, and the final folded constant.
pub(crate) struct EvalData<F: PrimeField> {
    owned_codewords: Vec<Vec<F>>,
    owned_trees: Vec<MerkleTree<MerkleConfig>>,
    owned_salts: Vec<Vec<Vec<u8>>>,
    intermediate_roots: Vec<Hash>,
    final_value: F,
}

/// Fold c_0 → c_1 → … → c_ell (a genuine FRI fold on the extended domain each round).
/// Commit c_1..c_{ell-1}; the last fold yields the constant codeword c_ell, whose
/// (common) value is `final_value = f̃(r)`.
fn fold_and_commit<F, MK>(witness: &Witness<F>, r: &[F], mut make_leaves: MK) -> EvalData<F>
where
    F: PrimeField + FftField,
    MK: FnMut(&[F]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>),
{
    let ell = r.len();
    let mut owned_codewords: Vec<Vec<F>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut owned_trees: Vec<MerkleTree<MerkleConfig>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut owned_salts: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut intermediate_roots: Vec<Hash> = Vec::with_capacity(ell.saturating_sub(1));

    let mut cur = witness.codeword.clone();
    let mut final_value = F::zero();

    for (i, &ri) in r.iter().enumerate() {
        let next = fold_codeword(&cur, ri);
        if i + 1 < ell {
            let (leaf_bytes, salts) = make_leaves(&next);
            let tree = build_tree(&leaf_bytes);
            intermediate_roots.push(tree.root());
            owned_codewords.push(next.clone());
            owned_trees.push(tree);
            owned_salts.push(salts);
            cur = next;
        } else {
            final_value = next[0];
        }
    }

    EvalData { owned_codewords, owned_trees, owned_salts, intermediate_roots, final_value }
}

pub(crate) fn build_eval<F: PrimeField + FftField, R: RngCore>(
    witness: &Witness<F>,
    r: &[F],
    zk: bool,
    rng: &mut R,
) -> EvalData<F> {
    if !zk {
        return build_eval_public(witness, r);
    }
    fold_and_commit(witness, r, |cw| make_leaf_bytes(cw, true, &mut *rng))
}

pub(crate) fn build_eval_public<F: PrimeField + FftField>(
    witness: &Witness<F>,
    r: &[F],
) -> EvalData<F> {
    fold_and_commit(witness, r, make_leaf_bytes_public)
}

/// Bind the fold commitments to the transcript and open random paths through them.
pub(crate) fn finalize_eval<F: PrimeField + FftField>(
    witness: &Witness<F>,
    ell: usize,
    data: EvalData<F>,
    transcript: &mut Transcript,
) -> Proof<F> {
    for root in &data.intermediate_roots {
        transcript.absorb(root);
    }
    transcript.absorb(&field_to_bytes(data.final_value));

    let n0 = witness.codeword.len();
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    let codeword = |i: usize| -> &[F] {
        if i == 0 { &witness.codeword } else { &data.owned_codewords[i - 1] }
    };
    let tree = |i: usize| -> &MerkleTree<MerkleConfig> {
        if i == 0 { &witness.tree } else { &data.owned_trees[i - 1] }
    };
    let salts = |i: usize| -> &[Vec<u8>] {
        if i == 0 { &witness.salts } else { &data.owned_salts[i - 1] }
    };

    let mut queries = Vec::with_capacity(num_queries());
    for &s in &query_lows {
        let mut a_vals = Vec::with_capacity(ell);
        let mut b_vals = Vec::with_capacity(ell);
        let mut a_paths = Vec::with_capacity(ell);
        let mut b_paths = Vec::with_capacity(ell);
        let mut a_salts = Vec::with_capacity(ell);
        let mut b_salts = Vec::with_capacity(ell);

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
            a_salts.push(salts(i)[low_i].clone());
            b_salts.push(salts(i)[low_i + half].clone());
            low = low_i;
            n_i = half;
        }
        queries.push(QueryProof { a_vals, b_vals, a_paths, b_paths, a_salts, b_salts });
    }

    Proof {
        intermediate_roots: data.intermediate_roots,
        final_value: data.final_value,
        queries,
    }
}

pub fn prove_eval<F: PrimeField + FftField, R: RngCore>(
    witness: &Witness<F>,
    r: &[F],
    transcript: &mut Transcript,
    zk: bool,
    rng: &mut R,
) -> Result<(F, Proof<F>), PcsError> {
    let data = build_eval(witness, r, zk, rng);
    let proof = finalize_eval(witness, r.len(), data, transcript);
    Ok((proof.final_value, proof))
}

pub fn verify_eval<F: PrimeField + FftField>(
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

    for root in &proof.intermediate_roots {
        transcript.absorb(root);
    }
    transcript.absorb(&field_to_bytes(proof.final_value));

    let domains = layer_domains::<F>(ell)?;
    let n0 = domains[0].0;
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    for (qi, &s) in query_lows.iter().enumerate() {
        let qp = &proof.queries[qi];
        if qp.a_vals.len() != ell
            || qp.b_vals.len() != ell
            || qp.a_paths.len() != ell
            || qp.b_paths.len() != ell
            || qp.a_salts.len() != ell
            || qp.b_salts.len() != ell
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

            let root_i = if i == 0 { &commitment.root } else { &proof.intermediate_roots[i - 1] };
            if !verify_leaf(a, &qp.a_salts[i], low_i, &qp.a_paths[i], root_i) {
                return Ok(false);
            }
            if !verify_leaf(b, &qp.b_salts[i], low_i + half, &qp.b_paths[i], root_i) {
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
    salt: &[u8],
    expected_idx: usize,
    path: &Path<MerkleConfig>,
    root: &Hash,
) -> bool {
    if path.leaf_index != expected_idx {
        return false;
    }
    let mut bytes = field_to_bytes(val);
    bytes.extend_from_slice(salt);
    path.verify(&(), &(), root, bytes.as_slice()).unwrap_or(false)
}

// The ZK path opens ŵ = w̃ + B and B through one fold chain for ŵ + βB. This avoids
// exposing two subtractable intermediate chains. The base layer still opens both
// codewords, so `snark.rs` adds random unconstrained witness slots and sizes them
// using `opened_symbols_bound`.

/// One query of a batched proof: the base layer opens the ± pair of each base codeword
/// (ŵ then B); deeper layers open the ± pair of the combined codeword.
#[derive(CanonicalSerialize)]
pub struct CombinedQuery<F: PrimeField> {
    pub w0_a: F,
    pub w0_b: F,
    pub w1_a: F,
    pub w1_b: F,
    pub w0_a_path: Path<MerkleConfig>,
    pub w0_b_path: Path<MerkleConfig>,
    pub w1_a_path: Path<MerkleConfig>,
    pub w1_b_path: Path<MerkleConfig>,
    pub w0_a_salt: Vec<u8>,
    pub w0_b_salt: Vec<u8>,
    pub w1_a_salt: Vec<u8>,
    pub w1_b_salt: Vec<u8>,
    pub a_vals: Vec<F>,
    pub b_vals: Vec<F>,
    pub a_paths: Vec<Path<MerkleConfig>>,
    pub b_paths: Vec<Path<MerkleConfig>>,
    pub a_salts: Vec<Vec<u8>>,
    pub b_salts: Vec<Vec<u8>>,
}

#[derive(CanonicalSerialize)]
pub struct CombinedProof<F: PrimeField> {
    /// Claimed terminal values eval0 = ŵ(r), eval1 = B(r); absorbed before β.
    pub eval0: F,
    pub eval1: F,
    /// Merkle roots of the combined codeword's layers 1..ell-1 (length ell-1).
    pub intermediate_roots: Vec<Hash>,
    /// The collapsed constant ŵ(r) + β·B(r).
    pub final_value: F,
    pub queries: Vec<CombinedQuery<F>>,
}

/// Fold a codeword all the way down `r` and return the collapsed constant.
fn fold_all<F: FftField>(codeword: &[F], r: &[F]) -> F {
    let mut cur = codeword.to_vec();
    for &ri in r {
        cur = fold_codeword(&cur, ri);
    }
    cur[0]
}

/// Prove ŵ(r) and B(r) jointly via one fold chain on ŵ + β·B.
/// `wit0` commits ŵ, `wit1` commits B (both hiding, over the same domain).
pub fn prove_eval_combined<F: PrimeField + FftField, R: RngCore>(
    wit0: &Witness<F>,
    wit1: &Witness<F>,
    r: &[F],
    transcript: &mut Transcript,
    rng: &mut R,
) -> Result<(F, F, CombinedProof<F>), PcsError> {
    let ell = r.len();
    let n0 = wit0.codeword.len();
    debug_assert_eq!(n0, wit1.codeword.len());

    let eval0 = fold_all(&wit0.codeword, r);
    let eval1 = fold_all(&wit1.codeword, r);
    transcript.absorb_field(eval0);
    transcript.absorb_field(eval1);
    let beta: F = transcript.squeeze_field();

    let mut cur: Vec<F> =
        wit0.codeword.iter().zip(&wit1.codeword).map(|(&a, &b)| a + beta * b).collect();
    let mut comb_codewords: Vec<Vec<F>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut comb_trees: Vec<MerkleTree<MerkleConfig>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut comb_salts: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ell.saturating_sub(1));
    let mut intermediate_roots: Vec<Hash> = Vec::with_capacity(ell.saturating_sub(1));
    let mut final_value = F::zero();
    for (i, &ri) in r.iter().enumerate() {
        let next = fold_codeword(&cur, ri);
        if i + 1 < ell {
            let (leaf_bytes, salts) = make_leaf_bytes(&next, true, &mut *rng);
            let tree = build_tree(&leaf_bytes);
            intermediate_roots.push(tree.root());
            comb_codewords.push(next.clone());
            comb_trees.push(tree);
            comb_salts.push(salts);
            cur = next;
        } else {
            final_value = next[0];
        }
    }

    for root in &intermediate_roots {
        transcript.absorb(root);
    }
    transcript.absorb(&field_to_bytes(final_value));
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    let mut queries = Vec::with_capacity(num_queries());
    for &s in &query_lows {
        let half0 = n0 / 2;
        let low0 = s % half0;

        let mut a_vals = Vec::with_capacity(ell - 1);
        let mut b_vals = Vec::with_capacity(ell - 1);
        let mut a_paths = Vec::with_capacity(ell - 1);
        let mut b_paths = Vec::with_capacity(ell - 1);
        let mut a_salts = Vec::with_capacity(ell - 1);
        let mut b_salts = Vec::with_capacity(ell - 1);

        let mut low = low0;
        let mut n_i = half0;
        for li in 0..ell.saturating_sub(1) {
            let half = n_i / 2;
            let low_i = low % half;
            let cw = &comb_codewords[li];
            a_vals.push(cw[low_i]);
            b_vals.push(cw[low_i + half]);
            a_paths.push(comb_trees[li].generate_proof(low_i).unwrap());
            b_paths.push(comb_trees[li].generate_proof(low_i + half).unwrap());
            a_salts.push(comb_salts[li][low_i].clone());
            b_salts.push(comb_salts[li][low_i + half].clone());
            low = low_i;
            n_i = half;
        }

        queries.push(CombinedQuery {
            w0_a: wit0.codeword[low0],
            w0_b: wit0.codeword[low0 + half0],
            w1_a: wit1.codeword[low0],
            w1_b: wit1.codeword[low0 + half0],
            w0_a_path: wit0.tree.generate_proof(low0).unwrap(),
            w0_b_path: wit0.tree.generate_proof(low0 + half0).unwrap(),
            w1_a_path: wit1.tree.generate_proof(low0).unwrap(),
            w1_b_path: wit1.tree.generate_proof(low0 + half0).unwrap(),
            w0_a_salt: wit0.salts[low0].clone(),
            w0_b_salt: wit0.salts[low0 + half0].clone(),
            w1_a_salt: wit1.salts[low0].clone(),
            w1_b_salt: wit1.salts[low0 + half0].clone(),
            a_vals,
            b_vals,
            a_paths,
            b_paths,
            a_salts,
            b_salts,
        });
    }

    Ok((eval0, eval1, CombinedProof { eval0, eval1, intermediate_roots, final_value, queries }))
}

/// Verify a batched proof, binding `proof.eval0 = ŵ(r)` and `proof.eval1 = B(r)` to
/// the commitments `comm0` (ŵ) and `comm1` (B).
pub fn verify_eval_combined<F: PrimeField + FftField>(
    comm0: &Commitment,
    comm1: &Commitment,
    r: &[F],
    proof: &CombinedProof<F>,
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

    transcript.absorb_field(proof.eval0);
    transcript.absorb_field(proof.eval1);
    let beta: F = transcript.squeeze_field();
    for root in &proof.intermediate_roots {
        transcript.absorb(root);
    }
    transcript.absorb(&field_to_bytes(proof.final_value));

    if proof.final_value != proof.eval0 + beta * proof.eval1 {
        return Ok(false);
    }

    let domains = layer_domains::<F>(ell)?;
    let n0 = domains[0].0;
    let query_lows = transcript.squeeze_indices(n0 / 2, num_queries());

    for (qi, &s) in query_lows.iter().enumerate() {
        let q = &proof.queries[qi];
        if q.a_vals.len() != ell - 1
            || q.b_vals.len() != ell - 1
            || q.a_paths.len() != ell - 1
            || q.b_paths.len() != ell - 1
            || q.a_salts.len() != ell - 1
            || q.b_salts.len() != ell - 1
        {
            return Ok(false);
        }

        let (n0_, omega0) = domains[0];
        let half0 = n0_ / 2;
        let low0 = s % half0;
        if !verify_leaf(q.w0_a, &q.w0_a_salt, low0, &q.w0_a_path, &comm0.root)
            || !verify_leaf(q.w0_b, &q.w0_b_salt, low0 + half0, &q.w0_b_path, &comm0.root)
            || !verify_leaf(q.w1_a, &q.w1_a_salt, low0, &q.w1_a_path, &comm1.root)
            || !verify_leaf(q.w1_b, &q.w1_b_salt, low0 + half0, &q.w1_b_path, &comm1.root)
        {
            return Ok(false);
        }
        let comb_a = q.w0_a + beta * q.w1_a;
        let comb_b = q.w0_b + beta * q.w1_b;
        let mut folded = fold_pair(comb_a, comb_b, r[0], omega0, low0);
        let mut low = low0;

        for li in 0..ell - 1 {
            let (n_i, omega_i) = domains[li + 1];
            let half = n_i / 2;
            let low_i = low % half;
            let a = q.a_vals[li];
            let b = q.b_vals[li];
            let root_i = &proof.intermediate_roots[li];
            if !verify_leaf(a, &q.a_salts[li], low_i, &q.a_paths[li], root_i)
                || !verify_leaf(b, &q.b_salts[li], low_i + half, &q.b_paths[li], root_i)
            {
                return Ok(false);
            }
            let matched = if low < half { a } else { b };
            if matched != folded {
                return Ok(false);
            }
            folded = fold_pair(a, b, r[li + 1], omega_i, low_i);
            low = low_i;
        }

        if folded != proof.final_value {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::r1cs::mle_of_vector;
    use crate::transcript::Transcript;

    fn rng() -> StdRng { StdRng::seed_from_u64(123) }

    #[test]
    fn test_fold_collapses_to_mle_eval() {
        let mut rng = rng();
        for ell in 1..=5usize {
            let evals: Vec<F> = (0..(1u64 << ell)).map(|_| F::rand(&mut rng)).collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
            let (_c, witness) = commit_public(evals.clone());
            let data = build_eval_public(&witness, &r);
            assert_eq!(data.final_value, mle_of_vector(&evals, ell, &r), "ell={ell}");
        }
    }

    #[test]
    fn test_round_trip_non_zk() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];

        let (commitment, witness) = commit(evals, false, &mut rng);
        let mut pt = Transcript::new(b"pcs-test");
        let mut pv = Transcript::new(b"pcs-test");

        let (v, proof) = prove_eval(&witness, &r, &mut pt, false, &mut rng).unwrap();
        assert!(verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_round_trip_zk() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(5u64), F::from(7u64), F::from(9u64)];

        let (commitment, witness) = commit(evals, true, &mut rng);
        let mut pt = Transcript::new(b"pcs-test-zk");
        let mut pv = Transcript::new(b"pcs-test-zk");

        let (v, proof) = prove_eval(&witness, &r, &mut pt, true, &mut rng).unwrap();
        assert!(verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_wrong_claimed_value_rejected() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];

        let (commitment, witness) = commit(evals, false, &mut rng);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, proof) = prove_eval(&witness, &r, &mut pt, false, &mut rng).unwrap();

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v + F::from(1u64), &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_forged_value_rejected() {
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let (commitment, witness) = commit_public(evals);

        let mut data = build_eval_public(&witness, &r);
        let true_v = data.final_value;
        let forged_v = true_v + F::from(1u64);
        data.final_value = forged_v;

        let mut pt = Transcript::new(b"forge");
        let proof = finalize_eval(&witness, r.len(), data, &mut pt);

        let mut vt = Transcript::new(b"forge");
        assert!(!verify_eval(&commitment, &r, forged_v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_combined_round_trip() {
        let mut rng = rng();
        for ell in 1..=5usize {
            let w0: Vec<F> = (0..(1u64 << ell)).map(|_| F::rand(&mut rng)).collect();
            let w1: Vec<F> = (0..(1u64 << ell)).map(|_| F::rand(&mut rng)).collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();

            let (c0, wit0) = commit(w0.clone(), true, &mut rng);
            let (c1, wit1) = commit(w1.clone(), true, &mut rng);

            let mut pt = Transcript::new(b"comb");
            let (e0, e1, proof) =
                prove_eval_combined(&wit0, &wit1, &r, &mut pt, &mut rng).unwrap();
            assert_eq!(e0, mle_of_vector(&w0, ell, &r), "eval0 ell={ell}");
            assert_eq!(e1, mle_of_vector(&w1, ell, &r), "eval1 ell={ell}");

            let mut vt = Transcript::new(b"comb");
            assert!(
                verify_eval_combined(&c0, &c1, &r, &proof, &mut vt).unwrap(),
                "valid combined proof rejected ell={ell}"
            );

            let mut bad = proof;
            bad.eval0 += F::from(1u64);
            let mut vt2 = Transcript::new(b"comb");
            assert!(
                !verify_eval_combined(&c0, &c1, &r, &bad, &mut vt2).unwrap(),
                "forged eval0 accepted ell={ell}"
            );
        }
    }

    #[test]
    fn test_tampered_leaf_rejected() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];
        let (commitment, witness) = commit(evals, false, &mut rng);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt, false, &mut rng).unwrap();

        proof.queries[0].a_vals[0] += F::from(1u64);

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }
}
