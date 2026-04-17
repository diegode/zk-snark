//! Fold-and-commit multilinear PCS, following the chapter's BaseFold variant.
//! Chapter: Construction 10 (Evaluation protocol), analysed in Proposition 5.
//!
//! The Boolean evaluation table is Möbius-transformed to multilinear
//! coefficients, Reed--Solomon encoded, and Merkle committed. Folding the
//! codeword with challenge `r_i` partially evaluates variable `x_i`; after `ell`
//! folds the remaining constant is `f̃(r)`. The proximity test opens the `±y_i`
//! pair at each layer for `t` random draws of `y_1`. Each folded root is absorbed
//! before the next folding challenge. Matrix openings fold along their own
//! sparse-evaluation sumcheck; `prove_eval` opens at a known point with the
//! evaluation sumcheck of Construction 12, Step 4.
//!
//! Three choices of representation go beyond the chapter's wording. None of
//! them changes a challenge or a check of the verifier:
//! - `f̂^{(i)}(y)` and `f̂^{(i)}(−y)` are sibling leaves, so the two authentication
//!   paths of a pair share every node above the leaves.
//! - Tables of one size that fold with the same challenges (the sparse encoding
//!   of a matrix) share one tree per layer, whose leaf is the column of their
//!   symbols at one position, and the same draws. Each table still runs
//!   Construction 10; a tree over columns binds every table (Proposition 3), and
//!   each table's draws stay uniform, so Proposition 5 applies to each table.
//! - Each layer is opened once, at the distinct pairs that the draws reach, by
//!   a multi-opening (`merkle`).
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use crate::{
    merkle::{
        BLOWUP, Hash, MerkleDomain, MerkleTree, PrivacySalt, leaf_bytes, num_queries,
        verify_opening,
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
/// Remark 2 (Two perspectives on the committed table): these are the coefficients of
/// `f̂(X) = f̃(X, X², …, X^{2^{ell-1}})`.
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
/// `c_0[j] = f̂(ω_0^j)` where `f̂(X) = Σ_b coeffs[b]·X^b`, `coeffs = evals_to_coeffs(evals)`,
/// and `ω_0` generates the size-`BLOWUP·evals.len()` domain `D_0`.  Unlike a
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

/// The fold of Construction 10, step 2, at the pair `±x`, given `1/2` and `1/(2x)`:
///   f̂^{(i)}(x²) = (f̂^{(i-1)}(x) + f̂^{(i-1)}(−x))/2 + r_i·(f̂^{(i-1)}(x) − f̂^{(i-1)}(−x))/(2x).
fn fold_at<F: Field>(f_x: F, f_neg_x: F, r_i: F, inv2: F, inv2x: F) -> F {
    (f_x + f_neg_x) * inv2 + r_i * (f_x - f_neg_x) * inv2x
}

/// One fold of Construction 10, step 2: maps the codeword `f̂^{(i-1)}` on `D_{i-1}`
/// (`|D_{i-1}| = cw.len()`) to `f̂^{(i)}` on `D_i = {x² : x ∈ D_{i-1}}` (half the size),
/// with `x = ω^j` and `−x = ω^{j+half}`.
fn fold_codeword<F: FftField>(cw: &[F], r_i: F) -> Vec<F> {
    let n = cw.len();
    let half = n / 2;
    let omega = Radix2EvaluationDomain::<F>::new(n)
        .expect("field supports NTT of this size")
        .group_gen;
    let inv2 = F::from(2u64).inverse().expect("2 is invertible");
    let omega_inv = omega.inverse().expect("omega is invertible");

    let mut out = Vec::with_capacity(half);
    let mut inv2x = inv2;
    for j in 0..half {
        out.push(fold_at(cw[j], cw[j + half], r_i, inv2, inv2x));
        inv2x *= omega_inv;
    }
    out
}

/// `[ (|D_i|, generator of D_i) ]` for the `ell` fold layers, sizes `BLOWUP·2^ell, …, BLOWUP·2`.
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

/// The codeword index held by `leaf` in a layer of size `n`: leaf `2k` holds `ω^k`
/// and leaf `2k+1` holds `ω^{k+n/2} = −ω^k`, so each ± pair is a pair of sibling leaves.
fn column_index(leaf: usize, n: usize) -> usize {
    leaf / 2 + (leaf % 2) * (n / 2)
}

/// Merkle-commit one layer of a group of codewords, one leaf per column.
pub(crate) fn commit_layer<F: PrimeField>(
    codewords: &[Vec<F>],
    salts: Option<&[PrivacySalt]>,
    domain: &MerkleDomain,
) -> MerkleTree {
    let n = codewords[0].len();
    let leaf = |leaf: usize| {
        let index = column_index(leaf, n);
        let column: Vec<F> = codewords.iter().map(|cw| cw[index]).collect();
        domain.leaf(&leaf_bytes(&column, salts.map(|s| &s[leaf])))
    };
    MerkleTree::new(n, leaf, domain)
}

/// The distinct pairs `±y_i` that the draws reach in a layer with `half` pairs:
/// draw `y_1` reaches the pair at index `y_1 mod half`.
fn opened_pairs(draws: &[usize], half: usize) -> Vec<usize> {
    let mut pairs: Vec<usize> = draws.iter().map(|&y| y % half).collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// The sibling leaves of the given pairs, in increasing order.
fn pair_leaves(pairs: &[usize]) -> Vec<usize> {
    pairs.iter().flat_map(|&k| [2 * k, 2 * k + 1]).collect()
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment {
    pub root: Hash,
}

/// Prover data: the Boolean tables `f`, their RS codewords `f̂^{(0)}`, and the Merkle
/// tree of `h_0` over the codewords' columns.
pub struct Witness<F: PrimeField> {
    pub tables: Vec<Vec<F>>,
    pub codewords: Vec<Vec<F>>,
    pub tree: MerkleTree,
}

/// Commit to public (non-hiding) tables of one size with no randomness; the tables
/// share one tree.
pub fn commit_public<F: PrimeField + FftField>(tables: Vec<Vec<F>>) -> (Commitment, Witness<F>) {
    assert!(!tables.is_empty() && tables.iter().all(|t| t.len() == tables[0].len()));
    let codewords: Vec<_> = tables.par_iter().map(|t| rs_encode(t)).collect();
    let tree = commit_layer(&codewords, None, &MerkleDomain::public(0));
    (
        Commitment { root: tree.root() },
        Witness {
            tables,
            codewords,
            tree,
        },
    )
}

/// The proximity test's opening of one committed layer.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerOpening<F: PrimeField> {
    /// For each opened pair in increasing order: the column at `y_i`, then the column
    /// at `−y_i`, each holding one symbol per table.
    pub columns: Vec<F>,
    /// Private layers only: the salts of the opened leaves, in leaf order.
    pub salts: Vec<PrivacySalt>,
    /// The digests of the multi-opening (`merkle::MerkleTree::open`).
    pub siblings: Vec<Hash>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof<F: PrimeField> {
    /// Roots `h_1, …, h_{ell-1}`; `h_i` is absorbed right after `r_i`, before `r_{i+1}`.
    pub intermediate_roots: Vec<Hash>,
    /// `v`: each table's collapsed constant `f^{(ell)} = f̃(r)`, absorbed right after the
    /// last fold.
    pub final_values: Vec<F>,
    /// The openings of `h_0, …, h_{ell-1}`.
    pub layers: Vec<LayerOpening<F>>,
}

impl<F: PrimeField> Proof<F> {
    /// This is only transcript replay, not an evaluation proof verifier. The
    /// caller must also check the sumcheck terminal identity and the proximity test.
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
            for &v in &self.final_values {
                transcript.absorb_field(v);
            }
        }
        Some(())
    }
}

/// Incremental fold state for a group of codewords folded with the same challenges.
/// Never construct the whole chain from an already known point: callers absorb
/// each round before drawing the next challenge.
pub(crate) struct FoldProver<F: PrimeField> {
    initial: Vec<Vec<F>>,
    ell: usize,
    round: usize,
    /// Codewords and trees of the committed layers `h_1, …, h_{ell-1}`.
    layers: Vec<(Vec<Vec<F>>, MerkleTree)>,
    final_values: Vec<F>,
}

impl<F: PrimeField + FftField> FoldProver<F> {
    pub(crate) fn new(codewords: &[Vec<F>]) -> Result<Self, PcsError> {
        let n = codewords.first().ok_or(PcsError::InvalidShape)?.len();
        if codewords.iter().any(|cw| cw.len() != n) || !n.is_power_of_two() || n < 2 * BLOWUP {
            return Err(PcsError::InvalidShape);
        }
        let ell = (n / BLOWUP).ilog2() as usize;
        layer_domains::<F>(ell)?;
        Ok(Self {
            initial: codewords.to_vec(),
            ell,
            round: 0,
            layers: Vec::new(),
            final_values: Vec::new(),
        })
    }

    pub(crate) fn advance_public(&mut self, r_i: F) {
        self.advance(r_i, None);
    }

    pub(crate) fn advance_private(&mut self, r_i: F, salts: Option<&[PrivacySalt]>) {
        assert_eq!(salts.is_some(), self.round + 1 < self.ell);
        self.advance(r_i, salts);
    }

    /// Steps 2–3 of Construction 10: fold with `r_i`, then commit `h_i` (or keep `v`).
    fn advance(&mut self, r_i: F, salts: Option<&[PrivacySalt]>) {
        assert!(self.round < self.ell, "too many PCS folds");
        let current = self.layers.last().map_or(&self.initial, |(cws, _)| cws);
        let next: Vec<Vec<F>> = current
            .par_iter()
            .map(|cw| fold_codeword(cw, r_i))
            .collect();
        if self.round + 1 < self.ell {
            let domain = match salts {
                Some(_) => MerkleDomain::private(self.round + 1),
                None => MerkleDomain::public(self.round + 1),
            };
            let tree = commit_layer(&next, salts, &domain);
            self.layers.push((next, tree));
        } else {
            self.final_values = next.iter().map(|cw| cw[0]).collect();
        }
        self.round += 1;
    }

    pub(crate) fn absorb_latest(&self, transcript: &mut Transcript) {
        assert!(self.round > 0);
        if self.round < self.ell {
            transcript.absorb(&self.layers.last().unwrap().1.root());
        } else {
            for &v in &self.final_values {
                transcript.absorb_field(v);
            }
        }
    }

    /// Open the proximity test's queries once every fold message has been absorbed
    /// in its round. `salts[i]` holds the leaf salts of layer `i` of a private group.
    pub(crate) fn finish(
        self,
        witness: &Witness<F>,
        salts: Option<&[Vec<PrivacySalt>]>,
        transcript: &mut Transcript,
    ) -> Proof<F> {
        assert_eq!(self.round, self.ell, "unfinished PCS folds");
        let n0 = witness.codewords[0].len();
        // Index of y_1 in D_1, which is also the index of y_0 = √y_1 in D_0.
        let draws = transcript.squeeze_indices(n0 / 2, num_queries());
        let layers = (0..self.ell)
            .map(|i| {
                let (codewords, tree) = match i {
                    0 => (&witness.codewords, &witness.tree),
                    _ => (&self.layers[i - 1].0, &self.layers[i - 1].1),
                };
                let n = codewords[0].len();
                let leaves = pair_leaves(&opened_pairs(&draws, n / 2));
                LayerOpening {
                    columns: leaves
                        .iter()
                        .flat_map(|&leaf| codewords.iter().map(move |cw| cw[column_index(leaf, n)]))
                        .collect(),
                    salts: salts.map_or_else(Vec::new, |s| {
                        leaves.iter().map(|&leaf| s[i][leaf]).collect()
                    }),
                    siblings: tree.open(&leaves),
                }
            })
            .collect();
        Proof {
            intermediate_roots: self.layers.iter().map(|(_, tree)| tree.root()).collect(),
            final_values: self.final_values,
            layers,
        }
    }
}

/// Evaluation proof at a point already known to the prover (Construction 12, Step 4).
/// `folds.final_values[0]` is `f̃(s)` at the sumcheck's terminal point `s`, not the
/// claimed `f̃(r)`.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
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
        || witness.tables.len() != 1
        || witness.tables[0].len() != 1usize << ell
        || witness.tables[0].len().checked_mul(BLOWUP) != Some(witness.codewords[0].len())
    {
        return Err(PcsError::InvalidShape);
    }
    Ok(())
}

/// Step 4 of Construction 12: sumcheck of `Σ_z f(z)·eq(z, r) = v`, where each round
/// challenge `s_i` is also the fold challenge `r_i` of Construction 10.
fn prove_evaluation_sumcheck<F: PrimeField>(
    evals: &[F],
    r: &[F],
    transcript: &mut Transcript,
    mut after_challenge: impl FnMut(F, &mut Transcript),
) -> (F, SumcheckProof<F>) {
    let mut f_tbl = evals.to_vec();
    let mut eq_tbl = build_eq_table(r);
    let v: F = f_tbl.iter().zip(&eq_tbl).map(|(&f, &e)| f * e).sum();
    transcript.absorb_field(v);
    let mut round_polys = Vec::with_capacity(r.len());
    for _ in r {
        let half = f_tbl.len() / 2;
        let mut g_i = vec![F::zero(); 3];
        for k in 0..half {
            for (t, g_t) in g_i.iter_mut().enumerate() {
                let t = F::from(t as u64);
                *g_t += (f_tbl[2 * k] + t * (f_tbl[2 * k + 1] - f_tbl[2 * k]))
                    * (eq_tbl[2 * k] + t * (eq_tbl[2 * k + 1] - eq_tbl[2 * k]));
            }
        }
        for &g_t in &g_i {
            transcript.absorb_field(g_t);
        }
        let s_i = transcript.squeeze_field();
        after_challenge(s_i, transcript);
        round_polys.push(g_i);
        for k in 0..half {
            f_tbl[k] = f_tbl[2 * k] + s_i * (f_tbl[2 * k + 1] - f_tbl[2 * k]);
            eq_tbl[k] = eq_tbl[2 * k] + s_i * (eq_tbl[2 * k + 1] - eq_tbl[2 * k]);
        }
        f_tbl.truncate(half);
        eq_tbl.truncate(half);
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
    transcript.absorb(b"pcs-eval/interleaved-v2");
    transcript.absorb(&witness.tree.root());
    absorb_point(r, transcript);
    let mut folds = FoldProver::new(&witness.codewords)?;
    let (v, sc) =
        prove_evaluation_sumcheck(&witness.tables[0], r, transcript, |s_i, transcript| {
            folds.advance_public(s_i);
            folds.absorb_latest(transcript);
        });
    Ok((
        v,
        EvaluationProof {
            sc,
            folds: folds.finish(witness, None, transcript),
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
    if r.is_empty() || proof.folds.final_values.len() != 1 {
        return Ok(false);
    }
    transcript.absorb(b"pcs-eval/interleaved-v2");
    transcript.absorb(&commitment.root);
    absorb_point(r, transcript);
    let Some((s, final_claim)) =
        sumcheck_verify_interleaved(&proof.sc, r.len(), 2, v, transcript, |i, _, t| {
            proof.folds.absorb_round(i, r.len(), t)
        })
    else {
        return Ok(false);
    };
    // Step 4's terminal check: g_ell(s_ell) = f̃(s)·eq(s, r), with f̃(s) the folded constant.
    if final_claim != proof.folds.final_values[0] * eq_points(&s, r) {
        return Ok(false);
    }
    verify_proximity(&commitment.root, &s, 1, false, &proof.folds, transcript)
}

/// Proximity test of Construction 10 for a group of `width` tables committed together.
/// The caller must have replayed `Proof::absorb_round` between the challenges in `r`,
/// and must check each table's `final_values` entry against its claim.
pub(crate) fn verify_proximity<F: PrimeField + FftField>(
    root: &Hash,
    r: &[F],
    width: usize,
    private: bool,
    proof: &Proof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    let ell = r.len();
    if ell == 0
        || width == 0
        || proof.intermediate_roots.len() != ell - 1
        || proof.final_values.len() != width
        || proof.layers.len() != ell
    {
        return Ok(false);
    }
    let domains = layer_domains::<F>(ell)?;
    // Index of y_1 in D_1, which is also the index of y_0 = √y_1 in D_0.
    let draws = transcript.squeeze_indices(domains[0].0 / 2, num_queries());
    let inv2 = F::from(2u64).inverse().expect("2 is invertible");

    // Each check of a draw's path depends only on the pair it reaches, so each opened
    // pair is checked once. `replayed` holds the previous layer's pairs, each with the
    // fold g_i of every table, to be matched against f̂^{(i)}(y_i).
    let mut replayed: Option<(Vec<usize>, Vec<Vec<F>>)> = None;
    for (i, (&(n, omega), opening)) in domains.iter().zip(&proof.layers).enumerate() {
        let half = n / 2;
        let pairs = opened_pairs(&draws, half);
        let leaves = pair_leaves(&pairs);
        if opening.columns.len() != leaves.len() * width
            || opening.salts.len() != if private { leaves.len() } else { 0 }
        {
            return Ok(false);
        }
        let domain = match private {
            true => MerkleDomain::private(i),
            false => MerkleDomain::public(i),
        };
        let hashes: Vec<Hash> = opening
            .columns
            .chunks_exact(width)
            .enumerate()
            .map(|(j, column)| domain.leaf(&leaf_bytes(column, opening.salts.get(j))))
            .collect();
        let root = if i == 0 {
            root
        } else {
            &proof.intermediate_roots[i - 1]
        };
        if !verify_opening(
            &domain,
            root,
            n.ilog2() as usize,
            &leaves,
            &hashes,
            &opening.siblings,
        ) {
            return Ok(false);
        }
        let column = |leaf: usize| &opening.columns[leaf * width..(leaf + 1) * width];

        // A draw reaching pair k of layer i−1 has y_i at index k of D_i: check g_i = f̂^{(i)}(y_i).
        if let Some((previous, folds)) = &replayed {
            for (&k, g_i) in previous.iter().zip(folds) {
                let Ok(j) = pairs.binary_search(&(k % half)) else {
                    return Ok(false);
                };
                if column(2 * j + k / half) != g_i.as_slice() {
                    return Ok(false);
                }
            }
        }

        // g_{i+1} replayed from the opened pair f̂^{(i)}(±y_i), with y_i = ω^k.
        let omega_inv = omega.inverse().expect("omega is invertible");
        let folds = pairs
            .iter()
            .enumerate()
            .map(|(j, &k)| {
                let inv2x = inv2 * omega_inv.pow([k as u64]);
                column(2 * j)
                    .iter()
                    .zip(column(2 * j + 1))
                    .map(|(&f_x, &f_neg_x)| fold_at(f_x, f_neg_x, r[i], inv2, inv2x))
                    .collect()
            })
            .collect();
        replayed = Some((pairs, folds));
    }

    // The last replayed folds must equal the sent constants: g_ell = v.
    let (_, folds) = replayed.unwrap();
    Ok(folds.iter().all(|g| *g == proof.final_values))
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
            let tables: Vec<Vec<F>> = (0..3)
                .map(|_| (0..(1u64 << ell)).map(|_| F::rand(&mut rng)).collect())
                .collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
            let (_c, witness) = commit_public(tables.clone());
            let mut folds = FoldProver::new(&witness.codewords).unwrap();
            for &ri in &r {
                folds.advance_public(ri);
            }
            for (table, v) in tables.iter().zip(&folds.final_values) {
                assert_eq!(*v, mle_of_vector(table, ell, &r), "ell={ell}");
            }
        }
    }

    #[test]
    fn test_round_trip_non_zk() {
        let mut rng = rng();
        for ell in 1..=5 {
            let evals: Vec<F> = (0..1usize << ell).map(|_| F::rand(&mut rng)).collect();
            let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
            let expected = mle_of_vector(&evals, ell, &r);
            let (commitment, witness) = commit_public(vec![evals]);
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

        let (commitment, witness) = commit_public(vec![evals]);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, proof) = prove_eval(&witness, &r, &mut pt).unwrap();

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v + F::from(1u64), &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_forged_value_rejected() {
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let (commitment, witness) = commit_public(vec![evals]);

        let mut pt = Transcript::new(b"forge");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        proof.folds.final_values[0] += F::from(1u64);

        let mut vt = Transcript::new(b"forge");
        assert!(!verify_eval(&commitment, &r, v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_deferred_fold_commitments_rejected() {
        let evals: Vec<F> = (1..=8u64).map(F::from).collect();
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let (comm, witness) = commit_public(vec![evals]);
        let mut pt = Transcript::new(b"deferred");
        pt.absorb(b"pcs-eval/interleaved-v2");
        pt.absorb(&comm.root);
        absorb_point(&r, &mut pt);
        let mut folds = FoldProver::new(&witness.codewords).unwrap();
        // Omitting interleaved roots must invalidate the transcript even when
        // the fold computations themselves are correct.
        let (v, sc) = prove_evaluation_sumcheck(&witness.tables[0], &r, &mut pt, |s_i, _| {
            folds.advance_public(s_i);
        });
        for (_, tree) in &folds.layers {
            pt.absorb(&tree.root());
        }
        pt.absorb_field(folds.final_values[0]);
        let proof = EvaluationProof {
            sc,
            folds: folds.finish(&witness, None, &mut pt),
        };
        let mut vt = Transcript::new(b"deferred");
        assert!(!verify_eval(&comm, &r, v, &proof, &mut vt).unwrap());
    }

    #[test]
    fn test_fold_root_binds_next_challenge() {
        let (comm, witness) = commit_public(vec![(1..=8u64).map(F::from).collect()]);
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let mut pt = Transcript::new(b"schedule");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        let replay = |proof: &EvaluationProof<F>| {
            let mut t = Transcript::new(b"schedule");
            t.absorb(b"pcs-eval/interleaved-v2");
            t.absorb(&comm.root);
            absorb_point(&r, &mut t);
            let mut challenges = Vec::new();
            let _ = sumcheck_verify_interleaved(&proof.sc, r.len(), 2, v, &mut t, |i, s_i, t| {
                challenges.push(s_i);
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
        let (comm, witness) = commit_public(vec![(1..=8u64).map(F::from).collect()]);
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64)];
        let mut pt = Transcript::new(b"binding");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();
        let accepts = |c: &Commitment, point: &[F], p: &EvaluationProof<F>| {
            verify_eval(c, point, v, p, &mut Transcript::new(b"binding")).unwrap()
        };
        assert!(accepts(&comm, &r, &proof));
        // Every layer must answer all pairs that the t draws reach.
        for i in 0..r.len() {
            let column = proof.folds.layers[i].columns.pop().unwrap();
            assert!(!accepts(&comm, &r, &proof));
            proof.folds.layers[i].columns.push(column);
        }
        assert!(accepts(&comm, &r, &proof));
        let mut wrong_point = r.clone();
        wrong_point[0] += F::from(1u64);
        assert!(!accepts(&comm, &wrong_point, &proof));
        let mut wrong_comm = comm.clone();
        wrong_comm.root[0] ^= 1;
        assert!(!accepts(&wrong_comm, &r, &proof));
        let root = proof.folds.intermediate_roots.pop().unwrap();
        assert!(!accepts(&comm, &r, &proof));
        proof.folds.intermediate_roots.push(root);
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
        let (commitment, witness) = commit_public(vec![evals]);
        let mut pt = Transcript::new(b"pcs-test");
        let (v, mut proof) = prove_eval(&witness, &r, &mut pt).unwrap();

        proof.folds.layers[0].columns[0] += F::from(1u64);

        let mut pv = Transcript::new(b"pcs-test");
        assert!(!verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }

    #[test]
    fn pairs_are_sibling_leaves_and_each_layer_is_opened_once() {
        let mut rng = rng();
        let ell = 9;
        let (_, witness) = commit_public(vec![(0..1 << ell).map(|_| F::rand(&mut rng)).collect()]);
        let n = witness.codewords[0].len();
        // Leaves 2k and 2k+1 hold f̂(ω^k) and f̂(−ω^k).
        let omega = Radix2EvaluationDomain::<F>::new(n).unwrap().group_gen;
        for k in [0, 5, n / 2 - 1] {
            let (plus, minus) = (column_index(2 * k, n), column_index(2 * k + 1, n));
            assert_eq!(omega.pow([plus as u64]), -omega.pow([minus as u64]));
        }
        // A pair's two paths share every digest above the leaves: h − 1 of them.
        assert_eq!(witness.tree.open(&[6, 7]).len(), n.ilog2() as usize - 1);
        let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
        let (_, proof) = prove_eval(&witness, &r, &mut Transcript::new(b"once")).unwrap();
        for (i, layer) in proof.folds.layers.iter().enumerate() {
            let pairs = layer.columns.len() / 2;
            assert!(pairs <= num_queries().min(n >> (i + 1)));
            assert!(layer.siblings.len() <= pairs * ((n >> i).ilog2() as usize - 1));
        }
        // The deepest layers hold fewer pairs than there are draws, and are opened in full.
        assert_eq!(proof.folds.layers[ell - 1].columns.len(), BLOWUP * 2);
        assert!(proof.folds.layers[ell - 1].siblings.is_empty());
    }

    #[test]
    fn tables_sharing_trees_are_each_bound() {
        let mut rng = rng();
        let ell = 4;
        let tables: Vec<Vec<F>> = (0..3)
            .map(|_| (0..1 << ell).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let r: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();
        let (comm, witness) = commit_public(tables.clone());
        let prove = |t: &mut Transcript| {
            let mut folds = FoldProver::new(&witness.codewords).unwrap();
            for (i, &ri) in r.iter().enumerate() {
                t.absorb_field(F::from(i as u64));
                folds.advance_public(ri);
                folds.absorb_latest(t);
            }
            folds.finish(&witness, None, t)
        };
        let accepts = |proof: &Proof<F>| {
            let mut t = Transcript::new(b"group");
            for i in 0..ell {
                t.absorb_field(F::from(i as u64));
                proof.absorb_round(i, ell, &mut t).unwrap();
            }
            verify_proximity(&comm.root, &r, tables.len(), false, proof, &mut t).unwrap()
        };
        let mut proof = prove(&mut Transcript::new(b"group"));
        assert!(accepts(&proof));
        for (table, v) in tables.iter().zip(&proof.final_values) {
            assert_eq!(*v, mle_of_vector(table, ell, &r));
        }
        // Changing any one table's symbol in any layer, or its constant, is caught.
        for i in 0..ell {
            for table in 0..tables.len() {
                proof.layers[i].columns[table] += F::from(1u64);
                assert!(!accepts(&proof));
                proof.layers[i].columns[table] -= F::from(1u64);
            }
        }
        for table in 0..tables.len() {
            proof.final_values[table] += F::from(1u64);
            assert!(!accepts(&proof));
            proof.final_values[table] -= F::from(1u64);
        }
        // A group of a different width is a different statement.
        let mut t = Transcript::new(b"group");
        for i in 0..ell {
            t.absorb_field(F::from(i as u64));
            proof.absorb_round(i, ell, &mut t).unwrap();
        }
        assert!(!verify_proximity(&comm.root, &r, 2, false, &proof, &mut t).unwrap());
    }
}
