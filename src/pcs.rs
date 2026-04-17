/// WHIR polynomial commitment scheme — post-fold single-query-phase.
///
/// Protocol (for a multilinear polynomial f̃ over ell variables, claimed value v = f̃(r)):
///
/// Commit phase (ell-1 rounds):
///   For i = 0 .. ell-2:
///     1. Fold table[i] with challenge r[i] → table[i+1]  (halves the table).
///     2. RS-encode table[i+1] into a shrinking domain of size BLOWUP·2^(ell-i-1),
///        build Merkle tree h_{i+1}, and absorb its root into the transcript.
///   After all folds: absorb final_value = (1-r[ell-1])·table[ell-1][0] + r[ell-1]·table[ell-1][1].
///
/// Query phase (single post-fold phase, t = NUM_QUERIES queries):
///   For each query position z ∈ [0, 2^(ell-1)):
///     - Open h_0 at the initial pair (table[0][2z], table[0][2z+1]) with two authentication paths.
///     - For each subsequent level i = 1..ell-1: send the sibling value
///       table[i][(z>>(i-1))^1] with an authentication path against h_i.
///     - Verifier folds the initial pair and each sibling recursively using r[0]..r[ell-1]
///       and checks the result equals final_value.
///
/// With shrinking domain the RS rate stays constant at 1/BLOWUP across all rounds
/// (folding_factor = 1), so t = NUM_QUERIES = 44 queries suffice for every round.
use ark_crypto_primitives::merkle_tree::{MerkleTree, Path};
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use rand::RngCore;

use crate::{
    merkle::{
        build_tree, field_to_bytes, make_leaf_bytes, Hash, MerkleConfig, BLOWUP, NUM_QUERIES,
    },
    transcript::Transcript,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcsError {
    InvalidDomain,
    DivisionByZero,
}

// ---------------------------------------------------------------------------
// RS encoders
// ---------------------------------------------------------------------------

/// Encode a 2^ell evaluation table into an RS codeword of size BLOWUP·2^ell.
///
/// Invariant: ext[BLOWUP · j] = evals[j] for all j.
pub fn rs_encode<F: FftField>(evals: &[F]) -> Vec<F> {
    rs_encode_to_size(evals, BLOWUP * evals.len())
}

/// Encode `input` into a codeword over a domain of size `target_size`.
///
/// Invariant: result[stride · k] = input[k], where stride = target_size / input.len().
pub fn rs_encode_to_size<F: FftField>(input: &[F], target_size: usize) -> Vec<F> {
    debug_assert!(target_size >= input.len());
    debug_assert!(target_size.is_power_of_two());
    let domain =
        Radix2EvaluationDomain::<F>::new(input.len()).expect("field supports NTT of this size");
    let ext_domain =
        Radix2EvaluationDomain::<F>::new(target_size).expect("field supports NTT of target size");
    let mut coeffs = input.to_vec();
    domain.ifft_in_place(&mut coeffs);
    coeffs.resize(target_size, F::zero());
    ext_domain.fft_in_place(&mut coeffs);
    coeffs
}

// ---------------------------------------------------------------------------
// Commitment
// ---------------------------------------------------------------------------

pub struct WhirCommitment {
    pub root: Hash,
}

/// Prover witness.
///
/// `tables[0]` = original evals (systematic values), size 2^ell.
/// `trees[0]`  = M₀, built from the RS-extended codeword, size BLOWUP·2^ell.
/// `salts[0]`  = salt bytes for M₀ leaves (length BLOWUP·2^ell).
pub struct WhirWitness<F: PrimeField> {
    pub tables: Vec<Vec<F>>,
    pub trees: Vec<MerkleTree<MerkleConfig>>,
    pub salts: Vec<Vec<Vec<u8>>>,
}

pub fn whir_commit<F: PrimeField + FftField, R: RngCore>(
    evals: Vec<F>,
    zk: bool,
    rng: &mut R,
) -> (WhirCommitment, WhirWitness<F>) {
    let ext = rs_encode(&evals);
    let (leaf_bytes, salts) = make_leaf_bytes(&ext, zk, rng);
    let tree = build_tree(&leaf_bytes);
    let root = tree.root();
    let commitment = WhirCommitment { root };
    let witness = WhirWitness {
        tables: vec![evals],
        trees: vec![tree],
        salts: vec![salts],
    };
    (commitment, witness)
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

/// Proof for a single query position z.
///
/// Contains the initial pair from h_0 (Merkle-proven) and one sibling per
/// fold level 1..ell-1, each Merkle-proven against the corresponding h_i.
pub struct WhirQueryProof<F: PrimeField> {
    /// (table[0][2z], table[0][2z+1]) — the initial pair from h_0.
    pub pair: (F, F),
    pub path_left: Path<MerkleConfig>,
    pub path_right: Path<MerkleConfig>,
    pub salt: (Vec<u8>, Vec<u8>),
    /// siblings[i] = table[i+1][(z>>i)^1]
    /// for i = 0..ell-2  (ell-1 entries total).
    pub siblings: Vec<F>,
    /// Merkle authentication path for each sibling against h_{i+1}.
    pub sibling_paths: Vec<Path<MerkleConfig>>,
    /// Salt bytes for each sibling leaf.
    pub sibling_salts: Vec<Vec<u8>>,
}

pub struct WhirProof<F: PrimeField> {
    /// Merkle roots h_1, ..., h_{ell-1}: absorbed into the transcript in order
    /// before query indices are derived.
    pub intermediate_roots: Vec<Hash>,
    /// f̃(r) — absorbed into transcript after all intermediate roots.
    pub final_value: F,
    /// NUM_QUERIES query proofs, all against h_0.
    pub queries: Vec<WhirQueryProof<F>>,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn whir_prove_eval<F: PrimeField + FftField, R: RngCore>(
    witness: &mut WhirWitness<F>,
    r: &[F],
    transcript: &mut Transcript,
    zk: bool,
    rng: &mut R,
) -> Result<(F, WhirProof<F>), PcsError> {
    let ell = r.len();

    // -----------------------------------------------------------------------
    // Phase 1: fold and commit.
    //
    // Build all ell-1 intermediate tables and trees.  Each tree encodes a
    // table of size 2^(ell-i-1) into a shrinking domain of BLOWUP·2^(ell-i-1),
    // keeping the RS rate constant at 1/BLOWUP.
    // -----------------------------------------------------------------------
    let mut owned_tables: Vec<Vec<F>> = Vec::with_capacity(ell - 1);
    let mut owned_trees: Vec<MerkleTree<MerkleConfig>> = Vec::with_capacity(ell - 1);
    let mut owned_salts: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ell - 1);
    let mut intermediate_roots: Vec<Hash> = Vec::with_capacity(ell - 1);

    let mut prev_table: Vec<F> = witness.tables[0].clone();

    for i in 0..ell - 1 {
        let ri = r[i];
        let half_i = prev_table.len() / 2;

        // Fold table[i] → table[i+1] using challenge r[i].
        let new_table: Vec<F> = (0..half_i)
            .map(|z| (F::one() - ri) * prev_table[2 * z] + ri * prev_table[2 * z + 1])
            .collect();

        // RS-encode into a shrinking domain (BLOWUP · new_table.len()),
        // build Merkle tree h_{i+1}, and absorb its root.
        let new_enc = rs_encode(&new_table);
        let (new_leaf_bytes, new_salts_vec) = make_leaf_bytes(&new_enc, zk, rng);
        let new_tree = build_tree(&new_leaf_bytes);
        let new_root = new_tree.root();

        transcript.absorb(&new_root);
        intermediate_roots.push(new_root);

        owned_tables.push(new_table.clone());
        owned_trees.push(new_tree);
        owned_salts.push(new_salts_vec);

        prev_table = new_table;
    }

    // Final scalar: fold the last 2-element table with r[ell-1].
    let last = &prev_table; // size 2
    let final_value = (F::one() - r[ell - 1]) * last[0] + r[ell - 1] * last[1];
    transcript.absorb(&field_to_bytes(final_value));

    // -----------------------------------------------------------------------
    // Phase 2: derive NUM_QUERIES query positions and build query proofs.
    //
    // All queries open h_0 (the initial commitment); fold consistency at each
    // subsequent level is certified by a sibling value with an authentication
    // path against the corresponding intermediate tree h_i.
    // -----------------------------------------------------------------------
    let half_0 = 1usize << (ell - 1); // pair positions in table[0]
    let query_positions = transcript.squeeze_indices(half_0, NUM_QUERIES);

    let initial_table = &witness.tables[0];
    let initial_tree = &witness.trees[0];
    let initial_salts = &witness.salts[0];

    let mut queries: Vec<WhirQueryProof<F>> = Vec::with_capacity(NUM_QUERIES);

    for &z in &query_positions {
        let leaf_l = BLOWUP * (2 * z);
        let leaf_r = BLOWUP * (2 * z + 1);

        let pair = (initial_table[2 * z], initial_table[2 * z + 1]);
        let path_left = initial_tree.generate_proof(leaf_l).unwrap();
        let path_right = initial_tree.generate_proof(leaf_r).unwrap();
        let salt = (initial_salts[leaf_l].clone(), initial_salts[leaf_r].clone());

        // Sibling at each fold level 1..ell-1.
        // At level `level`, the current fold value lives at index z>>(level-1) in
        // owned_tables[level-1]; its sibling is at (z>>(level-1)) ^ 1.
        let mut siblings = Vec::with_capacity(ell - 1);
        let mut sibling_paths = Vec::with_capacity(ell - 1);
        let mut sibling_salts = Vec::with_capacity(ell - 1);
        for level in 1..ell {
            let z_cur = z >> (level - 1);
            let z_sib = z_cur ^ 1;
            let sib_val = owned_tables[level - 1][z_sib];
            siblings.push(sib_val);

            let sib_leaf_idx = BLOWUP * z_sib;
            let sib_path = owned_trees[level - 1].generate_proof(sib_leaf_idx).unwrap();
            let sib_salt = owned_salts[level - 1][sib_leaf_idx].clone();
            sibling_paths.push(sib_path);
            sibling_salts.push(sib_salt);
        }

        queries.push(WhirQueryProof {
            pair,
            path_left,
            path_right,
            salt,
            siblings,
            sibling_paths,
            sibling_salts,
        });
    }

    Ok((
        final_value,
        WhirProof {
            intermediate_roots,
            final_value,
            queries,
        },
    ))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn whir_verify_eval<F: PrimeField + FftField>(
    commitment: &WhirCommitment,
    r: &[F],
    v: F,
    proof: &WhirProof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    let ell = r.len();

    if proof.intermediate_roots.len() != ell - 1 {
        return Ok(false);
    }
    if proof.queries.len() != NUM_QUERIES {
        return Ok(false);
    }

    // -----------------------------------------------------------------------
    // Absorb intermediate roots and final value (must mirror prover order).
    // -----------------------------------------------------------------------
    for root in &proof.intermediate_roots {
        transcript.absorb(root);
    }
    transcript.absorb(&field_to_bytes(proof.final_value));

    // -----------------------------------------------------------------------
    // Derive query positions and verify each query.
    // -----------------------------------------------------------------------
    let half_0 = 1usize << (ell - 1);
    let query_positions = transcript.squeeze_indices(half_0, NUM_QUERIES);

    let h0 = &commitment.root;

    for (idx, &z) in query_positions.iter().enumerate() {
        let qp = &proof.queries[idx];

        if qp.siblings.len() != ell - 1 {
            return Ok(false);
        }

        let leaf_l = BLOWUP * (2 * z);
        let leaf_r = BLOWUP * (2 * z + 1);

        // Verify the initial pair against h_0.
        if !verify_leaf(qp.pair.0, &qp.salt.0, leaf_l, &qp.path_left, h0) {
            return Ok(false);
        }
        if !verify_leaf(qp.pair.1, &qp.salt.1, leaf_r, &qp.path_right, h0) {
            return Ok(false);
        }

        // Fold recursively and verify each sibling against its intermediate root.
        let mut fold_val = (F::one() - r[0]) * qp.pair.0 + r[0] * qp.pair.1;

        for level in 1..ell {
            let sib_val = qp.siblings[level - 1];

            // Verify sibling against h_{level}.
            let h_level = &proof.intermediate_roots[level - 1];
            let z_cur = z >> (level - 1);
            let z_sib = z_cur ^ 1;
            let sib_leaf_idx = BLOWUP * z_sib;
            if !verify_leaf(
                sib_val,
                &qp.sibling_salts[level - 1],
                sib_leaf_idx,
                &qp.sibling_paths[level - 1],
                h_level,
            ) {
                return Ok(false);
            }

            // Reconstruct ordered pair: fold_val is at z_cur, sibling at z_cur^1.
            let (left, right) = if z_cur % 2 == 0 {
                (fold_val, sib_val)
            } else {
                (sib_val, fold_val)
            };

            fold_val = (F::one() - r[level]) * left + r[level] * right;
        }

        if fold_val != proof.final_value {
            return Ok(false);
        }
    }

    Ok(proof.final_value == v)
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use rand::{rngs::StdRng, SeedableRng};
    use crate::transcript::Transcript;

    fn rng() -> StdRng { StdRng::seed_from_u64(123) }

    #[test]
    fn test_rs_encode_systematic() {
        // Invariant: ext[BLOWUP * j] == evals[j] for all j.
        let evals: Vec<F> = (0..4u64).map(F::from).collect();
        let ext = rs_encode(&evals);
        assert_eq!(ext.len(), BLOWUP * evals.len());
        for (j, &e) in evals.iter().enumerate() {
            assert_eq!(ext[BLOWUP * j], e, "systematic property violated at j={j}");
        }
    }

    #[test]
    fn test_rs_encode_to_size() {
        // stride = target / input.len(); ext[stride * k] == input[k].
        let input: Vec<F> = (0..4u64).map(F::from).collect();
        let target = 16usize; // stride = 4
        let ext = rs_encode_to_size(&input, target);
        assert_eq!(ext.len(), target);
        let stride = target / input.len();
        for (k, &e) in input.iter().enumerate() {
            assert_eq!(ext[stride * k], e, "systematic property violated at k={k}");
        }
    }

    #[test]
    fn test_whir_round_trip_non_zk() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect(); // ell=2
        let r = vec![F::from(2u64), F::from(3u64)];

        let (commitment, mut witness) = whir_commit(evals, false, &mut rng);
        let mut pt = Transcript::new(b"whir-test");
        let mut pv = Transcript::new(b"whir-test");

        let (v, proof) = whir_prove_eval(&mut witness, &r, &mut pt, false, &mut rng).unwrap();
        assert!(whir_verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_whir_round_trip_zk() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect(); // ell=2
        let r = vec![F::from(5u64), F::from(7u64)];

        let (commitment, mut witness) = whir_commit(evals, true, &mut rng);
        let mut pt = Transcript::new(b"whir-test-zk");
        let mut pv = Transcript::new(b"whir-test-zk");

        let (v, proof) = whir_prove_eval(&mut witness, &r, &mut pt, true, &mut rng).unwrap();
        assert!(whir_verify_eval(&commitment, &r, v, &proof, &mut pv).unwrap());
    }

    #[test]
    fn test_whir_wrong_claimed_value_rejected() {
        let mut rng = rng();
        let evals: Vec<F> = (1..=4u64).map(F::from).collect();
        let r = vec![F::from(2u64), F::from(3u64)];

        let (commitment, mut witness) = whir_commit(evals, false, &mut rng);
        let mut pt = Transcript::new(b"whir-test");
        let (v, proof) = whir_prove_eval(&mut witness, &r, &mut pt, false, &mut rng).unwrap();

        let mut pv = Transcript::new(b"whir-test");
        assert!(!whir_verify_eval(&commitment, &r, v + F::from(1u64), &proof, &mut pv).unwrap());
    }
}
