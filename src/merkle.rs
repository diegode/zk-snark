/// Merkle tree infrastructure, security parameters, and RS-domain constants.
///
/// Re-used by both `pcs.rs` (RS encoding and evaluation proofs) and the transcript.
use ark_crypto_primitives::{
    crh::sha256::Sha256,
    merkle_tree::{ByteDigestConverter, Config, MerkleTree},
};
use ark_ff::PrimeField;
use rand::RngCore;

// ---------------------------------------------------------------------------
// Merkle configuration
// ---------------------------------------------------------------------------

pub struct MerkleConfig;

impl Config for MerkleConfig {
    type Leaf = [u8];
    type LeafDigest = Vec<u8>;
    type LeafInnerDigestConverter = ByteDigestConverter<Vec<u8>>;
    type InnerDigest = Vec<u8>;
    type LeafHash = Sha256;
    type TwoToOneHash = Sha256;
}

pub type Hash = Vec<u8>;

pub fn field_to_bytes<F: PrimeField>(v: F) -> Vec<u8> {
    let mut buf = Vec::new();
    v.serialize_compressed(&mut buf).unwrap();
    buf
}

pub(crate) fn build_tree(leaves: &[Vec<u8>]) -> MerkleTree<MerkleConfig> {
    MerkleTree::<MerkleConfig>::new(&(), &(), leaves.iter().map(Vec::as_slice)).unwrap()
}

// ---------------------------------------------------------------------------
// Security / RS parameters
// ---------------------------------------------------------------------------

/// Number of proximity-check queries per fold round: `t = ⌈λ/b⌉` in the chapter.
///
/// The domain shrinks each round (BLOWUP·2^(ell-i) at round i), keeping the RS rate
/// constant at ρ = 1/BLOWUP = 2^{-BLOWUP_BITS} across all rounds.  With folding_factor=1
/// (one variable folded per round) the per-round soundness error is ρ^t, so t
/// queries suffice at every round.  With NUM_QUERIES=44 and BLOWUP_BITS=3 this gives
/// NUM_QUERIES·BLOWUP_BITS = 132 bits of per-round security; a union bound over up to 2^4 = 16
/// fold levels preserves ≥ 128-bit security overall.
pub const NUM_QUERIES: usize = 44;

/// Domain-extension factor: BLOWUP = 8 → rate ρ = 1/8, relative distance δ = 7/8.
/// BLOWUP_BITS = 3 is the standard choice in practice.
pub const BLOWUP_BITS: usize = 3;
pub const BLOWUP: usize = 1 << BLOWUP_BITS;

// ---------------------------------------------------------------------------
// Salted leaf helper
// ---------------------------------------------------------------------------

pub(crate) fn make_leaf_bytes<F: PrimeField, R: RngCore>(
    evals: &[F],
    zk: bool,
    rng: &mut R,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut salts = Vec::with_capacity(evals.len());
    let leaf_bytes = evals
        .iter()
        .map(|&v| {
            let mut bytes = field_to_bytes(v);
            if zk {
                let mut salt = vec![0u8; 16];
                rng.fill_bytes(&mut salt);
                bytes.extend_from_slice(&salt);
                salts.push(salt);
            } else {
                salts.push(vec![]);
            }
            bytes
        })
        .collect();
    (leaf_bytes, salts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::One;
    use ark_std::Zero;
    use rand::thread_rng;
    use ark_std::UniformRand;

    #[test]
    fn test_field_to_bytes() {
        let mut rng = thread_rng();
        let zero = Fr::zero();
        let one = Fr::one();
        let random_elem = Fr::rand(&mut rng);

        let bytes_zero = field_to_bytes(zero);
        let bytes_one = field_to_bytes(one);
        let bytes_random = field_to_bytes(random_elem);

        // Fr serialization size is 32 bytes
        assert_eq!(bytes_zero.len(), 32);
        assert_eq!(bytes_one.len(), 32);
        assert_eq!(bytes_random.len(), 32);

        // Determinism check
        let bytes_random_again = field_to_bytes(random_elem);
        assert_eq!(bytes_random, bytes_random_again);

        // Uniqueness check
        assert_ne!(bytes_zero, bytes_one);
        assert_ne!(bytes_zero, bytes_random);
        assert_ne!(bytes_one, bytes_random);
    }
}
