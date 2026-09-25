//! Merkle commitments and the Reed--Solomon parameters used by the PCS.
use ark_crypto_primitives::{
    crh::sha256::Sha256,
    merkle_tree::{ByteDigestConverter, Config, MerkleTree},
};
use ark_ff::PrimeField;

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

/// Security target λ in bits. Sets the query budget for all openings.
/// The complete protocol still needs a quantitative soundness analysis.
pub const SECURITY_BITS: usize = 128;

/// Query paths per opening: `t = ⌈λ / log₂(2/(1+ρ))⌉`.
///
/// The current selection rule gives 155 queries at λ=128 and rate 1/8.
/// It is a heuristic allocation, not a bound on the complete protocol's error.
/// ZK randomness and auxiliary-vector lengths scale with this query budget.
pub fn num_queries() -> usize {
    let rho = 1.0 / BLOWUP as f64;
    let bits_per_query = (2.0 / (1.0 + rho)).log2();
    let target_bits = SECURITY_BITS as f64;
    (target_bits / bits_per_query).ceil() as usize
}

/// Domain-extension factor: `BLOWUP = 8`, hence rate `ρ = 1/8`.
pub const BLOWUP_BITS: usize = 3;
pub const BLOWUP: usize = 1 << BLOWUP_BITS;

/// Deterministic leaves for public preprocessing data such as the R1CS matrices.
pub(crate) fn make_leaf_bytes_public<F: PrimeField>(evals: &[F]) -> Vec<Vec<u8>> {
    evals.iter().map(|&v| field_to_bytes(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::One;
    use ark_std::UniformRand;
    use ark_std::Zero;
    use rand::thread_rng;

    #[test]
    fn test_field_to_bytes() {
        let mut rng = thread_rng();
        let zero = Fr::zero();
        let one = Fr::one();
        let random_elem = Fr::rand(&mut rng);

        let bytes_zero = field_to_bytes(zero);
        let bytes_one = field_to_bytes(one);
        let bytes_random = field_to_bytes(random_elem);

        assert_eq!(bytes_zero.len(), 32);
        assert_eq!(bytes_one.len(), 32);
        assert_eq!(bytes_random.len(), 32);

        let bytes_random_again = field_to_bytes(random_elem);
        assert_eq!(bytes_random, bytes_random_again);

        assert_ne!(bytes_zero, bytes_one);
        assert_ne!(bytes_zero, bytes_random);
        assert_ne!(bytes_one, bytes_random);
    }

    #[test]
    fn query_budget_rounding() {
        let rho = 1.0 / BLOWUP as f64;
        let bits_per_query = (2.0 / (1.0 + rho)).log2();
        let target = SECURITY_BITS as f64;

        let t = num_queries();
        assert!(t as f64 * bits_per_query >= target);
        assert!((t - 1) as f64 * bits_per_query < target);
    }
}
