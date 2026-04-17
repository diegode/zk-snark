//! Merkle commitments and the Reed--Solomon parameters used by the PCS.
use ark_crypto_primitives::{
    crh::sha256::Sha256,
    merkle_tree::{ByteDigestConverter, Config, MerkleTree},
};
use ark_ff::PrimeField;
use rand::RngCore;

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

/// Target security level λ, in bits.
pub const SECURITY_BITS: usize = 128;

/// Proximity-check queries per fold round, `t = ⌈λ / log₂(2/(1+ρ))⌉`.
///
/// In the chapter's unique-decoding analysis, a query misses a forged layer with
/// probability at most `(1+ρ)/2`.
/// (BLOWUP_BITS = 3 ⇒ ρ = 1/8 ⇒ t = ⌈128/0.830⌉ = 155.)
pub fn num_queries() -> usize {
    let rho = 1.0 / BLOWUP as f64;
    let bits_per_query = (2.0 / (1.0 + rho)).log2();
    let target_bits = SECURITY_BITS as f64;
    (target_bits / bits_per_query).ceil() as usize
}

/// Domain-extension factor: `BLOWUP = 8`, hence rate `ρ = 1/8`.
pub const BLOWUP_BITS: usize = 3;
pub const BLOWUP: usize = 1 << BLOWUP_BITS;

pub(crate) fn make_leaf_bytes<F: PrimeField, R: RngCore>(
    evals: &[F],
    zk: bool,
    rng: &mut R,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    if !zk {
        return make_leaf_bytes_public(evals);
    }
    let mut salts = Vec::with_capacity(evals.len());
    let leaf_bytes = evals
        .iter()
        .map(|&v| {
            let mut bytes = field_to_bytes(v);
            let mut salt = vec![0u8; 16];
            rng.fill_bytes(&mut salt);
            bytes.extend_from_slice(&salt);
            salts.push(salt);
            bytes
        })
        .collect();
    (leaf_bytes, salts)
}

/// Deterministic leaves for public preprocessing data such as the R1CS matrices.
pub(crate) fn make_leaf_bytes_public<F: PrimeField>(evals: &[F]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let leaf_bytes = evals.iter().map(|&v| field_to_bytes(v)).collect();
    let salts = vec![Vec::new(); evals.len()];
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
    fn num_queries_meets_security_target() {
        let rho = 1.0 / BLOWUP as f64;
        let bits_per_query = (2.0 / (1.0 + rho)).log2();
        let target = SECURITY_BITS as f64;

        let t = num_queries();
        assert!(
            t as f64 * bits_per_query >= target,
            "{t} queries give {:.1} bits < {target} target",
            t as f64 * bits_per_query
        );
        assert!((t - 1) as f64 * bits_per_query < target);
    }
}
