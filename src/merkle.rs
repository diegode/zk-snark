//! Merkle commitments and the Reed--Solomon parameters used by the PCS.
//! Chapter: Construction 9 (Merkle tree) and Proposition 3 (Merkle tree binding).
use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config, MerkleTree},
};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha2::{Digest, Sha256};
use std::borrow::Borrow;

// Leaf salts of Construction 13 (Masked sumcheck).
// Privacy budget: Q <= 2^64 oracle queries. For at most 2^64 total
// private leaves and at most 2 * num_queries() opened positions per tree, the
// hiding/programming terms sum_i q_i n_i Q (2^-s_MT + 2^-512) + Q 2^-256
// are below 2^-128. The last term uses the fresh 256-bit private root already
// included in every Fiat--Shamir input, without separate challenge salts.
pub const PRIVACY_QUERY_BOUND_BITS: usize = 64;
pub const PRIVACY_SALT_BYTES: usize = 40;
pub type PrivacySalt = [u8; PRIVACY_SALT_BYTES];

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct MerkleDomain {
    private: bool,
    layer: u64,
}

impl MerkleDomain {
    pub fn public(layer: usize) -> Self {
        Self {
            private: false,
            layer: layer as u64,
        }
    }

    pub(crate) fn private(layer: usize) -> Self {
        Self {
            private: true,
            layer: layer as u64,
        }
    }

    fn hasher(&self, role: &[u8]) -> Sha256 {
        let mut h = Sha256::new();
        h.update(b"zk-snark/merkle/v2");
        h.update([self.private as u8]);
        h.update(self.layer.to_le_bytes());
        h.update(role);
        h
    }
}

pub struct LeafHash;
pub struct NodeHash;

impl CRHScheme for LeafHash {
    type Input = [u8];
    type Output = Vec<u8>;
    type Parameters = MerkleDomain;

    fn setup<R: ark_std::rand::Rng>(
        _: &mut R,
    ) -> Result<MerkleDomain, ark_crypto_primitives::Error> {
        Ok(MerkleDomain::public(0))
    }

    fn evaluate<T: Borrow<[u8]>>(
        domain: &MerkleDomain,
        input: T,
    ) -> Result<Vec<u8>, ark_crypto_primitives::Error> {
        let mut h = domain.hasher(b"leaf");
        h.update((input.borrow().len() as u64).to_le_bytes());
        h.update(input.borrow());
        Ok(h.finalize().to_vec())
    }
}

impl TwoToOneCRHScheme for NodeHash {
    type Input = [u8];
    type Output = Vec<u8>;
    type Parameters = MerkleDomain;

    fn setup<R: ark_std::rand::Rng>(
        _: &mut R,
    ) -> Result<MerkleDomain, ark_crypto_primitives::Error> {
        Ok(MerkleDomain::public(0))
    }

    fn evaluate<T: Borrow<[u8]>>(
        domain: &MerkleDomain,
        left: T,
        right: T,
    ) -> Result<Vec<u8>, ark_crypto_primitives::Error> {
        let mut h = domain.hasher(b"node");
        for child in [left.borrow(), right.borrow()] {
            h.update((child.len() as u64).to_le_bytes());
            h.update(child);
        }
        Ok(h.finalize().to_vec())
    }

    fn compress<T: Borrow<Vec<u8>>>(
        domain: &MerkleDomain,
        left: T,
        right: T,
    ) -> Result<Vec<u8>, ark_crypto_primitives::Error> {
        Self::evaluate(domain, left.borrow().as_slice(), right.borrow().as_slice())
    }
}

pub struct MerkleConfig;

impl Config for MerkleConfig {
    type Leaf = [u8];
    type LeafDigest = Vec<u8>;
    type LeafInnerDigestConverter = ByteDigestConverter<Vec<u8>>;
    type InnerDigest = Vec<u8>;
    type LeafHash = LeafHash;
    type TwoToOneHash = NodeHash;
}

pub type Hash = Vec<u8>;

pub fn field_to_bytes<F: PrimeField>(v: F) -> Vec<u8> {
    let mut buf = Vec::new();
    v.serialize_compressed(&mut buf).unwrap();
    buf
}

pub(crate) fn build_tree(leaves: &[Vec<u8>]) -> MerkleTree<MerkleConfig> {
    build_tree_in_domain(leaves, &MerkleDomain::public(0))
}

pub(crate) fn build_tree_in_domain<L: AsRef<[u8]> + Send>(
    leaves: impl IntoIterator<Item = L>,
    domain: &MerkleDomain,
) -> MerkleTree<MerkleConfig> {
    MerkleTree::<MerkleConfig>::new(domain, domain, leaves).unwrap()
}

/// Salted leaf `m ‖ r`, hashed as `Com(m; r) = H(m ‖ r)` from Section 5.
pub(crate) fn private_leaf<F: PrimeField>(value: F, salt: &PrivacySalt) -> Vec<u8> {
    let mut bytes = field_to_bytes(value);
    bytes.extend_from_slice(salt);
    bytes
}

/// Target λ for the proximity-query term, not the complete SNARK error.
/// Field-size errors and the Fiat--Shamir compiler's loss are additional.
pub const SECURITY_BITS: usize = 128;

/// Query paths per opening: `t = ⌈λ / log₂(3/(2+ρ))⌉` (Construction 10, Evaluation protocol).
///
/// Uses proximity radius (1-ρ)/3 and the chapter's elementary weighted-folding
/// argument (Proposition 5), giving the query term `((2+ρ)/3)^t`.
/// At λ=128 and rate 1/8 this gives 258 queries. The additional field term
/// `O(N log N / |F|)` uses the Boolean-table length N after private padding.
pub fn num_queries() -> usize {
    let rho = 1.0 / BLOWUP as f64;
    let bits_per_query = (3.0 / (2.0 + rho)).log2();
    let target_bits = SECURITY_BITS as f64;
    (target_bits / bits_per_query).ceil() as usize
}

/// Domain-extension factor: `BLOWUP = 8`, hence rate `ρ = 1/8` (Definition 8, Reed–Solomon Code).
/// `BLOWUP_BITS` is `b = log₂(1/ρ)` in Construction 10.
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
        let t = num_queries();
        let target = 2.0_f64.powi(-(SECURITY_BITS as i32));
        let failure_per_query = (2.0 + 1.0 / BLOWUP as f64) / 3.0;
        assert_eq!(BLOWUP, 8);
        assert_eq!(t, 258);
        assert!(failure_per_query.powi(t as i32) <= target);
        assert!(failure_per_query.powi(t as i32 - 1) > target);
    }

    #[test]
    fn privacy_budget_covers_all_addressable_private_layers() {
        let opening_bits = (2 * num_queries()).next_power_of_two().ilog2() as usize;
        let overhead = PRIVACY_QUERY_BOUND_BITS + usize::BITS as usize + opening_bits;
        // Allocate a quarter of the privacy error to each of the two Merkle
        // terms and the private-root guessing term in the bound above.
        assert!(8 * PRIVACY_SALT_BYTES >= SECURITY_BITS + overhead + 2);
        assert!(512 >= SECURITY_BITS + overhead + 2);
        assert!(256 >= SECURITY_BITS + PRIVACY_QUERY_BOUND_BITS + 2);
    }

    #[test]
    fn hashes_separate_leaf_node_private_public_and_layer_domains() {
        let leaves = vec![vec![7; 32]; 8];
        let domains = [
            MerkleDomain::public(0),
            MerkleDomain::public(1),
            MerkleDomain::private(0),
            MerkleDomain::private(1),
        ];
        let roots: Vec<_> = domains
            .iter()
            .map(|d| build_tree_in_domain(&leaves, d).root())
            .collect();
        for i in 0..roots.len() {
            for j in 0..i {
                assert_ne!(roots[i], roots[j]);
            }
        }
        let tree = build_tree_in_domain(&leaves, &domains[2]);
        let path = tree.generate_proof(0).unwrap();
        assert!(
            path.verify(&domains[2], &domains[2], &tree.root(), leaves[0].as_slice())
                .unwrap()
        );
        assert!(
            !path
                .verify(&domains[3], &domains[3], &tree.root(), leaves[0].as_slice())
                .unwrap()
        );
        let bytes = [leaves[0].as_slice(), leaves[1].as_slice()].concat();
        assert_ne!(
            LeafHash::evaluate(&domains[0], &bytes[..]).unwrap(),
            NodeHash::evaluate(&domains[0], &leaves[0][..], &leaves[1][..]).unwrap()
        );
    }
}
