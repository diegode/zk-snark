//! Merkle commitments and the Reed--Solomon parameters used by the PCS.
//! Chapter: Construction 9 (Merkle tree) and Proposition 3 (Merkle tree binding).
//!
//! The tree uses Construction 9's heap layout: node 1 is the root, node `i` has
//! children `2i` and `2i+1`, and leaf `j` of `N = 2^h` is node `N + j`. A leaf
//! holds a column of field elements (one symbol of each table committed in the
//! tree) and, in private trees, a salt.
//!
//! An opening at several leaves sends their authentication paths as one
//! multi-opening: a sibling digest is sent only if neither an opened leaf nor a
//! node computed from them determines it. The verifier recomputes every omitted
//! digest, so the multi-opening carries the same information as the separate
//! paths of Construction 9.
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

// Leaf salts of Construction 13 (Masked sumcheck).
// Privacy budget: Q <= 2^64 oracle queries. For at most 2^64 total
// private leaves and at most 2 * num_queries() opened positions per tree, the
// hiding/programming terms sum_i q_i n_i Q (2^-s_MT + 2^-512) + Q 2^-256
// are below 2^-128. The last term uses the fresh 256-bit private root already
// included in every Fiat--Shamir input, without separate challenge salts.
pub const PRIVACY_QUERY_BOUND_BITS: usize = 64;
pub const PRIVACY_SALT_BYTES: usize = 40;
pub type PrivacySalt = [u8; PRIVACY_SALT_BYTES];

pub type Hash = [u8; 32];

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
        h.update(b"zk-snark/merkle/v3");
        h.update([self.private as u8]);
        h.update(self.layer.to_le_bytes());
        h.update(role);
        h
    }

    /// `node[N + j] = H(leaf_j)`, length-delimited and separated from inner nodes.
    pub fn leaf(&self, bytes: &[u8]) -> Hash {
        let mut h = self.hasher(b"leaf");
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
        h.finalize().into()
    }

    /// `node[i] = H(node[2i] ‖ node[2i+1])`.
    pub fn node(&self, left: &Hash, right: &Hash) -> Hash {
        let mut h = self.hasher(b"node");
        h.update(left);
        h.update(right);
        h.finalize().into()
    }
}

/// Construction 9: `nodes[i]` for `i = 1, …, 2N − 1`; `nodes[0]` is unused.
pub struct MerkleTree {
    nodes: Vec<Hash>,
}

impl MerkleTree {
    /// Build the tree over `n = 2^h` leaves, where `leaf(j)` hashes leaf `j`.
    pub fn new(n: usize, leaf: impl Fn(usize) -> Hash + Sync, domain: &MerkleDomain) -> Self {
        assert!(n.is_power_of_two(), "a Merkle tree needs 2^h leaves");
        let mut nodes = vec![[0; 32]; 2 * n];
        nodes[n..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, node)| *node = leaf(j));
        let mut width = n / 2;
        while width > 0 {
            let (upper, lower) = nodes.split_at_mut(2 * width);
            upper[width..]
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, parent)| *parent = domain.node(&lower[2 * k], &lower[2 * k + 1]));
            width /= 2;
        }
        Self { nodes }
    }

    pub fn root(&self) -> Hash {
        self.nodes[1]
    }

    /// The sibling digests of a multi-opening at sorted, distinct leaf `positions`,
    /// in the order `verify_opening` consumes them.
    pub fn open(&self, positions: &[usize]) -> Vec<Hash> {
        let leaves = self.nodes.len() / 2;
        let mut siblings = Vec::new();
        walk(leaves, positions, |node| {
            siblings.push(self.nodes[node]);
            Some(())
        });
        siblings
    }
}

/// Walk from the opened leaves to the root, level by level and left to right. At
/// each known node whose sibling is not known, `missing` is called with the
/// sibling's index. Returns the known nodes of each level, leaves first.
fn walk(
    leaves: usize,
    positions: &[usize],
    mut missing: impl FnMut(usize) -> Option<()>,
) -> Option<Vec<Vec<usize>>> {
    let mut level: Vec<usize> = positions.iter().map(|&p| leaves + p).collect();
    let mut levels = Vec::new();
    while level[0] > 1 {
        let mut parents = Vec::with_capacity(level.len());
        let mut i = 0;
        while i < level.len() {
            let node = level[i];
            if node.is_multiple_of(2) && level.get(i + 1) == Some(&(node + 1)) {
                i += 2;
            } else {
                missing(node ^ 1)?;
                i += 1;
            }
            parents.push(node / 2);
        }
        levels.push(level);
        level = parents;
    }
    levels.push(level);
    Some(levels)
}

/// Check a multi-opening of a tree with `2^height` leaves at sorted, distinct
/// `positions`, whose leaf hashes are `leaves`, against `root`. Every digest in
/// `siblings` must be used.
pub fn verify_opening(
    domain: &MerkleDomain,
    root: &Hash,
    height: usize,
    positions: &[usize],
    leaves: &[Hash],
    siblings: &[Hash],
) -> bool {
    let Some(n) = 1usize.checked_shl(height as u32) else {
        return false;
    };
    if positions.is_empty()
        || positions.len() != leaves.len()
        || positions.windows(2).any(|w| w[0] >= w[1])
        || positions[positions.len() - 1] >= n
    {
        return false;
    }
    let mut given = siblings.iter();
    let mut needed = Vec::new();
    let Some(levels) = walk(n, positions, |node| {
        needed.push((node, *given.next()?));
        Some(())
    }) else {
        return false;
    };
    if given.next().is_some() {
        return false;
    }
    // Replay the walk, now computing each level's hashes from the one below.
    let mut needed = needed.into_iter();
    let mut hashes: Vec<Hash> = leaves.to_vec();
    for (level, parents) in levels.iter().zip(&levels[1..]) {
        let mut next = Vec::with_capacity(parents.len());
        let mut i = 0;
        while i < level.len() {
            let node = level[i];
            if node.is_multiple_of(2) && level.get(i + 1) == Some(&(node + 1)) {
                next.push(domain.node(&hashes[i], &hashes[i + 1]));
                i += 2;
            } else {
                let (_, sibling) = needed.next().unwrap();
                next.push(if node.is_multiple_of(2) {
                    domain.node(&hashes[i], &sibling)
                } else {
                    domain.node(&sibling, &hashes[i])
                });
                i += 1;
            }
        }
        hashes = next;
    }
    hashes == [*root]
}

pub fn field_to_bytes<F: PrimeField>(v: F) -> Vec<u8> {
    let mut buf = Vec::new();
    v.serialize_compressed(&mut buf).unwrap();
    buf
}

/// A leaf's bytes: its column, then its salt in a private tree, so that the
/// leaf hash is `Com(m; r) = H(m ‖ r)` from Section 5.
pub(crate) fn leaf_bytes<F: PrimeField>(column: &[F], salt: Option<&PrivacySalt>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32 * column.len() + PRIVACY_SALT_BYTES);
    for &v in column {
        v.serialize_compressed(&mut bytes).unwrap();
    }
    if let Some(salt) = salt {
        bytes.extend_from_slice(salt);
    }
    bytes
}

/// Target λ for the proximity-query term, not the complete SNARK error.
/// Field-size errors and the Fiat--Shamir compiler's loss are additional.
pub const SECURITY_BITS: usize = 128;

/// Draws of `y_1` per opening: `t = ⌈λ / log₂(3/(2+ρ))⌉` (Construction 10, Evaluation protocol).
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::One;
    use ark_std::UniformRand;
    use ark_std::Zero;
    use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};

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

    fn tree(height: usize, domain: &MerkleDomain) -> (Vec<Hash>, MerkleTree) {
        let leaves: Vec<_> = (0..1u64 << height)
            .map(|i| domain.leaf(&i.to_le_bytes()))
            .collect();
        let tree = MerkleTree::new(leaves.len(), |j| leaves[j], domain);
        (leaves, tree)
    }

    #[test]
    fn heap_layout_matches_construction_9() {
        let domain = MerkleDomain::public(0);
        let (leaves, t) = tree(2, &domain);
        let left = domain.node(&leaves[0], &leaves[1]);
        let right = domain.node(&leaves[2], &leaves[3]);
        assert_eq!(t.root(), domain.node(&left, &right));
        // A single path carries the h siblings of Construction 9.
        assert_eq!(t.open(&[2]), vec![leaves[3], left]);
        assert!(verify_opening(
            &domain,
            &t.root(),
            2,
            &[2],
            &[leaves[2]],
            &[leaves[3], left]
        ));
        // Opening both leaves under one node needs only the other subtree.
        assert_eq!(t.open(&[0, 1]), vec![right]);
        let (_, single) = tree(0, &domain);
        assert!(single.open(&[0]).is_empty());
        assert!(verify_opening(
            &domain,
            &single.root(),
            0,
            &[0],
            &[single.root()],
            &[]
        ));
    }

    #[test]
    fn multi_openings_verify_and_bind_every_part() {
        let mut rng = StdRng::seed_from_u64(9);
        let domain = MerkleDomain::private(3);
        for height in [1, 4, 7] {
            let (leaves, t) = tree(height, &domain);
            for count in [1, 3, 1 << (height - 1), 1 << height] {
                let mut positions: Vec<usize> =
                    (0..count).map(|_| rng.gen_range(0..1 << height)).collect();
                positions.sort();
                positions.dedup();
                let opened: Vec<_> = positions.iter().map(|&p| leaves[p]).collect();
                let siblings = t.open(&positions);
                let accepts = |pos: &[usize], leaves: &[Hash], sib: &[Hash]| {
                    verify_opening(&domain, &t.root(), height, pos, leaves, sib)
                };
                assert!(accepts(&positions, &opened, &siblings));
                // The digests sent never exceed the separate paths' digests.
                assert!(siblings.len() <= height * positions.len());
                let mut bad = opened.clone();
                bad[0][0] ^= 1;
                assert!(!accepts(&positions, &bad, &siblings));
                for i in 0..siblings.len() {
                    let mut bad = siblings.clone();
                    bad[i][31] ^= 1;
                    assert!(!accepts(&positions, &opened, &bad));
                }
                let mut extra = siblings.clone();
                extra.push([0; 32]);
                assert!(!accepts(&positions, &opened, &extra));
                if let Some((_, fewer)) = siblings.split_last() {
                    assert!(!accepts(&positions, &opened, fewer));
                }
                assert!(!verify_opening(
                    &domain,
                    &t.root(),
                    height + 1,
                    &positions,
                    &opened,
                    &siblings
                ));
                assert!(!verify_opening(
                    &MerkleDomain::public(3),
                    &t.root(),
                    height,
                    &positions,
                    &opened,
                    &siblings
                ));
                if positions.len() > 1 {
                    let mut swapped = positions.clone();
                    swapped.swap(0, 1);
                    assert!(!accepts(&swapped, &opened, &siblings));
                }
                let last = *positions.last().unwrap();
                if last + 1 < 1 << height {
                    let mut moved = positions.clone();
                    *moved.last_mut().unwrap() += 1;
                    assert!(!accepts(&moved, &opened, &siblings));
                }
            }
            assert!(!verify_opening(&domain, &t.root(), height, &[], &[], &[]));
            assert!(!verify_opening(
                &domain,
                &t.root(),
                height,
                &[1 << height],
                &[leaves[0]],
                &t.open(&[0])
            ));
        }
    }

    #[test]
    fn hashes_separate_leaf_node_private_public_and_layer_domains() {
        let domains = [
            MerkleDomain::public(0),
            MerkleDomain::public(1),
            MerkleDomain::private(0),
            MerkleDomain::private(1),
        ];
        let leaf = [7; 32];
        let roots: Vec<_> = domains
            .iter()
            .map(|d| MerkleTree::new(8, |_| d.leaf(&leaf), d).root())
            .collect();
        for i in 0..roots.len() {
            for j in 0..i {
                assert_ne!(roots[i], roots[j]);
            }
        }
        let d = &domains[0];
        let bytes = [leaf, leaf].concat();
        assert_ne!(d.leaf(&bytes), d.node(&leaf, &leaf));
    }
}
