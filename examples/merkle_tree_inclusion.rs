//! ZK proof of Merkle-tree leaf inclusion.
//!
//! Public  : the tree root (one field element).
//! Private : the secret leaf value and its authentication path.
//!
//! The tree uses Poseidon as both the leaf hash and the two-to-one
//! compression function (field-native, ~few hundred constraints per call).
//!
//! NOTE: ARK and MDS are generated via `find_poseidon_ark_and_mds` from the
//! BLS12-381 Fr modulus (alpha=5, 8 full rounds, 57 partial rounds, t=3).

use ark_crypto_primitives::{
    crh::{
        poseidon::{
            constraints::{CRHGadget, TwoToOneCRHGadget},
            CRH, TwoToOneCRH,
        },
        CRHSchemeGadget, TwoToOneCRHSchemeGadget,
    },
    merkle_tree::{
        constraints::{ConfigGadget, PathVar},
        Config, IdentityDigestConverter, MerkleTree, Path,
    },
    sponge::poseidon::{find_poseidon_ark_and_mds, PoseidonConfig},
};
use ark_bls12_381::Fr as F;
use ark_ff::{One, PrimeField, Zero};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar, prelude::Boolean};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use rand::{rngs::StdRng, SeedableRng};
use zk_snark::snark::{prove, verify};

// ---------------------------------------------------------------------------
// Merkle-tree configuration (native)
// ---------------------------------------------------------------------------

struct FieldMTConfig;

impl Config for FieldMTConfig {
    type Leaf = [F];
    type LeafDigest = F;
    type LeafInnerDigestConverter = IdentityDigestConverter<F>;
    type InnerDigest = F;
    type LeafHash = CRH<F>;
    type TwoToOneHash = TwoToOneCRH<F>;
}

type MT = MerkleTree<FieldMTConfig>;

// ---------------------------------------------------------------------------
// Merkle-tree configuration (R1CS gadget)
// ---------------------------------------------------------------------------

struct FieldMTConfigVar;

impl ConfigGadget<FieldMTConfig, F> for FieldMTConfigVar {
    type Leaf = [FpVar<F>];
    type LeafDigest = FpVar<F>;
    type LeafInnerConverter = IdentityDigestConverter<FpVar<F>>;
    type InnerDigest = FpVar<F>;
    type LeafHash = CRHGadget<F>;
    type TwoToOneHash = TwoToOneCRHGadget<F>;
}

// ---------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------

/// Depth of the tree (number of sibling digests on an auth path).
/// For NUM_LEAVES = 8 = 2^3 the depth is 3.
const TREE_DEPTH: usize = 3;
const NUM_LEAVES: usize = 1 << TREE_DEPTH; // 8

struct MerkleInclusionCircuit {
    /// Poseidon parameters — same for both prover and verifier.
    params: PoseidonConfig<F>,
    /// Tree root — public input.
    root: F,
    /// The secret leaf (a single field element) — private witness.
    leaf: Option<F>,
    /// Authentication path — private witness.
    auth_path: Option<Path<FieldMTConfig>>,
}

impl ConstraintSynthesizer<F> for MerkleInclusionCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let params = self.params;

        // --- Public input: root -------------------------------------------
        let root_var = FpVar::new_input(cs.clone(), || Ok(self.root))?;

        // --- Private witnesses --------------------------------------------
        let leaf_val = self.leaf.unwrap_or_default();
        let leaf_var = FpVar::new_witness(cs.clone(), || Ok(leaf_val))?;

        // Provide a dummy all-zero auth path of the correct depth for the
        // verifier (where self.auth_path == None).
        let dummy_auth_path = Path::<FieldMTConfig> {
            leaf_sibling_hash: F::zero(),
            auth_path: vec![F::zero(); TREE_DEPTH - 1],
            leaf_index: 0,
        };
        let auth_path_val = self.auth_path.unwrap_or(dummy_auth_path);
        let path_var = PathVar::<FieldMTConfig, F, FieldMTConfigVar>::new_witness(
            ark_relations::ns!(cs, "auth_path"),
            || Ok(&auth_path_val),
        )?;

        // --- Constant parameters (no constraints) -------------------------
        let leaf_params_var =
            <CRHGadget<F> as CRHSchemeGadget<CRH<F>, F>>::ParametersVar::new_constant(
                ark_relations::ns!(cs, "leaf_params"),
                &params,
            )?;
        let two_to_one_params_var = <TwoToOneCRHGadget<F> as TwoToOneCRHSchemeGadget<
            TwoToOneCRH<F>,
            F,
        >>::ParametersVar::new_constant(
            ark_relations::ns!(cs, "two_to_one_params"),
            &params,
        )?;

        // --- Verify membership -------------------------------------------
        // path_var.verify_membership(…) recomputes the root from leaf + path
        // and returns a Boolean gadget.
        let is_member = path_var.verify_membership(
            &leaf_params_var,
            &two_to_one_params_var,
            &root_var,
            &[leaf_var][..],
        )?;
        is_member.enforce_equal(&Boolean::TRUE)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let (ark, mds) = find_poseidon_ark_and_mds::<F>(
        F::MODULUS_BIT_SIZE as u64,
        2,  // rate
        8,  // full_rounds
        57, // partial_rounds (BLS12-381 Fr, t=3, 128-bit security)
        0,  // skip_matrices
    );
    let params = PoseidonConfig { full_rounds: 8, partial_rounds: 57, alpha: 5, ark, mds, rate: 2, capacity: 1 };

    // --- Build a Merkle tree over 8 field-element leaves ------------------
    let leaves: Vec<Vec<F>> = (0..NUM_LEAVES as u64)
        .map(|i| vec![F::from(i * 100 + 7)]) // non-trivial but deterministic
        .collect();

    let tree = MT::new(&params, &params, leaves.iter().map(|l| l.as_slice()))
        .expect("tree construction failed");
    let root = tree.root();

    // The prover wants to prove membership of the leaf at index 5.
    let secret_index = 5usize;
    let secret_leaf = leaves[secret_index][0];
    let auth_path = tree.generate_proof(secret_index).expect("proof generation failed");

    // Sanity-check the native path before proving.
    assert!(
        auth_path.verify(&params, &params, &root, [secret_leaf]).unwrap(),
        "native path verification failed"
    );

    // --- Prove -------------------------------------------------------------
    println!("Proving Merkle inclusion for leaf {secret_index} (ZK mode)…");
    let proof = prove(
        MerkleInclusionCircuit {
            params: params.clone(),
            root,
            leaf: Some(secret_leaf),
            auth_path: Some(auth_path),
        },
        true,
        &mut rng,
    ).expect("failed to prove");

    // --- Verify (correct root) --------------------------------------------
    let ok = verify(
        MerkleInclusionCircuit { params: params.clone(), root, leaf: None, auth_path: None },
        &proof,
    ).expect("failed to verify");
    println!("Correct root verified : {ok}");
    assert!(ok, "valid Merkle proof rejected");

    // --- Verify (wrong root — should be rejected) -------------------------
    let wrong_root = root + F::one();
    let rejected = !verify(
        MerkleInclusionCircuit {
            params: params.clone(),
            root: wrong_root,
            leaf: None,
            auth_path: None,
        },
        &proof,
    ).expect("failed to verify");
    println!("Wrong root rejected   : {rejected}");
    assert!(rejected, "tampered root was accepted");
}
