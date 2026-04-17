//! Regression checks for circuit compilation and PCS authentication shapes.
//! The hash-chain candidate below is rejected and does not demonstrate a PCS
//! binding failure.
use ark_bls12_381::Fr as F;
use ark_ff::{One, Zero};
use ark_relations::gr1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError, Variable,
    predicate::PredicateConstraintSystem,
};
use ark_serialize::CanonicalDeserialize;
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
use zk_snark::{
    merkle::{BLOWUP, Hash, MerkleDomain, MerkleTree, field_to_bytes, num_queries},
    pcs::{Commitment, EvaluationProof, LayerOpening, Proof, verify_eval},
    snark::{SnarkError, prove, setup, verify},
    sumcheck::SumcheckProof,
    transcript::Transcript,
};

#[derive(Clone)]
struct BooleanCircuit {
    output: F,
    witness: F,
    custom_predicate: bool,
}

impl ConstraintSynthesizer<F> for BooleanCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(self.output))?;
        let w = cs.new_witness_variable(|| Ok(self.witness))?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + w,
            || ark_relations::lc!() + Variable::One,
            || ark_relations::lc!() + out,
        )?;
        if !self.custom_predicate {
            return cs.enforce_r1cs_constraint(
                || ark_relations::lc!() + w,
                || ark_relations::lc!() + w - Variable::One,
                ark_relations::gr1cs::LinearCombination::zero,
            );
        }
        cs.register_predicate(
            "BooleanWitness",
            PredicateConstraintSystem::new_polynomial_predicate_cs(
                1,
                vec![(F::one(), vec![(0, 2)]), (-F::one(), vec![(0, 1)])],
            ),
        )?;
        cs.enforce_constraint(
            "BooleanWitness",
            vec![Box::new(move || ark_relations::lc!() + w)
                as Box<
                    dyn FnOnce() -> ark_relations::gr1cs::LinearCombination<F>,
                >],
        )
    }
}

#[test]
fn setup_rejects_non_r1cs_constraint() {
    let bad = BooleanCircuit {
        output: F::from(2),
        witness: F::from(2),
        custom_predicate: true,
    };
    let cs = ConstraintSystem::<F>::new_ref();
    bad.clone().generate_constraints(cs.clone()).unwrap();
    cs.finalize();
    assert_eq!(
        cs.is_satisfied().unwrap(),
        false,
        "the circuit itself must reject"
    );
    for zk in [false, true] {
        assert!(matches!(
            setup(BooleanCircuit { output: F::zero(), witness: F::zero(), custom_predicate: true }, zk),
            Err(SnarkError::UnsupportedPredicate(label)) if label == "BooleanWitness"
        ));
        let (pp, _) = setup(
            BooleanCircuit {
                output: F::zero(),
                witness: F::zero(),
                custom_predicate: false,
            },
            zk,
        )
        .unwrap();
        assert!(matches!(
            prove(&pp, bad.clone(), &mut StdRng::seed_from_u64(91)),
            Err(SnarkError::UnsupportedPredicate(label)) if label == "BooleanWitness"
        ));
    }
}

struct ReplacedR1csPredicate;

impl ConstraintSynthesizer<F> for ReplacedR1csPredicate {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(F::zero()))?;
        let w = cs.new_witness_variable(|| Ok(F::zero()))?;
        cs.register_predicate(
            "R1CS",
            PredicateConstraintSystem::new_polynomial_predicate_cs(
                3,
                vec![(F::one(), vec![(0, 1)]), (-F::one(), vec![(2, 1)])],
            ),
        )?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + w,
            || ark_relations::lc!() + Variable::One,
            || ark_relations::lc!() + out,
        )
    }
}

#[test]
fn setup_rejects_replaced_r1cs_predicate() {
    assert!(matches!(
        setup(ReplacedR1csPredicate, false),
        Err(SnarkError::InvalidR1csPredicate)
    ));
}

struct DifferentR1csCircuit;

impl ConstraintSynthesizer<F> for DifferentR1csCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let out = cs.new_input_variable(|| Ok(F::zero()))?;
        let w = cs.new_witness_variable(|| Ok(F::zero()))?;
        for _ in 0..2 {
            cs.enforce_r1cs_constraint(
                || ark_relations::lc!() + w,
                || ark_relations::lc!() + Variable::One,
                || ark_relations::lc!() + out,
            )?;
        }
        Ok(())
    }
}

#[test]
fn prove_rejects_changed_r1cs_relation() {
    let (pp, _) = setup(
        BooleanCircuit {
            output: F::zero(),
            witness: F::zero(),
            custom_predicate: false,
        },
        false,
    )
    .unwrap();
    assert!(matches!(
        prove(
            &pp,
            DifferentR1csCircuit,
            &mut StdRng::seed_from_u64(91)
        ),
        Err(SnarkError::CircuitMismatch)
    ));
}

#[test]
fn expressing_boolean_constraint_as_r1cs_rejects_false_statement() {
    for zk in [false, true] {
        let (pp, vp) = setup(
            BooleanCircuit {
                output: F::zero(),
                witness: F::zero(),
                custom_predicate: false,
            },
            zk,
        )
        .unwrap();
        for value in [F::zero(), F::one(), F::from(2)] {
            let circuit = BooleanCircuit {
                output: value,
                witness: value,
                custom_predicate: false,
            };
            let proof = prove(&pp, circuit, &mut StdRng::seed_from_u64(91)).unwrap();
            assert_eq!(
                verify(&vp, &[F::one(), value], &proof).unwrap(),
                value != F::from(2)
            );
        }
    }
}

/// A one-variable opening of the constant `v` against `root`, whose single layer of
/// `2·BLOWUP` leaves answers every drawn pair with `(v, v)` and the given digests.
fn opening_with_siblings(v: F, root: Hash, siblings: Vec<Hash>) -> bool {
    let mut t = Transcript::new(b"security-review-pcs");
    t.absorb(b"pcs-eval/interleaved-v2");
    t.absorb(&root);
    t.absorb(&(1u64).to_le_bytes());
    t.absorb_field(F::zero());
    t.absorb_field(v);
    for x in [v, F::zero(), -v] {
        t.absorb_field(x);
    }
    let _: F = t.squeeze_field();
    t.absorb_field(v);
    let mut pairs = t.squeeze_indices(BLOWUP, num_queries());
    pairs.sort();
    pairs.dedup();
    let proof = EvaluationProof {
        sc: SumcheckProof {
            round_polys: vec![vec![v, F::zero(), -v]],
        },
        folds: Proof {
            intermediate_roots: vec![],
            final_values: vec![v],
            layers: vec![LayerOpening {
                columns: vec![v; 2 * pairs.len()],
                salts: vec![],
                siblings,
            }],
        },
    };
    verify_eval(
        &Commitment { root },
        &[F::zero()],
        v,
        &proof,
        &mut Transcript::new(b"security-review-pcs"),
    )
    .unwrap_or(false)
}

#[test]
fn public_pcs_rejects_openings_of_a_shorter_tree() {
    let v = F::from(7u64);
    let domain = MerkleDomain::public(0);
    let leaf = domain.leaf(&field_to_bytes(v));
    let short = MerkleTree::new(2, |_| leaf, &domain);
    // The tree's height comes from the protocol, not from the proof: neither the
    // short tree's (empty) multi-opening nor its digests reach its root.
    assert!(!opening_with_siblings(v, short.root(), vec![]));
    assert!(!opening_with_siblings(v, short.root(), vec![short.root()]));
    // The same constant opens against an honest commitment of the full height.
    let full = MerkleTree::new(2 * BLOWUP, |_| leaf, &domain);
    assert!(opening_with_siblings(v, full.root(), vec![]));
}

#[test]
fn hash_chain_candidate_is_rejected_by_public_pcs() {
    let (v0, h1, v1) = (1u64..)
        .find_map(|x| {
            let a = F::from(x);
            let h = Sha256::digest(field_to_bytes(a));
            F::deserialize_compressed(&h[..])
                .ok()
                .filter(|b| *b != a)
                .map(|b| (a, h.to_vec(), b))
        })
        .unwrap();
    let h2 = Sha256::digest(&h1);
    let root: Hash = Sha256::digest(h2).into();
    // This candidate assumes raw digest concatenation; leaves and inner nodes are
    // hashed under distinct domain prefixes. The opening's shape is also fixed by
    // the protocol, so the rejection is not evidence of PCS binding on its own.
    assert!(!opening_with_siblings(v0, root, vec![]));
    assert!(!opening_with_siblings(v1, root, vec![h2.into()]));
}
