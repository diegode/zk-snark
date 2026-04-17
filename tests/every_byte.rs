//! Damaged proofs are refused: flipping a bit of an honest proof's bytes, cutting
//! them short or appending to them must leave a proof that does not parse or that
//! the verifier rejects, without a panic. A byte that could change unnoticed would
//! belong to a part of the proof that the verifier never checks.
use ark_bls12_381::Fr as F;
use ark_ff::{BigInteger, One, PrimeField};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable};
use ark_serialize::CanonicalSerialize;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use zk_snark::snark::{SNARKProof, VerifierParams, prove, setup, verify};

/// `a·w = b` with `a, b` public: the smallest circuit with public inputs and a witness.
struct PubMul {
    a: F,
    b: F,
    w: F,
}

impl ConstraintSynthesizer<F> for PubMul {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let a = cs.new_input_variable(|| Ok(self.a))?;
        let b = cs.new_input_variable(|| Ok(self.b))?;
        let w = cs.new_witness_variable(|| Ok(self.w))?;
        cs.enforce_r1cs_constraint(
            || ark_relations::lc!() + a,
            || ark_relations::lc!() + w,
            || ark_relations::lc!() + b,
        )
    }
}

/// `x_{i+1} = x_i² + i` for `i < n`, with `x_0` and `x_n` public: large enough that
/// the draws leave pairs of every tree unopened, so its openings send digests.
struct Chain {
    n: usize,
}

impl ConstraintSynthesizer<F> for Chain {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let mut x = vec![F::from(3)];
        for i in 0..self.n {
            x.push(x[i] * x[i] + F::from(i as u64));
        }
        let first = cs.new_input_variable(|| Ok(x[0]))?;
        let last = cs.new_input_variable(|| Ok(x[self.n]))?;
        let mut previous = first;
        for i in 0..self.n {
            let next = match i + 1 == self.n {
                true => last,
                false => cs.new_witness_variable(|| Ok(x[i + 1]))?,
            };
            cs.enforce_r1cs_constraint(
                || ark_relations::lc!() + previous,
                || ark_relations::lc!() + previous,
                || ark_relations::lc!() + next + (-F::from(i as u64), Variable::One),
            )?;
            previous = next;
        }
        Ok(())
    }
}

fn honest(zk: bool) -> (VerifierParams, Vec<F>, SNARKProof<F>) {
    let circuit = || PubMul {
        a: F::from(3),
        b: F::from(15),
        w: F::from(5),
    };
    let (pp, vp) = setup::<F, _>(circuit(), zk).unwrap();
    let proof = prove(&pp, circuit(), &mut StdRng::seed_from_u64(5)).unwrap();
    (vp, vec![F::one(), F::from(3), F::from(15)], proof)
}

/// A clean refusal: the bytes do not parse, or the verifier returns `Ok(false)`.
fn refused(vp: &VerifierParams, public: &[F], bytes: &[u8]) -> bool {
    catch_unwind(AssertUnwindSafe(|| match SNARKProof::from_bytes(bytes) {
        None => true,
        Some(proof) => verify(vp, public, &proof) == Ok(false),
    }))
    .unwrap_or(false)
}

/// Flip bit `offset % 8` of the byte at each offset, one at a time, and of every
/// damaged copy require a clean refusal.
fn assert_damage_refused(vp: &VerifierParams, public: &[F], bytes: &[u8], offsets: Vec<usize>) {
    assert!(!refused(vp, public, bytes), "the honest proof is refused");
    let missed: Vec<usize> = offsets
        .par_iter()
        .copied()
        .filter(|&offset| {
            let mut damaged = bytes.to_vec();
            damaged[offset] ^= 1 << (offset % 8);
            !refused(vp, public, &damaged)
        })
        .collect();
    assert!(
        missed.is_empty(),
        "{} of {} damaged copies were not refused, first at bytes {:?}",
        missed.len(),
        offsets.len(),
        &missed[..missed.len().min(8)]
    );
}

#[test]
fn every_byte_of_a_plain_proof_is_checked() {
    let (vp, public, proof) = honest(false);
    let bytes = proof.to_bytes();
    assert_damage_refused(&vp, &public, &bytes, (0..bytes.len()).collect());
}

/// The one-constraint proof above opens every pair of its small trees, so no digest
/// is sent; here the multi-openings of the public trees carry digests too.
#[test]
fn every_part_of_a_larger_plain_proof_is_checked() {
    let n = 64;
    let (pp, vp) = setup::<F, _>(Chain { n }, false).unwrap();
    let proof = prove(&pp, Chain { n }, &mut StdRng::seed_from_u64(6)).unwrap();
    let plain = proof.plain.as_ref().unwrap();
    assert!(!plain.w_proof.folds.layers[0].siblings.is_empty());
    let mut x = F::from(3);
    for i in 0..n {
        x = x * x + F::from(i as u64);
    }
    let bytes = proof.to_bytes();
    let offsets = (0..bytes.len()).step_by(997).collect();
    assert_damage_refused(&vp, &[F::one(), F::from(3), x], &bytes, offsets);
}

/// The private opening's layers end the encoding. Every byte before them is
/// damaged; the long runs of columns, salts and digests in them, every 97th byte.
#[test]
fn every_part_of_a_zk_proof_is_checked() {
    let (vp, public, proof) = honest(true);
    let bytes = proof.to_bytes();
    let layers = proof
        .zk
        .as_ref()
        .unwrap()
        .opening
        .folds
        .layers
        .compressed_size();
    let start = bytes.len() - layers;
    let offsets = (0..start).chain((start..bytes.len()).step_by(97)).collect();
    assert_damage_refused(&vp, &public, &bytes, offsets);
}

#[test]
#[ignore = "damages all ~700 KB of a ZK proof: about twelve minutes on eight cores"]
fn every_byte_of_a_zk_proof_is_checked() {
    let (vp, public, proof) = honest(true);
    let bytes = proof.to_bytes();
    assert_damage_refused(&vp, &public, &bytes, (0..bytes.len()).collect());
}

#[test]
fn each_proof_has_one_encoding() {
    for zk in [false, true] {
        let (vp, public, proof) = honest(zk);
        let bytes = proof.to_bytes();
        assert!(!refused(&vp, &public, &bytes));

        let mut longer = bytes.clone();
        longer.push(0);
        assert!(SNARKProof::<F>::from_bytes(&longer).is_none());

        let step = if zk { 997 } else { 1 };
        let cuts: Vec<usize> = (0..bytes.len()).step_by(step).collect();
        assert!(
            cuts.par_iter()
                .all(|&len| SNARKProof::<F>::from_bytes(&bytes[..len]).is_none())
        );

        // A second spelling of a field element: x + p in place of x.
        let sumcheck = match (&proof.plain, &proof.zk) {
            (Some(plain), _) => &plain.piop_proof.outer_sc,
            (_, Some(zk)) => &zk.piop.outer_sc,
            _ => unreachable!(),
        };
        let x = sumcheck.round_polys[0][0];
        let mut spelling = x.into_bigint();
        assert!(!spelling.add_with_carry(&F::MODULUS));
        let mut encoded = Vec::new();
        x.serialize_compressed(&mut encoded).unwrap();
        let at = bytes
            .windows(encoded.len())
            .position(|w| w == encoded)
            .unwrap();
        let mut respelled = bytes.clone();
        respelled[at..at + encoded.len()].copy_from_slice(&spelling.to_bytes_le());
        assert!(SNARKProof::<F>::from_bytes(&respelled).is_none());
    }
}
