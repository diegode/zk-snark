//! ZK proof of a valid Schnorr signature (EdDSA-style over Jubjub), without
//! revealing the public key or the signature itself.
//!
//! Setting
//! -------
//!   Curve  : Jubjub (twisted Edwards, defined over BLS12-381's scalar field).
//!   Hash   : SHA-256 (native, off-circuit) — first 31 bytes → ~248-bit integer
//!            interpreted as a scalar, safely below both field moduli.
//!
//! Public  : the challenge  c  (one Fq element) derived as
//!               c = SHA-256(msg)[..31 bytes]
//!           The verifier knows the message and can re-derive c.
//!
//! Private : sk (secret key ∈ Fr), r (nonce ∈ Fr),
//!           pk = sk·G, R = r·G, s = r + c·sk  (all hidden).
//!
//! The circuit enforces the Schnorr equation
//!               s·G  =  R  +  c·pk
//! over the Jubjub base field Fq.

use ark_ec::{AffineRepr, CurveGroup};
use ark_bls12_381::Fr as Fq;
use ark_ed_on_bls12_381::{constraints::EdwardsVar, EdwardsAffine, Fr};
use ark_ff::{PrimeField, UniformRand};
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::Boolean,
    convert::ToBitsGadget,
    eq::EqGadget,
    fields::fp::FpVar,
    groups::CurveVar,
};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use rand::{rngs::StdRng, SeedableRng};
use sha2::{Digest, Sha256};
use zk_snark::snark::{prove, verify};

// ---------------------------------------------------------------------------
// Native helpers
// ---------------------------------------------------------------------------

/// Compute the Schnorr challenge from a message: c = SHA256(msg)[..31] as Fr.
/// Taking only 31 bytes (248 bits) ensures c < min(p_Fr, p_Fq).
fn schnorr_challenge_fr(msg: &[u8]) -> Fr {
    let hash = Sha256::digest(msg);
    Fr::from_le_bytes_mod_order(&hash[..31])
}

fn schnorr_challenge_fq(msg: &[u8]) -> Fq {
    let hash = Sha256::digest(msg);
    Fq::from_le_bytes_mod_order(&hash[..31]) // same 248-bit integer, different type
}

// ---------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------

struct SchnorrCircuit {
    /// Challenge c (public) — verifier derives this from the known message.
    challenge: Fq,
    /// Secret key (private).
    _sk: Option<Fr>,
    /// Nonce (private).
    _r: Option<Fr>,
    /// Response scalar s = r + c·sk (private).
    s: Option<Fr>,
    /// Commitment point R = r·G (private).
    r_point: Option<EdwardsAffine>,
    /// Public key pk = sk·G (private — hidden from verifier).
    pk: Option<EdwardsAffine>,
}

/// Decompose an Fr scalar into its `Fr::MODULUS_BIT_SIZE` little-endian bits
/// as Boolean<Fq> witnesses allocated on `cs`.
fn alloc_fr_bits(
    cs: ConstraintSystemRef<Fq>,
    val: Option<Fr>,
) -> Result<Vec<Boolean<Fq>>, SynthesisError> {
    let n_bits = Fr::MODULUS_BIT_SIZE as usize;
    let bits_val: Vec<bool> = match val {
        Some(v) => {
            let bigint = v.into_bigint();
            let limbs = bigint.as_ref();
            (0..n_bits).map(|i| (limbs[i / 64] >> (i % 64)) & 1 == 1).collect()
        }
        None => vec![false; n_bits],
    };
    bits_val
        .into_iter()
        .map(|b| Boolean::new_witness(cs.clone(), || Ok(b)))
        .collect()
}

impl ConstraintSynthesizer<Fq> for SchnorrCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fq>) -> Result<(), SynthesisError> {
        // --- Public input: challenge c -----------------------------------
        let c_var = FpVar::<Fq>::new_input(cs.clone(), || Ok(self.challenge))?;

        // --- Private witnesses --------------------------------------------

        // Response scalar s: allocate as Fr bits.
        let s_bits = alloc_fr_bits(cs.clone(), self.s)?;

        // Commitment point R = r·G.
        let r_point_val = self.r_point.unwrap_or_default();
        let r_var = EdwardsVar::new_witness(cs.clone(), || Ok(r_point_val))?;

        // Public key pk = sk·G.
        let pk_val = self.pk.unwrap_or_default();
        let pk_var = EdwardsVar::new_witness(cs.clone(), || Ok(pk_val))?;

        // --- Schnorr check: s·G == R + c·pk ------------------------------

        // Generator G as a circuit constant (no constraints).
        let generator = EdwardsVar::new_constant(cs.clone(), EdwardsAffine::generator())?;

        // lhs = s·G
        let lhs = generator.scalar_mul_le(s_bits.iter())?;

        // c_bits = bits of c treated as a scalar for Edwards multiplication.
        // c < 2^248 < p_Fr, so it represents the same integer in Fr and Fq.
        let c_bits = c_var.to_bits_le()?;

        // rhs = R + c·pk
        let c_pk = pk_var.scalar_mul_le(c_bits.iter())?;
        let rhs = r_var + c_pk;

        // Enforce lhs == rhs
        lhs.enforce_equal(&rhs)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let msg = b"Hello from the ZK world!";
    let g = EdwardsAffine::generator();

    // --- Key generation --------------------------------------------------
    let sk = Fr::rand(&mut rng);
    let pk: EdwardsAffine = (g * sk).into_affine();

    // --- Signing (Schnorr) -----------------------------------------------
    let r = Fr::rand(&mut rng);
    let r_point: EdwardsAffine = (g * r).into_affine();

    let c_fr = schnorr_challenge_fr(msg);
    let c_fq = schnorr_challenge_fq(msg);
    let s = r + c_fr * sk; // response scalar (in Fr)

    // Sanity check (native): s·G == R + c·pk
    let lhs_native = (g * s).into_affine();
    let rhs_native = (r_point + pk * c_fr).into_affine();
    assert_eq!(lhs_native, rhs_native, "native Schnorr equation failed");

    // --- Prove (ZK mode) -------------------------------------------------
    println!("Proving Schnorr signature knowledge (ZK mode)…");
    let proof = prove(
        SchnorrCircuit {
            challenge: c_fq,
            _sk: Some(sk),
            _r: Some(r),
            s: Some(s),
            r_point: Some(r_point),
            pk: Some(pk),
        },
        true,
        &mut rng,
    ).expect("failed to prove");

    // --- Verify (correct challenge) ---------------------------------------
    let ok = verify(
        SchnorrCircuit {
            challenge: c_fq,
            _sk: None,
            _r: None,
            s: None,
            r_point: None,
            pk: None,
        },
        &proof,
    ).expect("failed to verify");
    println!("Correct challenge verified : {ok}");
    assert!(ok, "valid Schnorr proof rejected");

    // --- Verify with wrong message (different challenge) ------------------
    let wrong_c = schnorr_challenge_fq(b"Wrong message");
    let rejected = !verify(
        SchnorrCircuit {
            challenge: wrong_c,
            _sk: None,
            _r: None,
            s: None,
            r_point: None,
            pk: None,
        },
        &proof,
    ).expect("failed to verify");
    println!("Wrong message rejected     : {rejected}");
    assert!(rejected, "wrong-message proof accepted");
}
