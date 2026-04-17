//! ZK proof of a valid Schnorr signature (EdDSA-style over Jubjub) under a
//! publicly committed key, without revealing the key or the signature itself.
//! Public  : m  — the message digest, SHA-256(msg)[..31 bytes] as an Fq
//!                element. The verifier knows the message and re-derives m.
//!           cm — Poseidon(pk.x, pk.y), a commitment to the signer's key,
//!                published once (e.g. at registration time).
//!
//! Private : pk = sk·G, R = r·G, s = r + c·sk  (all hidden; sk and r
//!           themselves never enter the circuit).
//!
//! The circuit enforces
//!               cm  =  Poseidon(pk)              (key binding)
//!               c   =  Poseidon(R, pk, m)        (Fiat–Shamir challenge)
//!               s·G =  R + c·pk                  (Schnorr equation)

use ark_bls12_381::Fr as Fq;
use ark_crypto_primitives::{
    crh::{
        poseidon::{
            constraints::{CRHGadget, CRHParametersVar},
            CRH,
        },
        CRHScheme, CRHSchemeGadget,
    },
    sponge::poseidon::{find_poseidon_ark_and_mds, PoseidonConfig},
};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ed_on_bls12_381::{constraints::EdwardsVar, EdwardsAffine, Fr};
use ark_ff::{BigInteger, One, PrimeField, UniformRand};
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::Boolean,
    convert::ToBitsGadget,
    eq::EqGadget,
    fields::fp::FpVar,
    groups::CurveVar,
};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use rand::{rngs::StdRng, SeedableRng};
use sha2::{Digest, Sha256};
use zk_snark::snark::{prove, setup, verify};

/// Digest a message into a field element: m = SHA256(msg)[..31] as Fq.
/// Taking only 31 bytes (248 bits < 255-bit modulus) keeps the map injective.
fn message_digest(msg: &[u8]) -> Fq {
    let hash = Sha256::digest(msg);
    Fq::from_le_bytes_mod_order(&hash[..31])
}

/// Fiat–Shamir challenge c = Poseidon(R.x, R.y, pk.x, pk.y, m), matching the
/// in-circuit derivation.
fn schnorr_challenge(
    params: &PoseidonConfig<Fq>,
    r_point: &EdwardsAffine,
    pk: &EdwardsAffine,
    m: Fq,
) -> Fq {
    CRH::<Fq>::evaluate(params, [r_point.x, r_point.y, pk.x, pk.y, m].as_slice())
        .expect("Poseidon evaluation failed")
}

/// Reduce an Fq element to Fr via its integer representative. The circuit
/// multiplies pk by the full 255-bit integer c, and pk lies in the
/// prime-order-r subgroup, so [c]·pk = [c mod r]·pk — consistent with this.
fn fq_to_fr(c: Fq) -> Fr {
    Fr::from_le_bytes_mod_order(&c.into_bigint().to_bytes_le())
}

struct SchnorrCircuit {
    /// Poseidon parameters — same for both prover and verifier.
    params: PoseidonConfig<Fq>,
    /// Message digest m (public) — verifier derives this from the known
    /// message. `None` at setup (only structure is read); `Some(m)` when
    /// proving.
    m: Option<Fq>,
    /// Key commitment cm = Poseidon(pk.x, pk.y) (public).
    cm: Option<Fq>,
    /// Response scalar s = r + c·sk (private).
    s: Option<Fr>,
    /// Commitment point R = r·G (private).
    r_point: Option<EdwardsAffine>,
    /// Public key pk = sk·G (private — hidden from the verifier).
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
        let m_var = FpVar::<Fq>::new_input(cs.clone(), || Ok(self.m.unwrap_or_default()))?;
        let cm_var = FpVar::<Fq>::new_input(cs.clone(), || Ok(self.cm.unwrap_or_default()))?;

        let s_bits = alloc_fr_bits(cs.clone(), self.s)?;

        let r_point_val = self.r_point.unwrap_or_default();
        let r_var = EdwardsVar::new_witness(cs.clone(), || Ok(r_point_val))?;

        let pk_val = self.pk.unwrap_or_default();
        let pk_var = EdwardsVar::new_witness(cs.clone(), || Ok(pk_val))?;

        let params_var = CRHParametersVar::new_constant(cs.clone(), &self.params)?;

        let pk_coords = [pk_var.x.clone(), pk_var.y.clone()];
        let cm_check = CRHGadget::<Fq>::evaluate(&params_var, &pk_coords)?;
        cm_check.enforce_equal(&cm_var)?;

        let challenge_input = [
            r_var.x.clone(),
            r_var.y.clone(),
            pk_var.x.clone(),
            pk_var.y.clone(),
            m_var,
        ];
        let c_var = CRHGadget::<Fq>::evaluate(&params_var, &challenge_input)?;

        let generator = EdwardsVar::new_constant(cs.clone(), EdwardsAffine::generator())?;

        let lhs = generator.scalar_mul_le(s_bits.iter())?;

        let c_bits = c_var.to_bits_le()?;

        let c_pk = pk_var.scalar_mul_le(c_bits.iter())?;
        let rhs = r_var + c_pk;

        lhs.enforce_equal(&rhs)?;

        Ok(())
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let msg = b"Hello from the ZK world!";
    let g = EdwardsAffine::generator();

    let (ark, mds) = find_poseidon_ark_and_mds::<Fq>(
        Fq::MODULUS_BIT_SIZE as u64,
        2,
        8,
        57,
        0,
    );
    let params = PoseidonConfig { full_rounds: 8, partial_rounds: 57, alpha: 5, ark, mds, rate: 2, capacity: 1 };

    let sk = Fr::rand(&mut rng);
    let pk: EdwardsAffine = (g * sk).into_affine();

    let cm = CRH::<Fq>::evaluate(&params, [pk.x, pk.y].as_slice())
        .expect("Poseidon evaluation failed");

    let m = message_digest(msg);
    let r = Fr::rand(&mut rng);
    let r_point: EdwardsAffine = (g * r).into_affine();

    let c_fq = schnorr_challenge(&params, &r_point, &pk, m);
    let c_fr = fq_to_fr(c_fq);
    let s = r + c_fr * sk;

    let lhs_native = (g * s).into_affine();
    let rhs_native = (r_point + pk * c_fr).into_affine();
    assert_eq!(lhs_native, rhs_native, "native Schnorr equation failed");

    let (pp, vp) = setup::<Fq, _>(SchnorrCircuit {
        params: params.clone(),
        m: None,
        cm: None,
        s: None,
        r_point: None,
        pk: None,
    }, true);

    println!("Proving Schnorr signature knowledge (ZK mode)…");
    let proof = prove(
        &pp,
        SchnorrCircuit {
            params: params.clone(),
            m: Some(m),
            cm: Some(cm),
            s: Some(s),
            r_point: Some(r_point),
            pk: Some(pk),
        },
        true,
        &mut rng,
    ).expect("failed to prove");

    let public = vec![Fq::one(), m, cm];
    let ok = verify(&vp, &public, &proof).expect("failed to verify");
    println!("Correct statement verified : {ok}");
    assert!(ok, "valid Schnorr proof rejected");

    let wrong_m = message_digest(b"Wrong message");
    let public_wrong_msg = vec![Fq::one(), wrong_m, cm];
    let rejected = !verify(&vp, &public_wrong_msg, &proof).expect("failed to verify");
    println!("Wrong message rejected     : {rejected}");
    assert!(rejected, "wrong-message proof accepted");

    let public_wrong_key = vec![Fq::one(), m, cm + Fq::one()];
    let rejected = !verify(&vp, &public_wrong_key, &proof).expect("failed to verify");
    println!("Wrong key binding rejected : {rejected}");
    assert!(rejected, "wrong-commitment proof accepted");
}
