//! Binding the additive sumcheck-mask evaluation `Z(x*)`.
//!
//! The additive mask (see `sumcheck.rs`) is `Z(x) = Σ_{i<ell} Σ_{1≤j≤deg} c[i][j-1]·x_i^j`.
//! Its terminal value
//!   `Z(x*) = Σ_{i,j} c[i][j-1]·(x*_i)^j = Σ_idx packed[idx]·W[idx]`
//! is an *inner product* of the committed coefficient vector with the public weight
//! vector `W[i·deg + (j-1)] = (x*_i)^j`. It is therefore not a native multilinear
//! PCS evaluation; we bind it with a small degree-2 sumcheck over `μ = ell_for(deg·ell)`
//! variables that reduces the inner product to a single point `k*`, where the
//! committed coefficient MLE `ĉ` is opened by the PCS. The verifier recomputes `Ŵ(k*)`
//! itself, so only `ĉ(k*)` is opened.

use ark_ff::{FftField, PrimeField};
use ark_serialize::CanonicalSerialize;
use ark_std::{rand::Rng, UniformRand};

use crate::{
    pcs::{opened_symbols_bound, prove_eval, verify_eval, PcsError, Commitment, Proof, Witness},
    piop::ell_for,
    r1cs::mle_of_vector,
    sumcheck::{sumcheck_verify, SumcheckProof},
    transcript::Transcript,
};

/// Number of boolean variables of the committed coefficient table.
///
/// The `deg·ell` real coefficients need only `⌈log2(deg·ell)⌉` variables, but a table
/// that small is *fully recoverable* from one evaluation proof: every symbol the proof
/// opens — a ± pair per query per fold layer — is a known linear functional of the
/// original table (encoding and folding are linear with public challenges), and the
/// proof opens far more such functionals than an unpadded table has entries, so their
/// rank generically reaches the table size and the whole table is a solvable linear
/// system. Recovering the coefficients undoes the round-polynomial masking entirely.
///
/// `μ` is enlarged so that random zero-weight filler outnumbers the symbols opened
/// by the PCS. This is the implementation's randomized-encoding countermeasure.
fn mask_mu(ell: usize, deg: usize) -> usize {
    let data = deg * ell;
    let mut mu = ell_for(data);
    while (1usize << mu) < data + 2 * opened_symbols_bound(mu) {
        mu += 1;
    }
    mu
}

/// Pack the mask coefficients and fill zero-weight slots with fresh randomness.
pub fn pack_mask_coeffs<F: PrimeField + UniformRand, R: Rng>(
    coeffs: &[Vec<F>],
    deg: usize,
    rng: &mut R,
) -> Vec<F> {
    let ell = coeffs.len();
    let mu = mask_mu(ell, deg);
    let mut packed: Vec<F> = (0..1usize << mu).map(|_| F::rand(rng)).collect();
    for (i, ci) in coeffs.iter().enumerate() {
        for (jm1, &c) in ci.iter().enumerate() {
            packed[i * deg + jm1] = c;
        }
    }
    packed
}

/// Public weight table `W[i·deg + (j-1)] = (point_i)^j` (j = 1..deg), zero-padded to `2^μ`.
fn weight_vector<F: PrimeField>(point: &[F], deg: usize) -> Vec<F> {
    let ell = point.len();
    let mu = mask_mu(ell, deg);
    let mut w = vec![F::zero(); 1usize << mu];
    for (i, &p) in point.iter().enumerate() {
        let mut pj = p;
        for jm1 in 0..deg {
            w[i * deg + jm1] = pj;
            pj *= p;
        }
    }
    w
}

#[derive(CanonicalSerialize)]
pub struct MaskEvalProof<F: PrimeField> {
    /// Degree-2 inner-product sumcheck over `{0,1}^μ` reducing `Z(x*)` to a point `k*`.
    pub sc: SumcheckProof<F>,
    /// PCS opening of the committed coefficient MLE `ĉ` at `k*`.
    pub coeff_proof: Proof<F>,
}

/// Prove `Z(point) = ⟨packed, W⟩` for the committed coefficient MLE, returning the
/// value and the proof. `coeff_wit` commits `packed` (the output of `pack_mask_coeffs`).
pub fn prove_mask_eval<F: PrimeField + FftField, R: Rng>(
    coeff_wit: &Witness<F>,
    packed: &[F],
    point: &[F],
    deg: usize,
    transcript: &mut Transcript,
    zk: bool,
    rng: &mut R,
) -> Result<(F, MaskEvalProof<F>), PcsError> {
    let mu = mask_mu(point.len(), deg);
    debug_assert_eq!(packed.len(), 1usize << mu);
    let w = weight_vector(point, deg);

    let v: F = packed.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
    transcript.absorb_field(v);

    let mut a = packed.to_vec();
    let mut b = w;
    let mut challenges = Vec::with_capacity(mu);
    let mut round_polys = Vec::with_capacity(mu);
    let two = F::from(2u64);
    let mut current = 1usize << mu;

    for _ in 0..mu {
        let half = current / 2;
        let mut s_j = vec![F::zero(); 3];
        for k in 0..half {
            let a0 = a[2 * k];
            let a1 = a[2 * k + 1];
            let b0 = b[2 * k];
            let b1 = b[2 * k + 1];
            s_j[0] += a0 * b0;
            s_j[1] += a1 * b1;
            s_j[2] += (two * a1 - a0) * (two * b1 - b0);
        }
        for &e in &s_j {
            transcript.absorb_field(e);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);
        round_polys.push(s_j);

        let omr = F::one() - r;
        for k in 0..half {
            a[k] = omr * a[2 * k] + r * a[2 * k + 1];
            b[k] = omr * b[2 * k] + r * b[2 * k + 1];
        }
        current = half;
    }

    let (_cv, coeff_proof) = prove_eval(coeff_wit, &challenges, transcript, zk, rng)?;

    Ok((v, MaskEvalProof { sc: SumcheckProof { round_polys }, coeff_proof }))
}

/// Verify that the committed coefficients evaluate to `claimed_v = Z(point)`.
pub fn verify_mask_eval<F: PrimeField + FftField>(
    coeff_commit: &Commitment,
    point: &[F],
    deg: usize,
    claimed_v: F,
    proof: &MaskEvalProof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    let mu = mask_mu(point.len(), deg);

    let Some((k_star, final_eval)) = sumcheck_verify(&proof.sc, mu, 2, claimed_v, transcript)
    else {
        return Ok(false);
    };

    let coeff_v = proof.coeff_proof.final_value;
    if !verify_eval(coeff_commit, &k_star, coeff_v, &proof.coeff_proof, transcript)? {
        return Ok(false);
    }
    let w = weight_vector(point, deg);
    let w_at_k = mle_of_vector(&w, mu, &k_star);

    Ok(coeff_v * w_at_k == final_eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pcs::commit, sumcheck::additive_mask_eval};
    use ark_bls12_381::Fr as F;
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    fn roundtrip(ell: usize, deg: usize) -> bool {
        let mut rng = rng();
        let coeffs: Vec<Vec<F>> = (0..ell)
            .map(|_| (0..deg).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let packed = pack_mask_coeffs(&coeffs, deg, &mut rng);
        let (comm, wit) = commit(packed.clone(), true, &mut rng);

        let point: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();

        let mut pt = Transcript::new(b"mask-test");
        let (v, proof) = prove_mask_eval(&wit, &packed, &point, deg, &mut pt, true, &mut rng).unwrap();

        assert_eq!(v, additive_mask_eval(&coeffs, &point), "value mismatch");

        let mut vt = Transcript::new(b"mask-test");
        verify_mask_eval(&comm, &point, deg, v, &proof, &mut vt).unwrap()
    }

    #[test]
    fn mask_eval_roundtrip() {
        assert!(roundtrip(1, 3)); // ell_row of the small test circuits
        assert!(roundtrip(2, 2)); // ell_col
        assert!(roundtrip(3, 3));
        assert!(roundtrip(4, 2));
    }

    #[test]
    fn wrong_value_rejected() {
        let mut rng = rng();
        let ell = 3;
        let deg = 3;
        let coeffs: Vec<Vec<F>> = (0..ell)
            .map(|_| (0..deg).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let packed = pack_mask_coeffs(&coeffs, deg, &mut rng);
        let (comm, wit) = commit(packed.clone(), true, &mut rng);
        let point: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();

        let mut pt = Transcript::new(b"mask-test");
        let (v, proof) = prove_mask_eval(&wit, &packed, &point, deg, &mut pt, true, &mut rng).unwrap();

        let mut vt = Transcript::new(b"mask-test");
        assert!(!verify_mask_eval(&comm, &point, deg, v + F::from(1u64), &proof, &mut vt).unwrap());
    }
}
