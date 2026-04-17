//! Sumcheck verification and the additive masks used by the Spartan PIOP.
//!
//! Each round sends the round polynomial at `0, …, degree`; the verifier uses
//! Lagrange interpolation to evaluate it at the Fiat--Shamir challenge.
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use ark_std::{rand::Rng, UniformRand};

use crate::transcript::Transcript;

#[derive(Clone, Debug, CanonicalSerialize)]
pub struct SumcheckProof<F: PrimeField> {
    /// `round_polys[j]` = `[s_j(0), s_j(1), …, s_j(degree)]`.
    pub round_polys: Vec<Vec<F>>,
}

/// Run the sumcheck verifier.
///
/// Returns `Some((challenges, final_eval_claim))` on success.
/// The caller must still perform the final oracle check.
pub fn sumcheck_verify<F: PrimeField>(
    proof: &SumcheckProof<F>,
    num_vars: usize,
    degree: usize,
    claimed_sum: F,
    transcript: &mut Transcript,
) -> Option<(Vec<F>, F)> {
    transcript.absorb_field(claimed_sum);

    let mut expected = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for s_j in &proof.round_polys {
        if s_j.len() != degree + 1 {
            return None;
        }

        if s_j[0] + s_j[1] != expected {
            return None;
        }

        for &v in s_j {
            transcript.absorb_field(v);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);

        expected = lagrange_eval(s_j, r);
    }

    if challenges.len() != num_vars {
        return None;
    }
    Some((challenges, expected))
}

// Additive (Libra/CFS-style) sumcheck mask.
//
// To mask a sumcheck whose summand has individual degree `deg` per variable, the
// prover draws a polynomial
//   Z(x) = Σ_{i<ell} R_i(x_i),   R_i(t) = Σ_{j=1}^{deg} c[i][j-1]·t^j     (no constant)
// with all c[i][j] independent and uniform. The contribution of `Z` to the round-i
// sumcheck polynomial is `2^{ell-1-i}·R_i(t)` plus a constant (Σ over the other
// variables) — a *linear* image of the fresh coefficients c[i][1..deg]. Hence every
// exposed round-polynomial coefficient of `g + τ·Z` is (its `g`-part) + (a fresh
// uniform value), i.e. uniform, with the degree-0 term pinned by the consistency
// constraint. Thus the witness-dependent higher coefficients are shifted by
// independent uniform values.
//
// The mask is non-multilinear (degree `deg` per variable), so its terminal value
// Z(x*) is not a native PCS evaluation; it is bound by a small inner-product
// sub-proof against a hiding commitment to the coefficients (see `mask.rs`).

/// Sample the coefficients `c[i][j-1]` (i < ell, 1 ≤ j ≤ deg) of an additive
/// degree-`deg`-per-variable mask. All coefficients are independent and uniform.
pub fn sample_additive_mask<F: PrimeField + UniformRand, R: Rng>(
    ell: usize,
    deg: usize,
    rng: &mut R,
) -> Vec<Vec<F>> {
    (0..ell)
        .map(|_| (0..deg).map(|_| F::rand(rng)).collect())
        .collect()
}

/// `R_i(z) = Σ_{j=1}^{deg} coeffs_i[j-1]·z^j` (no constant term).
fn univ_eval<F: PrimeField>(coeffs_i: &[F], z: F) -> F {
    let mut acc = F::zero();
    let mut zp = z;
    for &c in coeffs_i {
        acc += c * zp;
        zp *= z;
    }
    acc
}

/// `Z(point) = Σ_i R_i(point_i)` — the terminal mask evaluation.
pub fn additive_mask_eval<F: PrimeField>(coeffs: &[Vec<F>], point: &[F]) -> F {
    coeffs.iter().zip(point).map(|(ci, &p)| univ_eval(ci, p)).sum()
}

/// `Σ_{x∈{0,1}^ell} Z(x) = 2^{ell-1}·Σ_{i,j} c[i][j]`, since `x_i^j = x_i` on the cube.
/// This lets the masked sumcheck form its initial claim without enumerating the cube.
pub fn additive_mask_sum<F: PrimeField>(coeffs: &[Vec<F>]) -> F {
    let ell = coeffs.len();
    let total: F = coeffs.iter().flatten().copied().sum();
    F::from(2u64).pow([ell.saturating_sub(1) as u64]) * total
}

/// Evaluations `[contrib(0), …, contrib(deg)]` of the additive mask's contribution to
/// the round-`i` sumcheck polynomial, given the `i` challenges already bound:
///   contrib_i(t) = 2^{ell-1-i}·(Σ_{k<i} R_k(r_k) + R_i(t)) + 2^{ell-2-i}·Σ_{k>i} σ_k,
/// where r_k = challenges[k] and σ_k = R_k(1) = Σ_j c[k][j].
pub fn additive_mask_round<F: PrimeField>(
    coeffs: &[Vec<F>],
    i: usize,
    challenges: &[F],
) -> Vec<F> {
    let ell = coeffs.len();
    let deg = coeffs[i].len();
    let prefix: F = (0..i).map(|k| univ_eval(&coeffs[k], challenges[k])).sum();
    let suffix_sigma: F = ((i + 1)..ell)
        .map(|k| coeffs[k].iter().copied().sum::<F>())
        .sum();
    let pow_main = F::from(2u64).pow([(ell - 1 - i) as u64]);
    let suffix_term = if i + 1 < ell {
        F::from(2u64).pow([(ell - 2 - i) as u64]) * suffix_sigma
    } else {
        F::zero()
    };
    (0..=deg)
        .map(|t| pow_main * (prefix + univ_eval(&coeffs[i], F::from(t as u64))) + suffix_term)
        .collect()
}

/// Evaluate the unique polynomial of degree ≤ d passing through
/// (0, ys[0]), (1, ys[1]), …, (d, ys[d]) at the point r.
pub fn lagrange_eval<F: PrimeField>(ys: &[F], r: F) -> F {
    let d = ys.len() - 1;
    let mut result = F::zero();
    for i in 0..=d {
        let xi = F::from(i as u64);
        let mut basis = F::one();
        for j in 0..=d {
            if j != i {
                let xj = F::from(j as u64);
                basis *= (r - xj) * (xi - xj).inverse().unwrap();
            }
        }
        result += ys[i] * basis;
    }
    result
}

/// eq(x, z) = ∏_j (x_j·z_j + (1−x_j)·(1−z_j))
/// where x is given as a little-endian integer.
pub fn eq_eval<F: PrimeField>(x_int: usize, k: usize, z: &[F]) -> F {
    assert_eq!(z.len(), k);
    (0..k)
        .map(|j| {
            let xj = F::from(((x_int >> j) & 1) as u64);
            xj * z[j] + (F::one() - xj) * (F::one() - z[j])
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{One, Zero};
    use rand::{rngs::StdRng, SeedableRng};

    fn rng() -> StdRng { StdRng::seed_from_u64(42) }

    #[test]
    fn test_eq_eval() {
        let k = 3;
        let x_int = 3;

        let z_match = vec![F::from(1u64), F::from(1u64), F::from(0u64)];
        assert_eq!(eq_eval(x_int, k, &z_match), F::one());

        let z_no_match = vec![F::from(1u64), F::from(0u64), F::from(0u64)];
        assert_eq!(eq_eval(x_int, k, &z_no_match), F::zero());

        let z_mixed = vec![F::from(2u64), F::from(3u64), F::from(4u64)];
        let expected = F::from(2u64) * F::from(3u64) * (F::one() - F::from(4u64));
        assert_eq!(eq_eval(x_int, k, &z_mixed), expected);
    }

    #[test]
    fn test_lagrange_eval() {
        let ys1 = vec![F::from(3u64), F::from(7u64)];
        assert_eq!(lagrange_eval(&ys1, F::from(0u64)), F::from(3u64));
        assert_eq!(lagrange_eval(&ys1, F::from(1u64)), F::from(7u64));
        assert_eq!(lagrange_eval(&ys1, F::from(2u64)), F::from(11u64));

        let ys2 = vec![F::from(1u64), F::from(4u64), F::from(9u64)];
        assert_eq!(lagrange_eval(&ys2, F::from(0u64)), F::from(1u64));
        assert_eq!(lagrange_eval(&ys2, F::from(1u64)), F::from(4u64));
        assert_eq!(lagrange_eval(&ys2, F::from(2u64)), F::from(9u64));
        assert_eq!(lagrange_eval(&ys2, F::from(3u64)), F::from(16u64));
    }

    /// Brute-force Z(x) = Σ_i R_i(x_i) at a boolean point given by bits of `x_int`.
    fn additive_mask_brute<F: PrimeField>(coeffs: &[Vec<F>], x_int: usize) -> F {
        let mut acc = F::zero();
        for (i, ci) in coeffs.iter().enumerate() {
            let xi = F::from(((x_int >> i) & 1) as u64);
            let mut zp = xi;
            for &c in ci {
                acc += c * zp;
                zp *= xi;
            }
        }
        acc
    }

    #[test]
    fn test_additive_mask_sum_matches_hypercube() {
        let mut rng = rng();
        for ell in 1..=5 {
            for deg in 2..=3 {
                let coeffs = sample_additive_mask::<F, _>(ell, deg, &mut rng);
                let brute: F = (0..(1usize << ell))
                    .map(|x| additive_mask_brute(&coeffs, x))
                    .sum();
                assert_eq!(additive_mask_sum(&coeffs), brute, "sum mismatch ell={ell} deg={deg}");
            }
        }
    }

    #[test]
    fn test_additive_mask_eval_on_cube() {
        let mut rng = rng();
        let ell = 4;
        let coeffs = sample_additive_mask::<F, _>(ell, 3, &mut rng);
        for x in 0..(1usize << ell) {
            let point: Vec<F> = (0..ell).map(|i| F::from(((x >> i) & 1) as u64)).collect();
            assert_eq!(additive_mask_eval(&coeffs, &point), additive_mask_brute(&coeffs, x));
        }
    }

    #[test]
    fn test_additive_mask_round_reconstructs_partial_sum() {
        let mut rng = rng();
        let ell = 4;
        let deg = 3;
        let coeffs = sample_additive_mask::<F, _>(ell, deg, &mut rng);
        let challenges: Vec<F> = (0..ell).map(|_| F::rand(&mut rng)).collect();

        for i in 0..ell {
            let round = additive_mask_round(&coeffs, i, &challenges[..i]);
            for t in 0..=deg {
                let tf = F::from(t as u64);
                let mut expected = F::zero();
                for tail in 0..(1usize << (ell - 1 - i)) {
                    let mut point = challenges[..i].to_vec();
                    point.push(tf);
                    for b in 0..(ell - 1 - i) {
                        point.push(F::from(((tail >> b) & 1) as u64));
                    }
                    expected += additive_mask_eval(&coeffs, &point);
                }
                assert_eq!(round[t], expected, "round {i} contrib at t={t} mismatch");
            }
        }
    }

    #[test]
    fn test_sumcheck_verify_rejects_wrong_poly_length() {
        let proof = SumcheckProof { round_polys: vec![vec![F::from(2u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_consistency_failure() {
        let proof = SumcheckProof { round_polys: vec![vec![F::from(1u64), F::from(1u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(5u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_too_few_rounds() {
        let proof = SumcheckProof { round_polys: vec![vec![F::from(1u64), F::from(1u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 2, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_success() {
        let proof = SumcheckProof::<F> { round_polys: vec![] };
        let mut t = crate::transcript::Transcript::new(b"test");
        let (challenges, final_eval) =
            sumcheck_verify(&proof, 0, 1, F::from(42u64), &mut t).expect("empty sumcheck should succeed");
        assert!(challenges.is_empty());
        assert_eq!(final_eval, F::from(42u64));
    }
}
