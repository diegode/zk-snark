/// Generic sumcheck protocol, supporting polynomials of any individual degree.
///
/// The polynomial is represented as an evaluation closure `eval: Fn(&[F])->F`
/// that can be called at any point in F^n (not just boolean ones).
///
/// Each round j sends `degree + 1` evaluations of the univariate round
/// polynomial at 0, 1, …, degree.  The verifier reconstructs the round
/// polynomial via Lagrange interpolation and evaluates it at the challenge.
///
/// For the PCS polynomial g(x) = f(x)·eq(x, r) use `degree = 1`;
/// for the R1CS inner sumcheck (degree 2 per variable) use `degree = 2`.
use ark_ff::PrimeField;
use ark_std::{rand::Rng, UniformRand};

use crate::transcript::Transcript;

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SumcheckProof<F: PrimeField> {
    /// `round_polys[j]` = `[s_j(0), s_j(1), …, s_j(degree)]`.
    pub round_polys: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

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

        // Consistency: s_j(0) + s_j(1) must equal the previous expected value.
        if s_j[0] + s_j[1] != expected {
            return None;
        }

        for &v in s_j {
            transcript.absorb_field(v);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);

        // Evaluate s_j at r via Lagrange interpolation through (0,…,degree).
        expected = lagrange_eval(s_j, r);
    }

    if challenges.len() != num_vars {
        return None;
    }
    Some((challenges, expected))
}

// ---------------------------------------------------------------------------
// Zero-sum masking helper
// ---------------------------------------------------------------------------

/// Sample a random multilinear polynomial Z: {0,1}^ell -> F with ∑_x Z(x) = 0.
///
/// Returns the 2^ell evaluation table. The last entry is determined by the
/// rest so that the sum is exactly zero.
pub fn sample_zero_sum_masking<F: PrimeField + UniformRand, R: Rng>(
    ell: usize,
    rng: &mut R,
) -> Vec<F> {
    let n = 1usize << ell;
    let mut z: Vec<F> = (0..n - 1).map(|_| F::rand(rng)).collect();
    let sum: F = z.iter().copied().sum();
    z.push(-sum);
    z
}

/// Sample a degree-3-per-variable zero-sum masking polynomial for the outer sumcheck.
///
/// Returns four multilinear evaluation tables `(ze, za, zb, zc)`, each of size `2^ell`,
/// defining `Z_out(x) = ze(x)·(za(x)·zb(x) − zc(x))` with `∑_{x∈{0,1}^ell} Z_out(x) = 0`.
///
/// The zero-sum constraint is enforced by adjusting `zc[n-1]` after all other
/// entries are sampled uniformly: we set `zc[n-1]` so that
/// `∑ ze[x]·zc[x] = ∑ ze[x]·za[x]·zb[x]`.
pub fn sample_degree3_zero_sum_masking<F: PrimeField + UniformRand, R: Rng>(
    ell: usize,
    rng: &mut R,
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let n = 1usize << ell;
    let ze: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();
    let za: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();
    let zb: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();
    let mut zc: Vec<F> = (0..n - 1).map(|_| F::rand(rng)).collect();
    let target: F = (0..n).map(|x| ze[x] * za[x] * zb[x]).sum();
    let partial: F = (0..n - 1).map(|x| ze[x] * zc[x]).sum();
    let last_ze = ze[n - 1];
    let last_zc = if last_ze.is_zero() {
        F::zero() // negligible probability; zero-sum holds trivially when ze[n-1]=0
    } else {
        (target - partial) * last_ze.inverse().unwrap()
    };
    zc.push(last_zc);
    (ze, za, zb, zc)
}

/// Sample a degree-2-per-variable zero-sum masking polynomial for the inner sumcheck.
///
/// Returns two multilinear evaluation tables `(za, zb)`, each of size `2^ell`,
/// defining `Z_in(y) = za(y)·zb(y)` with `∑_{y∈{0,1}^ell} Z_in(y) = 0`.
///
/// Enforced by setting `zb[n-1] = −(∑_{y<n-1} za[y]·zb[y]) / za[n-1]`.
pub fn sample_degree2_zero_sum_masking<F: PrimeField + UniformRand, R: Rng>(
    ell: usize,
    rng: &mut R,
) -> (Vec<F>, Vec<F>) {
    let n = 1usize << ell;
    let za: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();
    let mut zb: Vec<F> = (0..n - 1).map(|_| F::rand(rng)).collect();
    let partial: F = (0..n - 1).map(|y| za[y] * zb[y]).sum();
    let last_za = za[n - 1];
    let last_zb = if last_za.is_zero() {
        F::zero()
    } else {
        -partial * last_za.inverse().unwrap()
    };
    zb.push(last_zb);
    (za, zb)
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

// ---------------------------------------------------------------------------
// eq polynomial helper
// ---------------------------------------------------------------------------

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
        // k = 3 variables.
        // x_int = 3 is 011 in binary (little-endian), so x = (1, 1, 0).
        let k = 3;
        let x_int = 3;

        // Case 1: z matches x.
        let z_match = vec![F::from(1u64), F::from(1u64), F::from(0u64)];
        assert_eq!(eq_eval(x_int, k, &z_match), F::one());

        // Case 2: z does not match x (boolean).
        let z_no_match = vec![F::from(1u64), F::from(0u64), F::from(0u64)];
        assert_eq!(eq_eval(x_int, k, &z_no_match), F::zero());

        // Case 3: z has non-boolean elements.
        // eq(x, z) = ∏ (x_i * z_i + (1-x_i) * (1-z_i))
        // For x = (1, 1, 0):
        // eq(x, z) = z_0 * z_1 * (1 - z_2)
        let z_mixed = vec![F::from(2u64), F::from(3u64), F::from(4u64)];
        let expected = F::from(2u64) * F::from(3u64) * (F::one() - F::from(4u64));
        assert_eq!(eq_eval(x_int, k, &z_mixed), expected);
    }

    #[test]
    fn test_lagrange_eval() {
        // Degree 1: ys = [3, 7] → p(x) = 4x + 3
        // p(0)=3, p(1)=7, p(2)=11 (extrapolation)
        let ys1 = vec![F::from(3u64), F::from(7u64)];
        assert_eq!(lagrange_eval(&ys1, F::from(0u64)), F::from(3u64));
        assert_eq!(lagrange_eval(&ys1, F::from(1u64)), F::from(7u64));
        assert_eq!(lagrange_eval(&ys1, F::from(2u64)), F::from(11u64));

        // Degree 2: ys = [1, 4, 9] → p(x) = x² + 2x + 1 = (x+1)²
        // p(0)=1, p(1)=4, p(2)=9, p(3)=16 (extrapolation)
        let ys2 = vec![F::from(1u64), F::from(4u64), F::from(9u64)];
        assert_eq!(lagrange_eval(&ys2, F::from(0u64)), F::from(1u64));
        assert_eq!(lagrange_eval(&ys2, F::from(1u64)), F::from(4u64));
        assert_eq!(lagrange_eval(&ys2, F::from(2u64)), F::from(9u64));
        assert_eq!(lagrange_eval(&ys2, F::from(3u64)), F::from(16u64));
    }

    #[test]
    fn test_zero_sum_masking_degree1() {
        let mut rng = rng();
        for ell in 1..=5 {
            let z = sample_zero_sum_masking::<F, _>(ell, &mut rng);
            assert_eq!(z.len(), 1 << ell);
            let sum: F = z.iter().copied().sum();
            assert_eq!(sum, F::zero(), "zero-sum violated for ell={ell}");
        }
    }

    #[test]
    fn test_zero_sum_masking_degree2() {
        let mut rng = rng();
        for ell in 1..=5 {
            let (za, zb) = sample_degree2_zero_sum_masking::<F, _>(ell, &mut rng);
            assert_eq!(za.len(), 1 << ell);
            assert_eq!(zb.len(), 1 << ell);
            let sum: F = za.iter().zip(zb.iter()).map(|(&a, &b)| a * b).sum();
            assert_eq!(sum, F::zero(), "degree-2 zero-sum violated for ell={ell}");
        }
    }

    #[test]
    fn test_zero_sum_masking_degree3() {
        let mut rng = rng();
        for ell in 1..=5 {
            let (ze, za, zb, zc) = sample_degree3_zero_sum_masking::<F, _>(ell, &mut rng);
            assert_eq!(ze.len(), 1 << ell);
            let sum: F = (0..(1 << ell)).map(|x| ze[x] * (za[x] * zb[x] - zc[x])).sum();
            assert_eq!(sum, F::zero(), "degree-3 zero-sum violated for ell={ell}");
        }
    }

    #[test]
    fn test_sumcheck_verify_rejects_wrong_poly_length() {
        // degree=1 requires 2 evaluations per round; providing 1 must fail.
        let proof = SumcheckProof { round_polys: vec![vec![F::from(2u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_consistency_failure() {
        // s_0(0) + s_0(1) = 1+1 = 2, but claimed_sum = 5.
        let proof = SumcheckProof { round_polys: vec![vec![F::from(1u64), F::from(1u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(5u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_too_few_rounds() {
        // 1 round provided but num_vars=2; consistency passes (1+1=2), then
        // challenges.len()=1 != 2 → None.
        let proof = SumcheckProof { round_polys: vec![vec![F::from(1u64), F::from(1u64)]] };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 2, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_success() {
        // Zero-variable sumcheck: empty proof, any claimed sum is accepted as the final value.
        let proof = SumcheckProof::<F> { round_polys: vec![] };
        let mut t = crate::transcript::Transcript::new(b"test");
        let (challenges, final_eval) =
            sumcheck_verify(&proof, 0, 1, F::from(42u64), &mut t).expect("empty sumcheck should succeed");
        assert!(challenges.is_empty());
        assert_eq!(final_eval, F::from(42u64));
    }
}
