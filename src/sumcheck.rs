//! Sumcheck verification shared by the polynomial IOPs.
//!
//! Each round sends the round polynomial at `0, …, degree`; the verifier uses
//! Lagrange interpolation to evaluate it at the Fiat--Shamir challenge.
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

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
    sumcheck_verify_interleaved(
        proof,
        num_vars,
        degree,
        claimed_sum,
        transcript,
        |_, _, _| Some(()),
    )
}

/// Replay fold commitments immediately after each sumcheck challenge, before
/// deriving the next one. Returning `None` from the hook rejects malformed folds.
pub(crate) fn sumcheck_verify_interleaved<F: PrimeField>(
    proof: &SumcheckProof<F>,
    num_vars: usize,
    degree: usize,
    claimed_sum: F,
    transcript: &mut Transcript,
    mut after_challenge: impl FnMut(usize, F, &mut Transcript) -> Option<()>,
) -> Option<(Vec<F>, F)> {
    if proof.round_polys.len() != num_vars {
        return None;
    }
    transcript.absorb_field(claimed_sum);

    let mut expected = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, s_j) in proof.round_polys.iter().enumerate() {
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
        after_challenge(round, r, transcript)?;
    }

    Some((challenges, expected))
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

    #[test]
    fn test_sumcheck_verify_rejects_wrong_poly_length() {
        let proof = SumcheckProof {
            round_polys: vec![vec![F::from(2u64)]],
        };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_consistency_failure() {
        let proof = SumcheckProof {
            round_polys: vec![vec![F::from(1u64), F::from(1u64)]],
        };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 1, 1, F::from(5u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_rejects_too_few_rounds() {
        let proof = SumcheckProof {
            round_polys: vec![vec![F::from(1u64), F::from(1u64)]],
        };
        let mut t = crate::transcript::Transcript::new(b"test");
        assert!(sumcheck_verify(&proof, 2, 1, F::from(2u64), &mut t).is_none());
    }

    #[test]
    fn test_sumcheck_verify_success() {
        let proof = SumcheckProof::<F> {
            round_polys: vec![],
        };
        let mut t = crate::transcript::Transcript::new(b"test");
        let (challenges, final_eval) = sumcheck_verify(&proof, 0, 1, F::from(42u64), &mut t)
            .expect("empty sumcheck should succeed");
        assert!(challenges.is_empty());
        assert_eq!(final_eval, F::from(42u64));
    }
}
