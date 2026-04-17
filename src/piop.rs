//! The two-layer Spartan PIOP from the chapter.
//!
//! The degree-3 outer sumcheck proves the R1CS identity; the batched degree-2
//! inner sumcheck reduces `Q_A + ρQ_B + γQ_C` to the witness and matrix MLEs at
//! one point. ZK mode adds the degree-matched additive masks whose terminal
//! values are bound in `mask.rs`.
use ark_ff::PrimeField;
use ark_relations::gr1cs::Matrix;
use ark_serialize::CanonicalSerialize;
use ark_std::{rand::Rng, UniformRand};

use crate::{
    r1cs::{build_eq_table, mle_of_matrix_at, mle_of_vector, q_eval, ConstraintMatrices},
    sumcheck::{
        additive_mask_eval, additive_mask_round, additive_mask_sum, sample_additive_mask,
        sumcheck_verify, SumcheckProof,
    },
    transcript::Transcript,
};

/// Per-variable degree of the additive masks: outer summand is degree 3, inner is degree 2.
const OUTER_MASK_DEG: usize = 3;
const INNER_MASK_DEG: usize = 2;

/// ceil(log2(n)), with a minimum of 1.
pub fn ell_for(n: usize) -> usize {
    assert!(n > 0, "ell_for: n must be > 0");
    if n <= 2 { 1 } else { (n - 1).ilog2() as usize + 1 }
}

#[derive(CanonicalSerialize)]
pub struct PiopProof<F: PrimeField> {
    /// Outer sumcheck proof (degree 3, ell_row rounds).
    pub outer_sc: SumcheckProof<F>,
    /// Prover's claims Q_A(x*), Q_B(x*), Q_C(x*).
    pub q_a_claim: F,
    pub q_b_claim: F,
    pub q_c_claim: F,
    /// Inner batched sumcheck proof for (Q_A + ρ·Q_B + γ·Q_C)(x*) (degree 2, ell_col rounds).
    pub inner_sc: SumcheckProof<F>,
    /// Terminal inner-sumcheck point y* ∈ F^ell_col and claimed w̃(y*).
    pub y_star: Vec<F>,
    pub u: F,
    /// Claimed matrix-MLE values Ã(x*,y*), B̃(x*,y*), C̃(x*,y*).
    /// Untrusted until certified by the matrix-evaluation proofs in `snark.rs`.
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
    /// ZK: terminal mask evaluations Z_out(x*), Z_in(y*). Bound to the mask
    /// commitments by the inner-product proofs in `snark.rs` (`mask.rs`).
    pub z_out_eval: Option<F>,
    pub z_in_eval: Option<F>,
    /// ZK: declared mask sums used in the masked sumchecks' initial claims.
    pub z_out_sum: Option<F>,
    pub z_in_sum: Option<F>,
}

/// Row-product table: table[i] = ∑_j M[i,j]·w[j].
/// At boolean x = i this equals Q_M(i).
fn build_mw_table<F: PrimeField>(m: &Matrix<F>, w: &[F], ell_row: usize) -> Vec<F> {
    let n = 1 << ell_row;
    let mut table = vec![F::zero(); n];
    for (i, row) in m.iter().enumerate() {
        for (c, j) in row {
            table[i] += *c * w[*j];
        }
    }
    table
}

/// Column-weight table for the batched inner polynomial at point x*:
///   table[col] = ∑_row (A + ρ·B + γ·C)[row,col] · eq(row, x_star)
/// At boolean y = col this equals (Ã + ρ·B̃ + γ·C̃)(x_star, col).
fn build_abc_col_table<F: PrimeField>(
    a: &Matrix<F>,
    b: &Matrix<F>,
    c: &Matrix<F>,
    x_star: &[F],
    rho: F,
    gamma: F,
    ell_col: usize,
) -> Vec<F> {
    let n_col = 1 << ell_col;
    let mut table = vec![F::zero(); n_col];
    let eq_tbl = build_eq_table(x_star);

    for (i, row) in a.iter().enumerate() {
        let rw = eq_tbl[i];
        for (c_val, j) in row {
            table[*j] += *c_val * rw;
        }
    }
    for (i, row) in b.iter().enumerate() {
        let rw = eq_tbl[i] * rho;
        for (c_val, j) in row {
            table[*j] += *c_val * rw;
        }
    }
    for (i, row) in c.iter().enumerate() {
        let rw = eq_tbl[i] * gamma;
        for (c_val, j) in row {
            table[*j] += *c_val * rw;
        }
    }
    table
}

/// Outer sumcheck: ∑_x eq(x,r)·[A(x)·B(x) − C(x)] (+ τ·Z_out) = τ·S_out, degree 3.
///
/// Each round evaluates s_j(t) at t = 0,1,2,3 using the bookkeeping tables, adds the
/// additive mask's analytic round contribution (scaled by τ), then folds the tables.
/// Returns (proof, challenges x*, Z_out(x*), Σ_x Z_out(x)) — the last two are `None`
/// when not in ZK mode.
fn sumcheck_outer_bookkeeping<F: PrimeField>(
    mut eq_tbl: Vec<F>,
    mut a_tbl: Vec<F>,
    mut b_tbl: Vec<F>,
    mut c_tbl: Vec<F>,
    ell: usize,
    coeffs: Option<Vec<Vec<F>>>,
    mask_combiner: F,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>, Option<F>, Option<F>) {
    let mask_sum = coeffs.as_ref().map(|c| additive_mask_sum(c));

    transcript.absorb_field(mask_combiner * mask_sum.unwrap_or(F::zero()));

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let mut current = 1usize << ell;

    for round_i in 0..ell {
        let half = current / 2;
        let mut s_j = vec![F::zero(); 4];

        for k in 0..half {
            let eq0 = eq_tbl[2 * k];
            let eq1 = eq_tbl[2 * k + 1];
            let a0 = a_tbl[2 * k];
            let a1 = a_tbl[2 * k + 1];
            let b0 = b_tbl[2 * k];
            let b1 = b_tbl[2 * k + 1];
            let c0 = c_tbl[2 * k];
            let c1 = c_tbl[2 * k + 1];

            for t in 0u64..=3 {
                let tf = F::from(t);
                let omtf = F::one() - tf;
                let eq_t = omtf * eq0 + tf * eq1;
                let a_t = omtf * a0 + tf * a1;
                let b_t = omtf * b0 + tf * b1;
                let c_t = omtf * c0 + tf * c1;
                s_j[t as usize] += eq_t * (a_t * b_t - c_t);
            }
        }

        if let Some(ref c) = coeffs {
            let mc = additive_mask_round(c, round_i, &challenges);
            for t in 0..4 {
                s_j[t] += mask_combiner * mc[t];
            }
        }

        for &v in &s_j {
            transcript.absorb_field(v);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);
        round_polys.push(s_j);

        let omr = F::one() - r;
        for k in 0..half {
            eq_tbl[k] = omr * eq_tbl[2 * k] + r * eq_tbl[2 * k + 1];
            a_tbl[k] = omr * a_tbl[2 * k] + r * a_tbl[2 * k + 1];
            b_tbl[k] = omr * b_tbl[2 * k] + r * b_tbl[2 * k + 1];
            c_tbl[k] = omr * c_tbl[2 * k] + r * c_tbl[2 * k + 1];
        }
        current = half;
    }

    let z_eval = coeffs.as_ref().map(|c| additive_mask_eval(c, &challenges));
    (SumcheckProof { round_polys }, challenges, z_eval, mask_sum)
}

/// Inner sumcheck: ∑_y combined(y)·w(y) (+ τ·Z_in) = claimed_sum + τ·S_in, degree 2.
///
/// `claimed_sum` is the unmasked base claim q_abc_claim; the masked claim absorbed is
/// `claimed_sum + τ·S_in`. Returns (proof, challenges y*, Z_in(y*), Σ_y Z_in(y)).
fn sumcheck_inner_bookkeeping<F: PrimeField>(
    mut a_tbl: Vec<F>,
    mut w_tbl: Vec<F>,
    ell: usize,
    claimed_sum: F,
    coeffs: Option<Vec<Vec<F>>>,
    mask_combiner: F,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>, Option<F>, Option<F>) {
    let mask_sum = coeffs.as_ref().map(|c| additive_mask_sum(c));

    transcript.absorb_field(claimed_sum + mask_combiner * mask_sum.unwrap_or(F::zero()));

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let two = F::from(2u64);
    let mut current = 1usize << ell;

    for round_i in 0..ell {
        let half = current / 2;
        let mut s_j = vec![F::zero(); 3];

        for k in 0..half {
            let a0 = a_tbl[2 * k];
            let a1 = a_tbl[2 * k + 1];
            let w0 = w_tbl[2 * k];
            let w1 = w_tbl[2 * k + 1];

            s_j[0] += a0 * w0;
            s_j[1] += a1 * w1;
            s_j[2] += (two * a1 - a0) * (two * w1 - w0);
        }

        if let Some(ref c) = coeffs {
            let mc = additive_mask_round(c, round_i, &challenges);
            for t in 0..3 {
                s_j[t] += mask_combiner * mc[t];
            }
        }

        for &v in &s_j {
            transcript.absorb_field(v);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);
        round_polys.push(s_j);

        let omr = F::one() - r;
        for k in 0..half {
            a_tbl[k] = omr * a_tbl[2 * k] + r * a_tbl[2 * k + 1];
            w_tbl[k] = omr * w_tbl[2 * k] + r * w_tbl[2 * k + 1];
        }
        current = half;
    }

    let z_eval = coeffs.as_ref().map(|c| additive_mask_eval(c, &challenges));
    (SumcheckProof { round_polys }, challenges, z_eval, mask_sum)
}

/// Commit the declared mask sum to the Fiat--Shamir transcript before deriving τ.
fn bind_mask_sum<F: PrimeField>(mask_sum: Option<F>, transcript: &mut Transcript) -> F {
    if let Some(sum) = mask_sum {
        transcript.absorb_field(sum);
        transcript.squeeze_field()
    } else {
        F::zero()
    }
}

pub fn piop_prove<F: PrimeField + UniformRand, R: Rng>(
    matrices: &ConstraintMatrices<F>,
    w: &[F],
    ell_row: usize,
    ell_col: usize,
    transcript: &mut Transcript,
    zk: bool,
    z_out_coeffs: Option<Vec<Vec<F>>>,
    z_in_coeffs: Option<Vec<Vec<F>>>,
    rng: &mut R,
) -> (PiopProof<F>, Vec<F>, Vec<F>, F) {
    assert!(!matrices.a.is_empty(), "piop_prove: circuit has no constraints");

    // `w` is the full half-split assignment described in `snark.rs`.
    let mut w_pad = w.to_vec();
    w_pad.resize(1 << ell_col, F::zero());

    let z_out_coeffs = if zk {
        Some(z_out_coeffs.unwrap_or_else(|| sample_additive_mask(ell_row, OUTER_MASK_DEG, rng)))
    } else {
        None
    };
    let z_in_coeffs = if zk {
        Some(z_in_coeffs.unwrap_or_else(|| sample_additive_mask(ell_col, INNER_MASK_DEG, rng)))
    } else {
        None
    };
    let declared_z_out_sum = z_out_coeffs.as_ref().map(|c| additive_mask_sum(c));
    let declared_z_in_sum = z_in_coeffs.as_ref().map(|c| additive_mask_sum(c));

    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    // The mask root and declared sum are both bound before this challenge.
    let tau_out = bind_mask_sum(declared_z_out_sum, transcript);

    let eq_tbl = build_eq_table(&r);
    let qa_tbl = build_mw_table(&matrices.a, &w_pad, ell_row);
    let qb_tbl = build_mw_table(&matrices.b, &w_pad, ell_row);
    let qc_tbl = build_mw_table(&matrices.c, &w_pad, ell_row);

    let (outer_sc, x_star, z_out_eval, z_out_sum) = sumcheck_outer_bookkeeping(
        eq_tbl, qa_tbl, qb_tbl, qc_tbl, ell_row, z_out_coeffs, tau_out, transcript,
    );
    debug_assert_eq!(z_out_sum, declared_z_out_sum);

    let q_a_claim = q_eval(&matrices.a, &w_pad, &x_star);
    let q_b_claim = q_eval(&matrices.b, &w_pad, &x_star);
    let q_c_claim = q_eval(&matrices.c, &w_pad, &x_star);

    transcript.absorb_field(q_a_claim);
    transcript.absorb_field(q_b_claim);
    transcript.absorb_field(q_c_claim);
    if let Some(z) = z_out_eval {
        transcript.absorb_field(z);
    }

    let rho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = q_a_claim + rho * q_b_claim + gamma * q_c_claim;

    let tau_in = bind_mask_sum(declared_z_in_sum, transcript);

    let abc_tbl = build_abc_col_table(
        &matrices.a, &matrices.b, &matrices.c,
        &x_star, rho, gamma, ell_col,
    );
    let w_tbl = w_pad.clone();

    let (inner_sc, y_star, z_in_eval, z_in_sum) = sumcheck_inner_bookkeeping(
        abc_tbl, w_tbl, ell_col, q_abc_claim, z_in_coeffs, tau_in, transcript,
    );
    debug_assert_eq!(z_in_sum, declared_z_in_sum);

    let u = mle_of_vector(&w_pad, ell_col, &y_star);

    transcript.absorb_field(u);
    if let Some(z) = z_in_eval {
        transcript.absorb_field(z);
    }

    // These matrix claims are certified against the preprocessing commitments later.
    let a_eval = mle_of_matrix_at(&matrices.a, &x_star, &y_star);
    let b_eval = mle_of_matrix_at(&matrices.b, &x_star, &y_star);
    let c_eval = mle_of_matrix_at(&matrices.c, &x_star, &y_star);
    transcript.absorb_field(a_eval);
    transcript.absorb_field(b_eval);
    transcript.absorb_field(c_eval);

    let proof = PiopProof {
        outer_sc,
        q_a_claim,
        q_b_claim,
        q_c_claim,
        inner_sc,
        y_star: y_star.clone(),
        u,
        a_eval,
        b_eval,
        c_eval,
        z_out_eval,
        z_in_eval,
        z_out_sum,
        z_in_sum,
    };
    (proof, x_star, y_star, u)
}

/// Returns `Some((x*, y*, u))` — the outer/inner terminal points and claimed witness MLE value.
///
/// The matrix MLEs Ã,B̃,C̃ at (x*,y*) are taken as the *claimed* values
/// `proof.{a,b,c}_eval`; the caller (snark verifier) must certify those against
/// the preprocessing commitments via the matrix-evaluation proofs, and must still
/// check the PCS openings of Z_out(x*), Z_in(y*), ŵ(y*).
pub fn piop_verify<F: PrimeField>(
    ell_row: usize,
    ell_col: usize,
    proof: &PiopProof<F>,
    transcript: &mut Transcript,
) -> Option<(Vec<F>, Vec<F>, F)> {
    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    let tau_out = match (proof.z_out_eval, proof.z_out_sum) {
        (Some(_), Some(sum)) => bind_mask_sum(Some(sum), transcript),
        (None, None) => F::zero(),
        _ => return None,
    };

    let claimed_outer = tau_out * proof.z_out_sum.unwrap_or(F::zero());
    let (x_star, outer_final) =
        sumcheck_verify(&proof.outer_sc, ell_row, 3, claimed_outer, transcript)?;

    let eq_val: F = (0..ell_row)
        .map(|j| x_star[j] * r[j] + (F::one() - x_star[j]) * (F::one() - r[j]))
        .product();
    let oracle_outer = eq_val * (proof.q_a_claim * proof.q_b_claim - proof.q_c_claim);
    let unmasked_outer = outer_final - tau_out * proof.z_out_eval.unwrap_or(F::zero());
    if unmasked_outer != oracle_outer {
        return None;
    }

    transcript.absorb_field(proof.q_a_claim);
    transcript.absorb_field(proof.q_b_claim);
    transcript.absorb_field(proof.q_c_claim);
    if let Some(z) = proof.z_out_eval {
        transcript.absorb_field(z);
    }

    let rho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = proof.q_a_claim + rho * proof.q_b_claim + gamma * proof.q_c_claim;

    let tau_in = match (proof.z_in_eval, proof.z_in_sum) {
        (Some(_), Some(sum)) => bind_mask_sum(Some(sum), transcript),
        (None, None) => F::zero(),
        _ => return None,
    };

    let claimed_inner = q_abc_claim + tau_in * proof.z_in_sum.unwrap_or(F::zero());
    let (y_star, inner_final) =
        sumcheck_verify(&proof.inner_sc, ell_col, 2, claimed_inner, transcript)?;

    let u = proof.u;

    let expected_abc = (proof.a_eval + rho * proof.b_eval + gamma * proof.c_eval) * u;
    let unmasked_abc = inner_final - tau_in * proof.z_in_eval.unwrap_or(F::zero());
    if unmasked_abc != expected_abc {
        return None;
    }

    transcript.absorb_field(u);
    if let Some(z) = proof.z_in_eval {
        transcript.absorb_field(z);
    }

    transcript.absorb_field(proof.a_eval);
    transcript.absorb_field(proof.b_eval);
    transcript.absorb_field(proof.c_eval);

    Some((x_star, y_star, u))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{One, Zero};

    #[test]
    fn test_ell_for() {
        assert_eq!(ell_for(1), 1);
        assert_eq!(ell_for(2), 1);
        assert_eq!(ell_for(3), 2);
        assert_eq!(ell_for(4), 2);
        assert_eq!(ell_for(5), 3);
        assert_eq!(ell_for(8), 3);
        assert_eq!(ell_for(9), 4);
    }

    #[test]
    fn mask_sum_is_bound_before_combiner() {
        let mut first = Transcript::new(b"mask-order");
        let mut second = Transcript::new(b"mask-order");

        let tau_first = bind_mask_sum(Some(F::from(1u64)), &mut first);
        let tau_second = bind_mask_sum(Some(F::from(2u64)), &mut second);

        assert_ne!(tau_first, tau_second);
    }

    #[test]
    fn test_build_mw_table() {
        let m: Matrix<F> = vec![
            vec![(F::from(2u64), 0)],
            vec![(F::from(3u64), 1)],
        ];
        let w = vec![F::from(5u64), F::from(7u64)];
        let table = build_mw_table(&m, &w, 1);

        assert_eq!(table.len(), 2);
        assert_eq!(table[0], F::from(10u64));
        assert_eq!(table[1], F::from(21u64));
    }

    #[test]
    fn test_build_abc_col_table_zero() {
        let empty: Matrix<F> = vec![vec![], vec![]];
        let x_star = vec![F::from(0u64)];
        let table = build_abc_col_table(&empty, &empty, &empty, &x_star, F::one(), F::one(), 1);
        assert!(table.iter().all(|&v| v == F::zero()));
    }
}
