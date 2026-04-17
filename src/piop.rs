/// R1CS PIOP: outer sumcheck (degree 3) + batched inner sumcheck (degree 2).
///
/// Protocol (matching book §SNARK):
///   - Outer sumcheck (ell_row rounds, deg 3): proves
///       ∑_{x ∈ {0,1}^ell_row} eq(x,r)·[Q_A(x)·Q_B(x)−Q_C(x)] = 0
///   - Verifier receives Q_A(x*), Q_B(x*), Q_C(x*) then sends batching scalars ρ, γ.
///   - Inner sumcheck (ell_col rounds, deg 2): proves
///       ∑_{y ∈ {0,1}^ell_col} [Ã(x*,y)+ρ·B̃(x*,y)+γ·C̃(x*,y)]·w̃(y) = Q_A(x*)+ρ·Q_B(x*)+γ·Q_C(x*)
///     producing terminal point y* and claimed w̃(y*) = u.
///   - Verifier checks (Ã(x*,y*)+ρ·B̃(x*,y*)+γ·C̃(x*,y*))·u = inner_final.
///
/// ZK variant adds one degree-matched zero-sum mask per sumcheck phase:
///   Z_out = ZE·(ZA·ZB−ZC) for the outer (degree 3; overhead: z_out_eval = Z_out(x*))
///   Z_in  = Zα·Zβ         for the inner (degree 2; overhead: z_in_eval = Z_in(y*))
///
/// Bookkeeping tables give O(nnz + 2^ell) complexity instead of O(ell × nnz × 2^ell).
use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintMatrices, Matrix};
use ark_std::{rand::Rng, UniformRand};

use crate::{
    r1cs::{build_eq_table, mle_of_matrix_at, mle_of_vector, q_eval},
    sumcheck::{
        sample_degree2_zero_sum_masking, sample_degree3_zero_sum_masking, sumcheck_verify,
        SumcheckProof,
    },
    transcript::Transcript,
};

// ---------------------------------------------------------------------------
// ell helper
// ---------------------------------------------------------------------------

/// ceil(log2(n)), with a minimum of 1.
pub fn ell_for(n: usize) -> usize {
    assert!(n > 0, "ell_for: n must be > 0");
    if n <= 2 { 1 } else { (n - 1).ilog2() as usize + 1 }
}

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

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
    /// ZK: accumulated masking evaluations (2 extra field elements total).
    pub z_out_eval: Option<F>,
    pub z_in_eval: Option<F>,
}

// ---------------------------------------------------------------------------
// Bookkeeping table helpers (private to this module)
// ---------------------------------------------------------------------------

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

/// Outer sumcheck: ∑_x eq(x,r)·[A(x)·B(x) − C(x)] = 0, degree 3.
///
/// Each round evaluates s_j(t) at t = 0,1,2,3 using the bookkeeping tables,
/// then folds all tables with the Fiat-Shamir challenge.
/// Returns (proof, challenges x*, optional z_eval for ZK masking).
fn sumcheck_outer_bookkeeping<F: PrimeField + UniformRand, R: Rng>(
    mut eq_tbl: Vec<F>,
    mut a_tbl: Vec<F>,
    mut b_tbl: Vec<F>,
    mut c_tbl: Vec<F>,
    ell: usize,
    zk: bool,
    z_tbl_in: Option<(Vec<F>, Vec<F>, Vec<F>, Vec<F>)>,
    rng: &mut R,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>, Option<F>) {
    transcript.absorb_field(F::zero()); // claimed sum = 0

    // ZK: degree-3-per-variable masking Z_out(x) = ze(x)·(za(x)·zb(x) − zc(x)).
    let mut z_tbl: Option<(Vec<F>, Vec<F>, Vec<F>, Vec<F>)> = if zk {
        Some(z_tbl_in.unwrap_or_else(|| sample_degree3_zero_sum_masking(ell, rng)))
    } else {
        None
    };

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let mut current = 1usize << ell;

    for _ in 0..ell {
        let half = current / 2;
        let mut s_j = vec![F::zero(); 4]; // evaluations at t = 0, 1, 2, 3

        for k in 0..half {
            let eq0 = eq_tbl[2 * k];
            let eq1 = eq_tbl[2 * k + 1];
            let a0 = a_tbl[2 * k];
            let a1 = a_tbl[2 * k + 1];
            let b0 = b_tbl[2 * k];
            let b1 = b_tbl[2 * k + 1];
            let c0 = c_tbl[2 * k];
            let c1 = c_tbl[2 * k + 1];

            // Each factor is linear in t: X(t) = (1-t)·X0 + t·X1.
            // s_j(t) += eq(t)·(a(t)·b(t) − c(t)), evaluated at t = 0,1,2,3.
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

        // ZK: add degree-3 masking Z_out(t) = ze(t)·(za(t)·zb(t) − zc(t)).
        if let Some((ref ze, ref za, ref zb, ref zc)) = z_tbl {
            for k in 0..half {
                let ze0 = ze[2 * k];     let ze1 = ze[2 * k + 1];
                let za0 = za[2 * k];     let za1 = za[2 * k + 1];
                let zb0 = zb[2 * k];     let zb1 = zb[2 * k + 1];
                let zc0 = zc[2 * k];     let zc1 = zc[2 * k + 1];
                for t in 0u64..=3 {
                    let tf = F::from(t);
                    let omtf = F::one() - tf;
                    let ze_t = omtf * ze0 + tf * ze1;
                    let za_t = omtf * za0 + tf * za1;
                    let zb_t = omtf * zb0 + tf * zb1;
                    let zc_t = omtf * zc0 + tf * zc1;
                    s_j[t as usize] += ze_t * (za_t * zb_t - zc_t);
                }
            }
        }

        for &v in &s_j {
            transcript.absorb_field(v);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);
        round_polys.push(s_j);

        // Fold all tables with the challenge r.
        let omr = F::one() - r;
        for k in 0..half {
            eq_tbl[k] = omr * eq_tbl[2 * k] + r * eq_tbl[2 * k + 1];
            a_tbl[k] = omr * a_tbl[2 * k] + r * a_tbl[2 * k + 1];
            b_tbl[k] = omr * b_tbl[2 * k] + r * b_tbl[2 * k + 1];
            c_tbl[k] = omr * c_tbl[2 * k] + r * c_tbl[2 * k + 1];
            if let Some((ref mut ze, ref mut za, ref mut zb, ref mut zc)) = z_tbl {
                ze[k] = omr * ze[2 * k] + r * ze[2 * k + 1];
                za[k] = omr * za[2 * k] + r * za[2 * k + 1];
                zb[k] = omr * zb[2 * k] + r * zb[2 * k + 1];
                zc[k] = omr * zc[2 * k] + r * zc[2 * k + 1];
            }
        }
        current = half;
    }

    let z_eval = z_tbl.map(|(ze, za, zb, zc)| ze[0] * (za[0] * zb[0] - zc[0]));
    (SumcheckProof { round_polys }, challenges, z_eval)
}

/// Inner sumcheck: ∑_y combined(y)·w(y) = claimed_sum, degree 2.
///
/// Each round evaluates s_j(t) at t = 0,1,2 using the bookkeeping tables,
/// then folds both tables with the Fiat-Shamir challenge.
/// Returns (proof, challenges y*, optional z_eval for ZK masking).
fn sumcheck_inner_bookkeeping<F: PrimeField + UniformRand, R: Rng>(
    mut a_tbl: Vec<F>,  // (Ã + ρ·B̃ + γ·C̃)(x*, ·) at boolean points
    mut w_tbl: Vec<F>,  // w̃(·) at boolean points = w_pad itself
    ell: usize,
    claimed_sum: F,
    zk: bool,
    z_tbl_in: Option<(Vec<F>, Vec<F>)>,
    rng: &mut R,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>, Option<F>) {
    transcript.absorb_field(claimed_sum);

    // ZK: degree-2-per-variable masking Z_in(y) = za(y)·zb(y).
    let mut z_tbl: Option<(Vec<F>, Vec<F>)> = if zk {
        Some(z_tbl_in.unwrap_or_else(|| sample_degree2_zero_sum_masking(ell, rng)))
    } else {
        None
    };

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let two = F::from(2u64);
    let mut current = 1usize << ell;

    for _ in 0..ell {
        let half = current / 2;
        let mut s_j = vec![F::zero(); 3]; // evaluations at t = 0, 1, 2

        for k in 0..half {
            let a0 = a_tbl[2 * k];
            let a1 = a_tbl[2 * k + 1];
            let w0 = w_tbl[2 * k];
            let w1 = w_tbl[2 * k + 1];

            // t=0: a0·w0
            s_j[0] += a0 * w0;
            // t=1: a1·w1
            s_j[1] += a1 * w1;
            // t=2: (2·a1 − a0)·(2·w1 − w0)
            s_j[2] += (two * a1 - a0) * (two * w1 - w0);
        }

        // ZK: add degree-2 masking Z_in(t) = za(t)·zb(t).
        if let Some((ref za, ref zb)) = z_tbl {
            for k in 0..half {
                let za0 = za[2 * k]; let za1 = za[2 * k + 1];
                let zb0 = zb[2 * k]; let zb1 = zb[2 * k + 1];
                s_j[0] += za0 * zb0;
                s_j[1] += za1 * zb1;
                s_j[2] += (two * za1 - za0) * (two * zb1 - zb0);
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
            if let Some((ref mut za, ref mut zb)) = z_tbl {
                za[k] = omr * za[2 * k] + r * za[2 * k + 1];
                zb[k] = omr * zb[2 * k] + r * zb[2 * k + 1];
            }
        }
        current = half;
    }

    let z_eval = z_tbl.map(|(za, zb)| za[0] * zb[0]);
    (SumcheckProof { round_polys }, challenges, z_eval)
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn piop_prove<F: PrimeField + UniformRand, R: Rng>(
    matrices: &ConstraintMatrices<F>,
    w: &[F],
    transcript: &mut Transcript,
    zk: bool,
    z_out_vec: Option<(Vec<F>, Vec<F>, Vec<F>, Vec<F>)>,
    z_in_vec: Option<(Vec<F>, Vec<F>)>,
    rng: &mut R,
) -> (PiopProof<F>, Vec<F>, Vec<F>, F) {
    assert!(!matrices.a.is_empty(), "piop_prove: circuit has no constraints");
    let ell_row = ell_for(matrices.num_constraints);
    let num_vars = matrices.num_instance_variables + matrices.num_witness_variables;
    let ell_col = ell_for(num_vars);

    // Pad witness to 2^ell_col.
    let mut w_pad = w.to_vec();
    w_pad.resize(1 << ell_col, F::zero());

    // --- Verifier's outer challenge r (Fiat-Shamir) -------------------------
    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    // --- Build bookkeeping tables for the outer sumcheck --------------------
    let eq_tbl = build_eq_table(&r);
    let qa_tbl = build_mw_table(&matrices.a, &w_pad, ell_row);
    let qb_tbl = build_mw_table(&matrices.b, &w_pad, ell_row);
    let qc_tbl = build_mw_table(&matrices.c, &w_pad, ell_row);

    // --- Outer sumcheck: ∑_x eq(x,r)·[QA(x)·QB(x) − QC(x)] = 0 -----------
    let (outer_sc, x_star, z_out_eval) = sumcheck_outer_bookkeeping(
        eq_tbl, qa_tbl, qb_tbl, qc_tbl, ell_row, zk, z_out_vec, rng, transcript,
    );

    // --- Q_A, Q_B, Q_C at x* -----------------------------------------------
    let q_a_claim = q_eval(&matrices.a, &w_pad, &x_star);
    let q_b_claim = q_eval(&matrices.b, &w_pad, &x_star);
    let q_c_claim = q_eval(&matrices.c, &w_pad, &x_star);

    transcript.absorb_field(q_a_claim);
    transcript.absorb_field(q_b_claim);
    transcript.absorb_field(q_c_claim);
    if let Some(z) = z_out_eval {
        transcript.absorb_field(z);
    }

    // --- Batching challenges ρ, γ --------------------------------------------
    let rho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = q_a_claim + rho * q_b_claim + gamma * q_c_claim;

    // --- Build bookkeeping tables for the inner sumcheck --------------------
    // combined(col) = (Ã+ρ·B̃+γ·C̃)(x*, col_bits) at boolean col points.
    let abc_tbl = build_abc_col_table(
        &matrices.a, &matrices.b, &matrices.c,
        &x_star, rho, gamma, ell_col,
    );
    // w_tbl: at boolean y, w̃(y) = w_pad[y] (MLE of w_pad evaluated at hypercube = w_pad itself).
    let w_tbl = w_pad.clone();

    // --- Inner sumcheck: ∑_y combined(y)·w(y) = q_abc_claim ----------------
    let (inner_sc, y_star, z_in_eval) = sumcheck_inner_bookkeeping(
        abc_tbl, w_tbl, ell_col, q_abc_claim, zk, z_in_vec, rng, transcript,
    );

    let u = mle_of_vector(&w_pad, ell_col, &y_star);

    transcript.absorb_field(u);
    if let Some(z) = z_in_eval {
        transcript.absorb_field(z);
    }

    let proof = PiopProof {
        outer_sc,
        q_a_claim,
        q_b_claim,
        q_c_claim,
        inner_sc,
        y_star: y_star.clone(),
        u,
        z_out_eval,
        z_in_eval,
    };
    (proof, x_star, y_star, u)
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Returns `Some((x*, y*, u))` — the outer/inner terminal points and claimed witness MLE value.
/// The caller (snark verifier) must still check the WHIR openings of Z_out(x*), Z_in(y*), ŵ(y*).
pub fn piop_verify<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    proof: &PiopProof<F>,
    transcript: &mut Transcript,
) -> Option<(Vec<F>, Vec<F>, F)> {
    let ell_row = ell_for(matrices.num_constraints);
    let num_vars = matrices.num_instance_variables + matrices.num_witness_variables;
    let ell_col = ell_for(num_vars);

    // --- Outer challenge r ---------------------------------------------------
    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    // --- Outer sumcheck verify -----------------------------------------------
    let (x_star, outer_final) =
        sumcheck_verify(&proof.outer_sc, ell_row, 3, F::zero(), transcript)?;

    // Oracle check: (outer_final − z_out) = eq(x*,r)·[Q_A·Q_B − Q_C]
    let eq_val: F = (0..ell_row)
        .map(|j| x_star[j] * r[j] + (F::one() - x_star[j]) * (F::one() - r[j]))
        .product();
    let oracle_outer = eq_val * (proof.q_a_claim * proof.q_b_claim - proof.q_c_claim);
    let unmasked_outer = outer_final - proof.z_out_eval.unwrap_or(F::zero());
    if unmasked_outer != oracle_outer {
        return None;
    }

    transcript.absorb_field(proof.q_a_claim);
    transcript.absorb_field(proof.q_b_claim);
    transcript.absorb_field(proof.q_c_claim);
    if let Some(z) = proof.z_out_eval {
        transcript.absorb_field(z);
    }

    // --- Batching challenges ρ, γ --------------------------------------------
    let rho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = proof.q_a_claim + rho * proof.q_b_claim + gamma * proof.q_c_claim;

    // --- Inner sumcheck verify -----------------------------------------------
    let (y_star, inner_final) =
        sumcheck_verify(&proof.inner_sc, ell_col, 2, q_abc_claim, transcript)?;

    // Verifier evaluates matrix MLEs at (x*, y*).
    let a_val = mle_of_matrix_at(&matrices.a, &x_star, &y_star);
    let b_val = mle_of_matrix_at(&matrices.b, &x_star, &y_star);
    let c_val = mle_of_matrix_at(&matrices.c, &x_star, &y_star);
    let u = proof.u;

    // Inner oracle check: (a_val + ρ·b_val + γ·c_val)·u = inner_final − z_in.
    let expected_abc = (a_val + rho * b_val + gamma * c_val) * u;
    let unmasked_abc = inner_final - proof.z_in_eval.unwrap_or(F::zero());
    if unmasked_abc != expected_abc {
        return None;
    }

    transcript.absorb_field(u);
    if let Some(z) = proof.z_in_eval {
        transcript.absorb_field(z);
    }

    Some((x_star, y_star, u))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{One, Zero};

    #[test]
    fn test_ell_for() {
        assert_eq!(ell_for(1), 1); // minimum
        assert_eq!(ell_for(2), 1); // 2 fits in 1 bit
        assert_eq!(ell_for(3), 2); // needs 2 bits
        assert_eq!(ell_for(4), 2); // 4 fits in 2 bits
        assert_eq!(ell_for(5), 3); // needs 3 bits
        assert_eq!(ell_for(8), 3); // 8 fits in 3 bits
        assert_eq!(ell_for(9), 4); // needs 4 bits
    }

    #[test]
    fn test_build_mw_table() {
        // A[0,0] = 2, A[1,1] = 3
        let m: Matrix<F> = vec![
            vec![(F::from(2u64), 0)],
            vec![(F::from(3u64), 1)],
        ];
        let w = vec![F::from(5u64), F::from(7u64)];
        let table = build_mw_table(&m, &w, 1); // ell_row=1 → 2 rows

        assert_eq!(table.len(), 2);
        assert_eq!(table[0], F::from(10u64)); // row 0: 2*5
        assert_eq!(table[1], F::from(21u64)); // row 1: 3*7
    }

    #[test]
    fn test_build_abc_col_table_zero() {
        // Empty matrices → all-zero table.
        let empty: Matrix<F> = vec![vec![], vec![]];
        let x_star = vec![F::from(0u64)];
        let table = build_abc_col_table(&empty, &empty, &empty, &x_star, F::one(), F::one(), 1);
        assert!(table.iter().all(|&v| v == F::zero()));
    }
}
