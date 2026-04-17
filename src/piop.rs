//! The non-hiding two-layer Spartan PIOP.
//! Chapter: Construction 12 (Circuit-SAT via R1CS and global Sumcheck), Steps 2–3.
//!
//! The degree-3 outer sumcheck proves the R1CS identity; the batched degree-2
//! inner sumcheck reduces `Q_A + ϱQ_B + γQ_C` to the witness and matrix MLEs at
//! one point. ZK proofs use the masked reduction in `zk_piop`.
use ark_ff::PrimeField;
use ark_relations::gr1cs::Matrix;
use ark_serialize::CanonicalSerialize;

use crate::{
    r1cs::{ConstraintMatrices, build_eq_table, mle_of_matrix_at, mle_of_vector, q_eval},
    sumcheck::{SumcheckProof, sumcheck_verify},
    transcript::Transcript,
};

/// ceil(log2(n)), with a minimum of 1.
pub fn ell_for(n: usize) -> usize {
    assert!(n > 0, "ell_for: n must be > 0");
    if n <= 2 {
        1
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

#[derive(CanonicalSerialize)]
pub struct PiopProof<F: PrimeField> {
    /// Step 2: outer sumcheck proof (degree 3, ell_row rounds).
    pub outer_sc: SumcheckProof<F>,
    /// Step 3: prover's claims Q_A(x*), Q_B(x*), Q_C(x*).
    pub q_a_claim: F,
    pub q_b_claim: F,
    pub q_c_claim: F,
    /// Step 3: inner batched sumcheck proof for (Q_A + ϱ·Q_B + γ·Q_C)(x*) (degree 2, ell_col rounds).
    pub inner_sc: SumcheckProof<F>,
    /// The claim u = w̃(y*), opened in Step 4 by `snark.rs`.
    pub u: F,
    /// Claimed matrix-MLE values Ã(x*,y*), B̃(x*,y*), C̃(x*,y*).
    /// Untrusted until certified by the matrix-evaluation proofs in `snark.rs`.
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
}

/// Table of the row-product polynomial Q_M (Section 7.1): table[i] = ∑_j M[i,j]·w[j].
/// At boolean x = i this equals Q_M(i).
pub(crate) fn build_q_table<F: PrimeField>(m: &Matrix<F>, w: &[F], ell_row: usize) -> Vec<F> {
    let n = 1 << ell_row;
    let mut table = vec![F::zero(); n];
    for (i, row) in m.iter().enumerate() {
        for (c, j) in row {
            table[i] += *c * w[*j];
        }
    }
    table
}

/// Table of L(y) = Ã(x*, y) + ϱ·B̃(x*, y) + γ·C̃(x*, y) (as in Construction 13):
///   table[col] = ∑_row (A + ϱ·B + γ·C)[row,col] · eq(row, x_star)
pub(crate) fn build_l_table<F: PrimeField>(
    a: &Matrix<F>,
    b: &Matrix<F>,
    c: &Matrix<F>,
    x_star: &[F],
    varrho: F,
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
        let rw = eq_tbl[i] * varrho;
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

/// Step 2: degree-3 sumcheck of eq(x,r)·[Q_A(x)·Q_B(x) − Q_C(x)] with claimed sum zero,
/// i.e. equation (3) of Section 7.1.
fn sumcheck_outer_bookkeeping<F: PrimeField>(
    mut eq_tbl: Vec<F>,
    mut qa_tbl: Vec<F>,
    mut qb_tbl: Vec<F>,
    mut qc_tbl: Vec<F>,
    ell: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>) {
    transcript.absorb_field(F::zero());

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let mut current = 1usize << ell;

    for _ in 0..ell {
        let half = current / 2;
        let mut g_i = vec![F::zero(); 4];

        for k in 0..half {
            let eq0 = eq_tbl[2 * k];
            let eq1 = eq_tbl[2 * k + 1];
            let a0 = qa_tbl[2 * k];
            let a1 = qa_tbl[2 * k + 1];
            let b0 = qb_tbl[2 * k];
            let b1 = qb_tbl[2 * k + 1];
            let c0 = qc_tbl[2 * k];
            let c1 = qc_tbl[2 * k + 1];

            for t in 0u64..=3 {
                let tf = F::from(t);
                let omtf = F::one() - tf;
                let eq_t = omtf * eq0 + tf * eq1;
                let a_t = omtf * a0 + tf * a1;
                let b_t = omtf * b0 + tf * b1;
                let c_t = omtf * c0 + tf * c1;
                g_i[t as usize] += eq_t * (a_t * b_t - c_t);
            }
        }

        for &v in &g_i {
            transcript.absorb_field(v);
        }
        let x_i = transcript.squeeze_field::<F>();
        challenges.push(x_i);
        round_polys.push(g_i);

        let one_minus_x_i = F::one() - x_i;
        for k in 0..half {
            eq_tbl[k] = one_minus_x_i * eq_tbl[2 * k] + x_i * eq_tbl[2 * k + 1];
            qa_tbl[k] = one_minus_x_i * qa_tbl[2 * k] + x_i * qa_tbl[2 * k + 1];
            qb_tbl[k] = one_minus_x_i * qb_tbl[2 * k] + x_i * qb_tbl[2 * k + 1];
            qc_tbl[k] = one_minus_x_i * qc_tbl[2 * k] + x_i * qc_tbl[2 * k + 1];
        }
        current = half;
    }

    (SumcheckProof { round_polys }, challenges)
}

/// Step 3: degree-2 sumcheck of L(y)·w̃(y).
fn sumcheck_inner_bookkeeping<F: PrimeField>(
    mut l_tbl: Vec<F>,
    mut w_tbl: Vec<F>,
    ell: usize,
    claimed_sum: F,
    transcript: &mut Transcript,
) -> (SumcheckProof<F>, Vec<F>) {
    transcript.absorb_field(claimed_sum);

    let mut challenges = Vec::with_capacity(ell);
    let mut round_polys = Vec::with_capacity(ell);
    let two = F::from(2u64);
    let mut current = 1usize << ell;

    for _ in 0..ell {
        let half = current / 2;
        let mut g_i = vec![F::zero(); 3];

        for k in 0..half {
            let l0 = l_tbl[2 * k];
            let l1 = l_tbl[2 * k + 1];
            let w0 = w_tbl[2 * k];
            let w1 = w_tbl[2 * k + 1];

            g_i[0] += l0 * w0;
            g_i[1] += l1 * w1;
            g_i[2] += (two * l1 - l0) * (two * w1 - w0);
        }

        for &v in &g_i {
            transcript.absorb_field(v);
        }
        let y_i = transcript.squeeze_field::<F>();
        challenges.push(y_i);
        round_polys.push(g_i);

        let one_minus_y_i = F::one() - y_i;
        for k in 0..half {
            l_tbl[k] = one_minus_y_i * l_tbl[2 * k] + y_i * l_tbl[2 * k + 1];
            w_tbl[k] = one_minus_y_i * w_tbl[2 * k] + y_i * w_tbl[2 * k + 1];
        }
        current = half;
    }

    (SumcheckProof { round_polys }, challenges)
}

/// Run Steps 2–3, returning the proof, x*, y* and the claim u = w̃(y*).
pub fn piop_prove<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    w: &[F],
    ell_row: usize,
    ell_col: usize,
    transcript: &mut Transcript,
) -> (PiopProof<F>, Vec<F>, Vec<F>, F) {
    assert!(
        !matrices.a.is_empty(),
        "piop_prove: circuit has no constraints"
    );

    // `w` is the full half-split assignment described in `snark.rs`.
    let mut w_pad = w.to_vec();
    w_pad.resize(1 << ell_col, F::zero());

    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    let eq_tbl = build_eq_table(&r);
    let qa_tbl = build_q_table(&matrices.a, &w_pad, ell_row);
    let qb_tbl = build_q_table(&matrices.b, &w_pad, ell_row);
    let qc_tbl = build_q_table(&matrices.c, &w_pad, ell_row);

    let (outer_sc, x_star) =
        sumcheck_outer_bookkeeping(eq_tbl, qa_tbl, qb_tbl, qc_tbl, ell_row, transcript);

    let q_a_claim = q_eval(&matrices.a, &w_pad, &x_star);
    let q_b_claim = q_eval(&matrices.b, &w_pad, &x_star);
    let q_c_claim = q_eval(&matrices.c, &w_pad, &x_star);

    transcript.absorb_field(q_a_claim);
    transcript.absorb_field(q_b_claim);
    transcript.absorb_field(q_c_claim);

    let varrho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = q_a_claim + varrho * q_b_claim + gamma * q_c_claim;

    let l_tbl = build_l_table(
        &matrices.a,
        &matrices.b,
        &matrices.c,
        &x_star,
        varrho,
        gamma,
        ell_col,
    );
    let w_tbl = w_pad.clone();

    let (inner_sc, y_star) =
        sumcheck_inner_bookkeeping(l_tbl, w_tbl, ell_col, q_abc_claim, transcript);

    let u = mle_of_vector(&w_pad, ell_col, &y_star);

    transcript.absorb_field(u);

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
        u,
        a_eval,
        b_eval,
        c_eval,
    };
    (proof, x_star, y_star, u)
}

/// Returns `Some((x*, y*, u))` — the outer/inner terminal points and claimed witness MLE value.
///
/// The matrix MLEs Ã,B̃,C̃ at (x*,y*) are taken as the *claimed* values
/// `proof.{a,b,c}_eval`; the caller (snark verifier) must certify those against
/// the preprocessing commitments via the matrix-evaluation proofs, and must still
/// check the Step 4 opening of `u`.
pub fn piop_verify<F: PrimeField>(
    ell_row: usize,
    ell_col: usize,
    proof: &PiopProof<F>,
    transcript: &mut Transcript,
) -> Option<(Vec<F>, Vec<F>, F)> {
    let r: Vec<F> = (0..ell_row).map(|_| transcript.squeeze_field()).collect();

    let (x_star, outer_final) =
        sumcheck_verify(&proof.outer_sc, ell_row, 3, F::zero(), transcript)?;

    let eq_val: F = (0..ell_row)
        .map(|j| x_star[j] * r[j] + (F::one() - x_star[j]) * (F::one() - r[j]))
        .product();
    let oracle_outer = eq_val * (proof.q_a_claim * proof.q_b_claim - proof.q_c_claim);
    if outer_final != oracle_outer {
        return None;
    }

    transcript.absorb_field(proof.q_a_claim);
    transcript.absorb_field(proof.q_b_claim);
    transcript.absorb_field(proof.q_c_claim);

    let varrho: F = transcript.squeeze_field();
    let gamma: F = transcript.squeeze_field();
    let q_abc_claim = proof.q_a_claim + varrho * proof.q_b_claim + gamma * proof.q_c_claim;

    let (y_star, inner_final) =
        sumcheck_verify(&proof.inner_sc, ell_col, 2, q_abc_claim, transcript)?;

    let u = proof.u;

    let expected_abc = (proof.a_eval + varrho * proof.b_eval + gamma * proof.c_eval) * u;
    if inner_final != expected_abc {
        return None;
    }

    transcript.absorb_field(u);

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
    fn test_build_q_table() {
        let m: Matrix<F> = vec![vec![(F::from(2u64), 0)], vec![(F::from(3u64), 1)]];
        let w = vec![F::from(5u64), F::from(7u64)];
        let table = build_q_table(&m, &w, 1);

        assert_eq!(table.len(), 2);
        assert_eq!(table[0], F::from(10u64));
        assert_eq!(table[1], F::from(21u64));
    }

    #[test]
    fn test_build_l_table_zero() {
        let empty: Matrix<F> = vec![vec![], vec![]];
        let x_star = vec![F::from(0u64)];
        let table = build_l_table(&empty, &empty, &empty, &x_star, F::one(), F::one(), 1);
        assert!(table.iter().all(|&v| v == F::zero()));
    }
}
