/// MLE helper functions for the R1CS-based PIOP.
///
/// The R1CS instance is represented as a pair:
///   (ConstraintMatrices<F>, witness: Vec<F>)
/// produced by `cs.to_matrices()` from an `ark-relations` `ConstraintSystem`.
///
/// Variable-index convention (from ark-relations):
///   0             = constant "1"
///   1..num_public = public instance variables
///   rest          = private witness variables
/// The flat witness vector w mirrors this order: w[i] = value of variable i.
use ark_ff::PrimeField;
use ark_relations::r1cs::Matrix;

use crate::sumcheck::eq_eval;

// ---------------------------------------------------------------------------
// eq-table builder (used here and by piop)
// ---------------------------------------------------------------------------

/// Evaluation table of eq(·, r): table[k] = ∏_j (k_j·r_j + (1−k_j)·(1−r_j)).
pub(crate) fn build_eq_table<F: PrimeField>(r: &[F]) -> Vec<F> {
    let ell = r.len();
    let mut table = vec![F::zero(); 1 << ell];
    table[0] = F::one();
    for (i, &ri) in r.iter().enumerate() {
        for j in 0..(1 << i) {
            let tmp = table[j];
            table[j] = tmp * (F::one() - ri);
            table[j + (1 << i)] = tmp * ri;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// MLE of a flat vector
// ---------------------------------------------------------------------------

/// Evaluate the multilinear extension of `v` at point `z ∈ F^ell`.
///
///   ṽ(z) = ∑_{x ∈ {0,1}^ell} v[x] · eq(x, z)
pub fn mle_of_vector<F: PrimeField>(v: &[F], ell: usize, z: &[F]) -> F {
    assert_eq!(v.len(), 1 << ell);
    assert_eq!(z.len(), ell);
    (0..v.len()).map(|x| v[x] * eq_eval(x, ell, z)).sum()
}

// ---------------------------------------------------------------------------
// Q_M(x*) — prover-side evaluation
// ---------------------------------------------------------------------------

/// Compute Q_M(x*) = ∑_{y ∈ {0,1}^ell} M̃(x*, y) · w[y]
///
/// Efficient formula (O(nnz)):
///   Q_M(x*) = ∑_i eq(i, ell, x*) · (∑_{(c,j) ∈ M[i]} c · w[j])
pub fn q_eval<F: PrimeField>(m: &Matrix<F>, w: &[F], x_star: &[F]) -> F {
    let eq_tbl = build_eq_table(x_star);
    m.iter().enumerate().map(|(i, row)| {
        let row_dot: F = row.iter().map(|(c, j)| *c * w[*j]).sum();
        eq_tbl[i] * row_dot
    }).sum()
}

// ---------------------------------------------------------------------------
// M̃(x, y) — verifier-side evaluation
// ---------------------------------------------------------------------------

/// Evaluate M̃(x, y) at arbitrary continuous (x, y).
///
///   M̃(x, y) = ∑_i eq(i, ell_row, x) · ∑_{(c,j)∈M[i]} c · eq(j, ell_col, y)
pub fn mle_of_matrix_at<F: PrimeField>(
    m: &Matrix<F>,
    x: &[F],
    y: &[F],
) -> F {
    let eq_x = build_eq_table(x);
    let eq_y = build_eq_table(y);

    m.iter().enumerate().map(|(i, row)| {
        let col_sum: F = row.iter().map(|(c, j)| *c * eq_y[*j]).sum();
        eq_x[i] * col_sum
    }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::Field;

    #[test]
    fn test_mle_of_vector() {
        let ell = 2;
        let v = vec![F::from(10u64), F::from(20u64), F::from(30u64), F::from(40u64)];

        // Boolean hypercube vertices: should evaluate exactly to the vector elements
        let z_00 = vec![F::from(0u64), F::from(0u64)];
        assert_eq!(mle_of_vector(&v, ell, &z_00), F::from(10u64));

        let z_10 = vec![F::from(1u64), F::from(0u64)];
        assert_eq!(mle_of_vector(&v, ell, &z_10), F::from(20u64));

        let z_01 = vec![F::from(0u64), F::from(1u64)];
        assert_eq!(mle_of_vector(&v, ell, &z_01), F::from(30u64));

        let z_11 = vec![F::from(1u64), F::from(1u64)];
        assert_eq!(mle_of_vector(&v, ell, &z_11), F::from(40u64));

        // Non-boolean evaluation point: z = (2, 3)
        // MLE(z_0, z_1) = 10(1-z_0)(1-z_1) + 20(z_0)(1-z_1) + 30(1-z_0)(z_1) + 40(z_0)(z_1)
        // For z = (2, 3):
        // 10 * (-1) * (-2) + 20 * 2 * (-2) + 30 * (-1) * 3 + 40 * 2 * 3
        // 20 - 80 - 90 + 240 = 90
        let z_23 = vec![F::from(2u64), F::from(3u64)];
        assert_eq!(mle_of_vector(&v, ell, &z_23), F::from(90u64));
    }

    // Matrix: A[0,0] = 2, A[1,1] = 3  (2 rows, 2 cols, 1-indexed for ell_row=ell_col=1)
    fn test_matrix() -> ark_relations::r1cs::Matrix<F> {
        vec![
            vec![(F::from(2u64), 0)],
            vec![(F::from(3u64), 1)],
        ]
    }

    #[test]
    fn test_q_eval() {
        let a = test_matrix();
        let w = vec![F::from(5u64), F::from(7u64)];

        // Q_A(x*) = ∑_i eq(i, ell, x*) · (∑_j A[i,j] · w[j])
        // At x* = (0,): eq(0,(0,))=1, eq(1,(0,))=0  →  1·(2·5) + 0·(3·7) = 10
        assert_eq!(q_eval(&a, &w, &[F::from(0u64)]), F::from(10u64));
        // At x* = (1,): eq(0,(1,))=0, eq(1,(1,))=1  →  0 + 1·(3·7) = 21
        assert_eq!(q_eval(&a, &w, &[F::from(1u64)]), F::from(21u64));
    }

    #[test]
    fn test_mle_of_matrix_at() {
        let a = test_matrix();
        let zero = vec![F::from(0u64)];
        let one  = vec![F::from(1u64)];

        // At boolean corners M̃ should equal the matrix entries.
        assert_eq!(mle_of_matrix_at(&a, &zero, &zero), F::from(2u64));
        assert_eq!(mle_of_matrix_at(&a, &zero, &one),  F::from(0u64));
        assert_eq!(mle_of_matrix_at(&a, &one,  &zero), F::from(0u64));
        assert_eq!(mle_of_matrix_at(&a, &one,  &one),  F::from(3u64));

        // Linearity check at a non-boolean point x=(1/2), y=(1/2).
        // M̃(1/2, 1/2) = eq(0,(1/2))·[2·eq(0,(1/2))] + eq(1,(1/2))·[3·eq(1,(1/2))]
        //              = (1/2)·2·(1/2) + (1/2)·3·(1/2) = 1/2 + 3/4 = 5/4
        let inv2 = F::from(2u64).inverse().unwrap();
        let expected = F::from(2u64) * inv2 * inv2 + F::from(3u64) * inv2 * inv2;
        assert_eq!(mle_of_matrix_at(&a, &[inv2], &[inv2]), expected);
    }

    #[test]
    fn test_build_eq_table() {
        let r = vec![F::from(2u64), F::from(3u64)];
        let ell = r.len();
        let table = build_eq_table(&r);

        assert_eq!(table.len(), 1 << ell);

        // Every entry must equal eq_eval(k, ell, r).
        for k in 0..(1 << ell) {
            assert_eq!(table[k], eq_eval(k, ell, &r), "eq_table[{k}] mismatch");
        }

        // Partition-of-unity: entries always sum to 1.
        let sum: F = table.iter().sum();
        assert_eq!(sum, F::from(1u64));
    }
}
