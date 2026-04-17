//! Spartan-style computation commitment using a direct sparse sumcheck instead
//! of Spark.
//! Chapter: the preprocessing and sparse-evaluation sumcheck of Construction 12, Steps 1 and 3.
//!
//! For each nonzero `(i_k, j_k, v_k)`, preprocessing commits tables for `v_k`
//! and every row/column index bit. To prove `M̃(x*,y*) = v`, sumcheck reduces
//! `Σ_k v_k·eq(i_k,x*)·eq(j_k,y*)` to a point `k*`, where all tables are opened.
//! The `O(log N)` bit tables account for the extra logarithmic factor described
//! in the README. They all fold with the sumcheck's challenges, so they share one
//! Merkle tree per fold layer, whose leaves are their columns (see `pcs`).
use ark_ff::{FftField, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use ark_relations::gr1cs::Matrix;

use crate::{
    pcs::{Commitment, FoldProver, PcsError, Proof, Witness, commit_public, verify_proximity},
    piop::ell_for,
    sumcheck::{SumcheckProof, sumcheck_verify_interleaved},
    transcript::Transcript,
};

/// Public commitment to one matrix encoding (lives in the verifying key): one tree
/// over the columns of its tables, in the order val, row bits, column bits.
#[derive(Clone)]
pub struct MatrixCommitments {
    pub tables: Commitment,
    /// Number of boolean variables of the sparse encoding: `s = ell_for(nnz)`.
    pub s: usize,
    pub ell_row: usize,
    pub ell_col: usize,
}

impl MatrixCommitments {
    /// Bind the encoding's shape and root to the transcript. Prover and verifier
    /// must call this in the same order.
    pub fn absorb_into(&self, transcript: &mut Transcript) {
        for dimension in [self.s, self.ell_row, self.ell_col] {
            transcript.absorb(&(dimension as u64).to_le_bytes());
        }
        transcript.absorb(&self.tables.root);
    }

    /// Tables per leaf: `val`, then `ell_row` row bits and `ell_col` column bits.
    fn width(&self) -> usize {
        1 + self.ell_row + self.ell_col
    }
}

/// Prover-side encoding: the committed tables together with their PCS witness
/// and public commitment (lives in the proving key).
pub struct MatrixEncoding<F: PrimeField> {
    pub commitments: MatrixCommitments,
    pub witness: Witness<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MatrixEvalProof<F: PrimeField> {
    /// Degree-`d` sumcheck over `{0,1}^s` reducing `M̃(x*,y*)` to a point `k*`.
    pub sc: SumcheckProof<F>,
    /// PCS openings of all tables at `k*`; `final_values` lists val, then the row
    /// bits, then the column bits.
    pub folds: Proof<F>,
}

/// Build and commit the sparse encoding of `m`.  Deterministic and public.
///
/// `m` is a row-indexed sparse matrix; `ell_row`/`ell_col` are the row/column
/// bit-widths shared by the whole R1CS instance.
pub fn encode_matrix<F: PrimeField + FftField>(
    m: &Matrix<F>,
    ell_row: usize,
    ell_col: usize,
) -> MatrixEncoding<F> {
    let mut nz: Vec<(usize, usize, F)> = Vec::new();
    for (i, row) in m.iter().enumerate() {
        for (c, j) in row {
            nz.push((i, *j, *c));
        }
    }

    let s = ell_for(nz.len().max(1));
    let size = 1usize << s;

    let mut val = vec![F::zero(); size];
    let mut rbits: Vec<Vec<F>> = vec![vec![F::zero(); size]; ell_row];
    let mut cbits: Vec<Vec<F>> = vec![vec![F::zero(); size]; ell_col];

    for (k, &(i, j, c)) in nz.iter().enumerate() {
        val[k] = c;
        for b in 0..ell_row {
            rbits[b][k] = F::from(((i >> b) & 1) as u64);
        }
        for b in 0..ell_col {
            cbits[b][k] = F::from(((j >> b) & 1) as u64);
        }
    }

    let mut tables: Vec<Vec<F>> = Vec::with_capacity(1 + ell_row + ell_col);
    tables.push(val);
    tables.extend(rbits);
    tables.extend(cbits);
    let (commitment, witness) = commit_public(tables);

    MatrixEncoding {
        commitments: MatrixCommitments {
            tables: commitment,
            s,
            ell_row,
            ell_col,
        },
        witness,
    }
}

/// `eq(b, z) = b·z + (1−b)(1−z)`.
#[inline]
fn eq_factor<F: PrimeField>(b: F, z: F) -> F {
    b * z + (F::one() - b) * (F::one() - z)
}

fn bind_statement<F: PrimeField>(
    comm: &MatrixCommitments,
    x: &[F],
    y: &[F],
    transcript: &mut Transcript,
) {
    transcript.absorb(b"matrix-eval/interleaved-v2");
    comm.absorb_into(transcript);
    for &v in x.iter().chain(y) {
        transcript.absorb_field(v);
    }
}

/// Prove `M̃(x*, y*) = v`, returning the value `v` and the evaluation proof.
pub fn prove_matrix_eval<F: PrimeField + FftField>(
    enc: &MatrixEncoding<F>,
    x_star: &[F],
    y_star: &[F],
    transcript: &mut Transcript,
) -> Result<(F, MatrixEvalProof<F>), PcsError> {
    let s = enc.commitments.s;
    let ell_row = enc.commitments.ell_row;
    let ell_col = enc.commitments.ell_col;
    let d = 1 + ell_row + ell_col;
    debug_assert_eq!(x_star.len(), ell_row);
    debug_assert_eq!(y_star.len(), ell_col);

    let tables = &enc.witness.tables;
    let mut val = tables[0].clone();
    let mut rbits: Vec<Vec<F>> = tables[1..1 + ell_row].to_vec();
    let mut cbits: Vec<Vec<F>> = tables[1 + ell_row..].to_vec();

    let v = {
        let mut acc = F::zero();
        for k in 0..(1usize << s) {
            let vk = val[k];
            if vk.is_zero() {
                continue;
            }
            let mut term = vk;
            for b in 0..ell_row {
                term *= eq_factor(rbits[b][k], x_star[b]);
            }
            for b in 0..ell_col {
                term *= eq_factor(cbits[b][k], y_star[b]);
            }
            acc += term;
        }
        acc
    };

    bind_statement(&enc.commitments, x_star, y_star, transcript);
    transcript.absorb_field(v);

    let mut folds = FoldProver::new(&enc.witness.codewords)?;

    let mut round_polys = Vec::with_capacity(s);
    let mut current = 1usize << s;

    for _ in 0..s {
        let half = current / 2;
        let mut g_i = vec![F::zero(); d + 1];

        for k in 0..half {
            for t in 0..=d {
                let tf = F::from(t as u64);
                let omt = F::one() - tf;
                let mut prod = omt * val[2 * k] + tf * val[2 * k + 1];
                for b in 0..ell_row {
                    let rb = omt * rbits[b][2 * k] + tf * rbits[b][2 * k + 1];
                    prod *= eq_factor(rb, x_star[b]);
                }
                for b in 0..ell_col {
                    let cb = omt * cbits[b][2 * k] + tf * cbits[b][2 * k + 1];
                    prod *= eq_factor(cb, y_star[b]);
                }
                g_i[t] += prod;
            }
        }

        for &e in &g_i {
            transcript.absorb_field(e);
        }
        // Each challenge k_i also folds every table (Construction 10, step 1).
        let k_i = transcript.squeeze_field::<F>();
        folds.advance_public(k_i);
        folds.absorb_latest(transcript);
        round_polys.push(g_i);

        let one_minus_k_i = F::one() - k_i;
        for k in 0..half {
            val[k] = one_minus_k_i * val[2 * k] + k_i * val[2 * k + 1];
            for b in 0..ell_row {
                rbits[b][k] = one_minus_k_i * rbits[b][2 * k] + k_i * rbits[b][2 * k + 1];
            }
            for b in 0..ell_col {
                cbits[b][k] = one_minus_k_i * cbits[b][2 * k] + k_i * cbits[b][2 * k + 1];
            }
        }
        current = half;
    }

    // Only query openings remain; the roots were sent during the sumcheck.
    Ok((
        v,
        MatrixEvalProof {
            sc: SumcheckProof { round_polys },
            folds: folds.finish(&enc.witness, None, transcript),
        },
    ))
}

/// Verify that the committed matrix evaluates to `claimed_v` at `(x*, y*)`.
pub fn verify_matrix_eval<F: PrimeField + FftField>(
    comm: &MatrixCommitments,
    x_star: &[F],
    y_star: &[F],
    claimed_v: F,
    proof: &MatrixEvalProof<F>,
    transcript: &mut Transcript,
) -> Result<bool, PcsError> {
    let s = comm.s;
    let ell_row = comm.ell_row;
    let ell_col = comm.ell_col;
    let d = 1 + ell_row + ell_col;

    if x_star.len() != ell_row || y_star.len() != ell_col {
        return Ok(false);
    }

    bind_statement(comm, x_star, y_star, transcript);
    let Some((k_star, final_eval)) =
        sumcheck_verify_interleaved(&proof.sc, s, d, claimed_v, transcript, |i, _, t| {
            proof.folds.absorb_round(i, s, t)
        })
    else {
        return Ok(false);
    };

    let width = comm.width();
    if !verify_proximity(
        &comm.tables.root,
        &k_star,
        width,
        false,
        &proof.folds,
        transcript,
    )? {
        return Ok(false);
    }
    let (val_v, bits) = proof.folds.final_values.split_first().unwrap();
    let (rbit_v, cbit_v) = bits.split_at(ell_row);

    let mut g = *val_v;
    for b in 0..ell_row {
        g *= eq_factor(rbit_v[b], x_star[b]);
    }
    for b in 0..ell_col {
        g *= eq_factor(cbit_v[b], y_star[b]);
    }

    Ok(g == final_eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::mle_of_matrix_at;
    use ark_bls12_381::Fr as F;
    use ark_ff::Zero;

    fn sample_matrix() -> Matrix<F> {
        vec![
            vec![(F::from(2u64), 0)],
            vec![(F::from(3u64), 2)],
            vec![(F::from(5u64), 1)],
            vec![(F::from(7u64), 3)],
        ]
    }

    fn roundtrip(m: &Matrix<F>, ell_row: usize, ell_col: usize, x: Vec<F>, y: Vec<F>) -> bool {
        let enc = encode_matrix(m, ell_row, ell_col);
        let comm = enc.commitments.clone();

        let mut pt = Transcript::new(b"mtest");
        let (v, proof) = prove_matrix_eval(&enc, &x, &y, &mut pt).unwrap();

        assert_eq!(v, mle_of_matrix_at(m, &x, &y), "claimed v mismatch with oracle");

        let mut vt = Transcript::new(b"mtest");
        verify_matrix_eval(&comm, &x, &y, v, &proof, &mut vt).unwrap()
    }

    #[test]
    fn boolean_corners_match_entries() {
        let m = sample_matrix();
        for (i, j, expected) in [(0usize, 0usize, 2u64), (1, 2, 3), (2, 1, 5), (3, 3, 7), (0, 1, 0)] {
            let x = vec![F::from((i & 1) as u64), F::from(((i >> 1) & 1) as u64)];
            let y = vec![F::from((j & 1) as u64), F::from(((j >> 1) & 1) as u64)];
            assert_eq!(mle_of_matrix_at(&m, &x, &y), F::from(expected));
            assert!(roundtrip(&m, 2, 2, x, y), "boolean corner ({i},{j}) failed");
        }
    }

    #[test]
    fn random_point_matches_oracle() {
        let m = sample_matrix();
        let x = vec![F::from(11u64), F::from(13u64)];
        let y = vec![F::from(17u64), F::from(19u64)];
        assert!(roundtrip(&m, 2, 2, x, y));
    }

    #[test]
    fn wrong_claimed_value_rejected() {
        let m = sample_matrix();
        let enc = encode_matrix(&m, 2, 2);
        let comm = enc.commitments.clone();
        let x = vec![F::from(11u64), F::from(13u64)];
        let y = vec![F::from(17u64), F::from(19u64)];

        let mut pt = Transcript::new(b"mtest");
        let (v, proof) = prove_matrix_eval(&enc, &x, &y, &mut pt).unwrap();

        let mut vt = Transcript::new(b"mtest");
        assert!(!verify_matrix_eval(&comm, &x, &y, v + F::from(1u64), &proof, &mut vt).unwrap());
    }

    #[test]
    fn single_nonzero_and_padding() {
        let m: Matrix<F> = vec![vec![(F::from(9u64), 1)], vec![]];
        let x = vec![F::from(4u64)];
        let y = vec![F::from(6u64)];
        assert!(roundtrip(&m, 1, 1, x, y));
    }

    #[test]
    fn all_zero_matrix() {
        let m: Matrix<F> = vec![vec![], vec![]];
        let x = vec![F::from(4u64)];
        let y = vec![F::from(6u64)];
        let enc = encode_matrix(&m, 1, 1);
        let comm = enc.commitments.clone();
        let mut pt = Transcript::new(b"mtest");
        let (v, proof) = prove_matrix_eval(&enc, &x, &y, &mut pt).unwrap();
        assert_eq!(v, F::zero());
        let mut vt = Transcript::new(b"mtest");
        assert!(verify_matrix_eval(&comm, &x, &y, v, &proof, &mut vt).unwrap());
    }
}
