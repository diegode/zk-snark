//! Spartan-style computation commitment using a direct sparse sumcheck instead
//! of Spark.
//!
//! For each nonzero `(i_k, j_k, v_k)`, preprocessing commits tables for `v_k`
//! and every row/column index bit. To prove `M̃(x*,y*) = v`, sumcheck reduces
//! `Σ_k v_k·eq(i_k,x*)·eq(j_k,y*)` to a point `k*`, where all tables are opened.
//! The `O(log N)` bit tables account for the extra logarithmic factor described
//! in the README.
use ark_ff::{FftField, PrimeField};
use ark_serialize::CanonicalSerialize;
use rayon::prelude::*;

use ark_relations::gr1cs::Matrix;

use crate::{
    pcs::{
        build_eval_public, commit_public, finalize_eval, verify_eval, PcsError,
        Commitment, Proof, Witness,
    },
    piop::ell_for,
    sumcheck::{sumcheck_verify, SumcheckProof},
    transcript::Transcript,
};

/// Public commitments to one matrix encoding (lives in the verifying key).
#[derive(Clone)]
pub struct MatrixCommitments {
    pub val: Commitment,
    pub rbits: Vec<Commitment>,
    pub cbits: Vec<Commitment>,
    /// Number of boolean variables of the sparse encoding: `s = ell_for(nnz)`.
    pub s: usize,
    pub ell_row: usize,
    pub ell_col: usize,
}

impl MatrixCommitments {
    /// Bind all encoding roots to the transcript (val, then row bits, then column
    /// bits). Prover and verifier must call this in the same order.
    pub fn absorb_into(&self, transcript: &mut Transcript) {
        transcript.absorb(&self.val.root);
        for c in &self.rbits {
            transcript.absorb(&c.root);
        }
        for c in &self.cbits {
            transcript.absorb(&c.root);
        }
    }
}

/// Prover-side encoding: the committed tables together with their PCS witnesses
/// and public commitments (lives in the proving key).
pub struct MatrixEncoding<F: PrimeField> {
    pub commitments: MatrixCommitments,
    pub val_wit: Witness<F>,
    pub rbit_wits: Vec<Witness<F>>,
    pub cbit_wits: Vec<Witness<F>>,
}

#[derive(CanonicalSerialize)]
pub struct MatrixEvalProof<F: PrimeField> {
    /// Degree-`d` sumcheck over `{0,1}^s` reducing `M̃(x*,y*)` to a point `k*`.
    pub sc: SumcheckProof<F>,
    /// PCS opening of `val` at `k*`.
    pub val_proof: Proof<F>,
    /// PCS openings of each `rbit_b` at `k*` (length ell_row).
    pub rbit_proofs: Vec<Proof<F>>,
    /// PCS openings of each `cbit_b` at `k*` (length ell_col).
    pub cbit_proofs: Vec<Proof<F>>,
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

    let mut all_tables: Vec<Vec<F>> = Vec::with_capacity(1 + ell_row + ell_col);
    all_tables.push(val);
    all_tables.extend(rbits);
    all_tables.extend(cbits);

    let mut committed: Vec<(Commitment, Witness<F>)> = all_tables
        .into_par_iter()
        .map(commit_public)
        .collect();

    let mut drain = committed.drain(..);
    let (val_comm, val_wit) = drain.next().unwrap();
    let mut rbit_comms = Vec::with_capacity(ell_row);
    let mut rbit_wits = Vec::with_capacity(ell_row);
    for _ in 0..ell_row {
        let (c, w) = drain.next().unwrap();
        rbit_comms.push(c);
        rbit_wits.push(w);
    }
    let mut cbit_comms = Vec::with_capacity(ell_col);
    let mut cbit_wits = Vec::with_capacity(ell_col);
    for _ in 0..ell_col {
        let (c, w) = drain.next().unwrap();
        cbit_comms.push(c);
        cbit_wits.push(w);
    }
    drop(drain);

    MatrixEncoding {
        commitments: MatrixCommitments {
            val: val_comm,
            rbits: rbit_comms,
            cbits: cbit_comms,
            s,
            ell_row,
            ell_col,
        },
        val_wit,
        rbit_wits,
        cbit_wits,
    }
}

/// `eq(b, z) = b·z + (1−b)(1−z)`.
#[inline]
fn eq_factor<F: PrimeField>(b: F, z: F) -> F {
    b * z + (F::one() - b) * (F::one() - z)
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

    let mut val = enc.val_wit.evals.clone();
    let mut rbits: Vec<Vec<F>> = enc.rbit_wits.iter().map(|w| w.evals.clone()).collect();
    let mut cbits: Vec<Vec<F>> = enc.cbit_wits.iter().map(|w| w.evals.clone()).collect();

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

    transcript.absorb_field(v);

    let mut challenges = Vec::with_capacity(s);
    let mut round_polys = Vec::with_capacity(s);
    let mut current = 1usize << s;

    for _ in 0..s {
        let half = current / 2;
        let mut s_j = vec![F::zero(); d + 1];

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
                s_j[t] += prod;
            }
        }

        for &e in &s_j {
            transcript.absorb_field(e);
        }
        let r = transcript.squeeze_field::<F>();
        challenges.push(r);
        round_polys.push(s_j);

        let omr = F::one() - r;
        for k in 0..half {
            val[k] = omr * val[2 * k] + r * val[2 * k + 1];
            for b in 0..ell_row {
                rbits[b][k] = omr * rbits[b][2 * k] + r * rbits[b][2 * k + 1];
            }
            for b in 0..ell_col {
                cbits[b][k] = omr * cbits[b][2 * k] + r * cbits[b][2 * k + 1];
            }
        }
        current = half;
    }

    // Open every sparse-encoding table at the sumcheck's terminal point.
    let witnesses: Vec<&Witness<F>> = std::iter::once(&enc.val_wit)
        .chain(enc.rbit_wits.iter())
        .chain(enc.cbit_wits.iter())
        .collect();

    let datas: Vec<_> = witnesses
        .par_iter()
        .map(|w| build_eval_public(w, &challenges))
        .collect();

    let proofs: Vec<Proof<F>> = witnesses
        .iter()
        .zip(datas)
        .map(|(w, data)| finalize_eval(w, challenges.len(), data, transcript))
        .collect();

    let mut it = proofs.into_iter();
    let val_proof = it.next().unwrap();
    let rbit_proofs: Vec<Proof<F>> = (0..ell_row).map(|_| it.next().unwrap()).collect();
    let cbit_proofs: Vec<Proof<F>> = (0..ell_col).map(|_| it.next().unwrap()).collect();

    Ok((
        v,
        MatrixEvalProof {
            sc: SumcheckProof { round_polys },
            val_proof,
            rbit_proofs,
            cbit_proofs,
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

    if x_star.len() != ell_row
        || y_star.len() != ell_col
        || proof.rbit_proofs.len() != ell_row
        || proof.cbit_proofs.len() != ell_col
        || comm.rbits.len() != ell_row
        || comm.cbits.len() != ell_col
    {
        return Ok(false);
    }

    let Some((k_star, final_eval)) = sumcheck_verify(&proof.sc, s, d, claimed_v, transcript) else {
        return Ok(false);
    };

    let val_v = proof.val_proof.final_value;
    if !verify_eval(&comm.val, &k_star, val_v, &proof.val_proof, transcript)? {
        return Ok(false);
    }
    let mut rbit_v = Vec::with_capacity(ell_row);
    for (c, p) in comm.rbits.iter().zip(proof.rbit_proofs.iter()) {
        let v = p.final_value;
        if !verify_eval(c, &k_star, v, p, transcript)? {
            return Ok(false);
        }
        rbit_v.push(v);
    }
    let mut cbit_v = Vec::with_capacity(ell_col);
    for (c, p) in comm.cbits.iter().zip(proof.cbit_proofs.iter()) {
        let v = p.final_value;
        if !verify_eval(c, &k_star, v, p, transcript)? {
            return Ok(false);
        }
        cbit_v.push(v);
    }

    let mut g = val_v;
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
