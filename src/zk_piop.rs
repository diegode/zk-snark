//! Native masked Spartan reduction. No witness or inner-mask evaluation is sent.
use crate::{
    piop::{build_abc_col_table, build_mw_table},
    r1cs::{ConstraintMatrices, build_eq_table, mle_of_matrix_at},
    sumcheck::{SumcheckProof, lagrange_eval},
    transcript::Transcript,
    zk_pcs::{LinearForm, absorb, fold, nonzero},
};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use rand::{CryptoRng, Rng};

pub(crate) const OUTER_DEG: usize = 5;
pub(crate) const INNER_DEG: usize = 2;

/// a0 + sum_i sum_{j=1}^degree c_ij X_i^j, including an independent constant.
pub(crate) struct Mask<F: PrimeField> {
    pub coeffs: Vec<F>,
    vars: usize,
    degree: usize,
}

impl<F: PrimeField> Mask<F> {
    pub fn sample<R: Rng + CryptoRng>(vars: usize, degree: usize, rng: &mut R) -> Self {
        Self {
            coeffs: (0..1 + vars * degree).map(|_| F::rand(rng)).collect(),
            vars,
            degree,
        }
    }
    pub fn eval(&self, point: &[F]) -> F {
        self.coeffs[0]
            + point
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let mut p = F::one();
                    self.coeffs[1 + i * self.degree..1 + (i + 1) * self.degree]
                        .iter()
                        .map(|&c| {
                            p *= x;
                            c * p
                        })
                        .sum::<F>()
                })
                .sum::<F>()
    }
    pub fn sum(&self) -> F {
        F::from(2u64).pow([self.vars as u64]) * self.coeffs[0]
            + F::from(2u64).pow([(self.vars - 1) as u64])
                * self.coeffs[1..].iter().copied().sum::<F>()
    }
    fn round(&self, prefix: &[F], x: F) -> F {
        let i = prefix.len();
        let remaining = self.vars - i - 1;
        let mut point = prefix.to_vec();
        point.push(x);
        let fixed = self.eval(&point);
        let future: F = self.coeffs[1 + (i + 1) * self.degree..]
            .iter()
            .copied()
            .sum();
        let count = F::from(2u64).pow([remaining as u64]);
        count * fixed + count * F::from(2u64).inverse().unwrap() * future
    }
}

pub(crate) struct Secrets<F: PrimeField> {
    pub eta: [F; 3],
    pub outer: Mask<F>,
    pub inner: Mask<F>,
}

impl<F: PrimeField> Secrets<F> {
    pub fn sample<R: Rng + CryptoRng>(rows: usize, cols: usize, rng: &mut R) -> Self {
        Self {
            eta: [F::rand(rng), F::rand(rng), F::rand(rng)],
            outer: Mask::sample(rows, OUTER_DEG, rng),
            inner: Mask::sample(cols, INNER_DEG, rng),
        }
    }
    pub fn pack(&self, witness: &[F]) -> Vec<F> {
        let mut v = witness.to_vec();
        v.extend(self.eta);
        v.extend(&self.outer.coeffs);
        v.extend(&self.inner.coeffs);
        v.resize(v.len().next_power_of_two(), F::zero());
        v
    }
}

#[derive(CanonicalSerialize)]
pub struct ZkPiopProof<F: PrimeField> {
    pub outer_sc: SumcheckProof<F>,
    /// Masked values Q_M(x*) + eta_M x_last(x_last-1).
    pub q_claims: [F; 3],
    pub inner_sc: SumcheckProof<F>,
    /// Matrix values are public, certified by the sparse-matrix proofs.
    pub matrix_evals: [F; 3],
    pub outer_sum: F,
    pub inner_sum: F,
    pub outer_eval: F,
}

pub(crate) struct Context<F: PrimeField> {
    pub x: Vec<F>,
    pub y: Vec<F>,
    rho: F,
    gamma: F,
    tau_in: F,
    terminal: F,
}

fn challenge<F: PrimeField>(last: bool, t: &mut Transcript) -> F {
    loop {
        let x = t.squeeze_field::<F>();
        if !last || (!x.is_zero() && x != F::one()) {
            return x;
        }
    }
}

fn eq<F: PrimeField>(x: &[F], y: &[F]) -> F {
    x.iter()
        .zip(y)
        .map(|(&a, &b)| a * b + (F::one() - a) * (F::one() - b))
        .product()
}

pub(crate) fn prove<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    assignment: &[F],
    rows: usize,
    cols: usize,
    secrets: &Secrets<F>,
    t: &mut Transcript,
) -> (ZkPiopProof<F>, Context<F>) {
    let outer_sum = secrets.outer.sum();
    let inner_sum = secrets.inner.sum();
    t.absorb_field(outer_sum);
    t.absorb_field(inner_sum);
    let tau_out = nonzero::<F>(t);
    let tau_in = nonzero::<F>(t);
    let r: Vec<_> = (0..rows).map(|_| t.squeeze_field::<F>()).collect();
    let mut e = build_eq_table(&r);
    let mut abc: Vec<_> = [&matrices.a, &matrices.b, &matrices.c]
        .iter()
        .map(|m| build_mw_table(m, assignment, rows))
        .collect();
    let mut outer = Vec::new();
    let mut x = Vec::new();
    t.absorb_field(tau_out * outer_sum);
    for i in 0..rows {
        let p: Vec<_> = (0..=OUTER_DEG)
            .map(|j| {
                let z = F::from(j as u64);
                let ef = fold(&e, z);
                let af: Vec<_> = abc
                    .iter()
                    .enumerate()
                    .map(|(m, a)| {
                        let mut a = fold(a, z);
                        if i + 1 == rows {
                            a[0] += secrets.eta[m] * z * (z - F::one());
                        }
                        a
                    })
                    .collect();
                ef.iter()
                    .enumerate()
                    .map(|(k, &e)| e * (af[0][k] * af[1][k] - af[2][k]))
                    .sum::<F>()
                    + tau_out * secrets.outer.round(&x, z)
            })
            .collect();
        absorb(t, &p);
        let a = challenge(i + 1 == rows, t);
        x.push(a);
        outer.push(p);
        e = fold(&e, a);
        abc = abc.iter().map(|v| fold(v, a)).collect();
    }
    let phi = x[rows - 1] * (x[rows - 1] - F::one());
    let q_claims = std::array::from_fn(|i| abc[i][0] + secrets.eta[i] * phi);
    let outer_eval = secrets.outer.eval(&x);
    absorb(t, &q_claims);
    t.absorb_field(outer_eval);
    let rho = t.squeeze_field::<F>();
    let gamma = t.squeeze_field::<F>();
    let delta = phi * (secrets.eta[0] + rho * secrets.eta[1] + gamma * secrets.eta[2]);
    let mut l = build_abc_col_table(&matrices.a, &matrices.b, &matrices.c, &x, rho, gamma, cols);
    let mut w = assignment.to_vec();
    let mut inner = Vec::new();
    let mut y = Vec::new();
    t.absorb_field(q_claims[0] + rho * q_claims[1] + gamma * q_claims[2] + tau_in * inner_sum);
    let inv2 = F::from(2u64).inverse().unwrap();
    let mut delta_round = delta;
    for _ in 0..cols {
        delta_round *= inv2;
        let p: Vec<_> = (0..=INNER_DEG)
            .map(|j| {
                let z = F::from(j as u64);
                fold(&l, z)
                    .iter()
                    .zip(fold(&w, z))
                    .map(|(&a, b)| a * b)
                    .sum::<F>()
                    + delta_round
                    + tau_in * secrets.inner.round(&y, z)
            })
            .collect();
        absorb(t, &p);
        let a = t.squeeze_field::<F>();
        y.push(a);
        inner.push(p);
        l = fold(&l, a);
        w = fold(&w, a);
    }
    let terminal = lagrange_eval(inner.last().unwrap(), *y.last().unwrap());
    let matrix_evals = [&matrices.a, &matrices.b, &matrices.c].map(|m| mle_of_matrix_at(m, &x, &y));
    absorb(t, &matrix_evals);
    (
        ZkPiopProof {
            outer_sc: SumcheckProof { round_polys: outer },
            q_claims,
            inner_sc: SumcheckProof { round_polys: inner },
            matrix_evals,
            outer_sum,
            inner_sum,
            outer_eval,
        },
        Context {
            x,
            y,
            rho,
            gamma,
            tau_in,
            terminal,
        },
    )
}

pub(crate) fn verify<F: PrimeField>(
    rows: usize,
    cols: usize,
    p: &ZkPiopProof<F>,
    t: &mut Transcript,
) -> Option<Context<F>> {
    if rows == 0
        || cols == 0
        || p.outer_sc.round_polys.len() != rows
        || p.inner_sc.round_polys.len() != cols
    {
        return None;
    }
    t.absorb_field(p.outer_sum);
    t.absorb_field(p.inner_sum);
    let tau_out = nonzero::<F>(t);
    let tau_in = nonzero::<F>(t);
    let r: Vec<_> = (0..rows).map(|_| t.squeeze_field::<F>()).collect();
    let mut expected = tau_out * p.outer_sum;
    t.absorb_field(expected);
    let mut x = Vec::new();
    for (i, poly) in p.outer_sc.round_polys.iter().enumerate() {
        if poly.len() != OUTER_DEG + 1 || poly[0] + poly[1] != expected {
            return None;
        }
        absorb(t, poly);
        let a = challenge(i + 1 == rows, t);
        x.push(a);
        expected = lagrange_eval(poly, a);
    }
    if expected
        != eq(&r, &x) * (p.q_claims[0] * p.q_claims[1] - p.q_claims[2]) + tau_out * p.outer_eval
    {
        return None;
    }
    absorb(t, &p.q_claims);
    t.absorb_field(p.outer_eval);
    let rho = t.squeeze_field::<F>();
    let gamma = t.squeeze_field::<F>();
    expected = p.q_claims[0] + rho * p.q_claims[1] + gamma * p.q_claims[2] + tau_in * p.inner_sum;
    t.absorb_field(expected);
    let mut y = Vec::new();
    for poly in &p.inner_sc.round_polys {
        if poly.len() != INNER_DEG + 1 || poly[0] + poly[1] != expected {
            return None;
        }
        absorb(t, poly);
        let a = t.squeeze_field::<F>();
        y.push(a);
        expected = lagrange_eval(poly, a);
    }
    absorb(t, &p.matrix_evals);
    Some(Context {
        x,
        y,
        rho,
        gamma,
        tau_in,
        terminal: expected,
    })
}

/// Batch all four linear constraints after their claimed values are fixed.
/// The last is L(y*) W(y*) + Delta/2^cols + tau_in Z_in(y*).
pub(crate) fn relation<F: PrimeField>(
    p: &ZkPiopProof<F>,
    c: &Context<F>,
    public: &[F],
    t: &mut Transcript,
) -> (LinearForm<F>, F) {
    let rows = c.x.len();
    let cols = c.y.len();
    let half = 1usize << (cols - 1);
    let outer_start = half + 3;
    let inner_start = outer_start + 1 + OUTER_DEG * rows;
    let len = (inner_start + 1 + INNER_DEG * cols).next_power_of_two();
    let mut form = LinearForm::new(len.ilog2() as usize);
    let beta = nonzero::<F>(t);
    let b2 = beta.square();
    let b3 = b2 * beta;
    let count_out = F::from(2u64).pow([rows as u64]);
    let count_in = F::from(2u64).pow([cols as u64]);
    let inv2 = F::from(2u64).inverse().unwrap();
    // S_out + beta Z_out(x*) + beta^2 S_in + beta^3 terminal_private.
    form.unit(outer_start, count_out + beta);
    for (i, &x) in c.x.iter().enumerate() {
        let mut xp = F::one();
        for j in 0..OUTER_DEG {
            xp *= x;
            form.unit(
                outer_start + 1 + i * OUTER_DEG + j,
                count_out * inv2 + beta * xp,
            );
        }
    }
    form.unit(inner_start, b2 * count_in + b3 * c.tau_in);
    for (i, &y) in c.y.iter().enumerate() {
        let mut yp = F::one();
        for j in 0..INNER_DEG {
            yp *= y;
            form.unit(
                inner_start + 1 + i * INNER_DEG + j,
                b2 * count_in * inv2 + b3 * c.tau_in * yp,
            );
        }
    }
    let phi = c.x[rows - 1] * (c.x[rows - 1] - F::one());
    let scale = b3 * phi * count_in.inverse().unwrap();
    for (i, a) in [F::one(), c.rho, c.gamma].into_iter().enumerate() {
        form.unit(half + i, scale * a);
    }
    let l = p.matrix_evals[0] + c.rho * p.matrix_evals[1] + c.gamma * p.matrix_evals[2];
    let yt = c.y[cols - 1];
    let mut factors: Vec<_> = c.y[..cols - 1].iter().map(|&y| [F::one() - y, y]).collect();
    factors.resize(form.vars, [F::one(), F::zero()]);
    form.terms.push((b3 * l * (F::one() - yt), factors));
    let io: F = public
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            v * c.y[..cols - 1]
                .iter()
                .enumerate()
                .map(|(j, &y)| if (i >> j) & 1 == 1 { y } else { F::one() - y })
                .product::<F>()
        })
        .sum();
    let target =
        p.outer_sum + beta * p.outer_eval + b2 * p.inner_sum + b3 * (c.terminal - l * yt * io);
    (form, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{Field, One, UniformRand, Zero};
    use rand::{SeedableRng, rngs::StdRng};

    fn solve(mut matrix: Vec<Vec<F>>, mut rhs: Vec<F>) -> Option<Vec<F>> {
        let cols = matrix[0].len();
        let mut pivot = 0;
        let mut positions = Vec::new();
        for col in 0..cols {
            let Some(row) = (pivot..matrix.len()).find(|&i| !matrix[i][col].is_zero()) else {
                continue;
            };
            matrix.swap(pivot, row);
            rhs.swap(pivot, row);
            let inv = matrix[pivot][col].inverse().unwrap();
            for v in &mut matrix[pivot] {
                *v *= inv;
            }
            rhs[pivot] *= inv;
            let p = matrix[pivot].clone();
            let r = rhs[pivot];
            for i in 0..matrix.len() {
                if i != pivot {
                    let scale = matrix[i][col];
                    for j in col..cols {
                        matrix[i][j] -= scale * p[j];
                    }
                    rhs[i] -= scale * r;
                }
            }
            positions.push(col);
            pivot += 1;
        }
        if rhs[pivot..].iter().any(|r| !r.is_zero()) {
            return None;
        }
        assert_eq!(
            pivot, cols,
            "mask transcript map must have full column rank"
        );
        let mut answer = vec![F::zero(); cols];
        for (i, &col) in positions.iter().enumerate() {
            answer[col] = rhs[i];
        }
        Some(answer)
    }

    fn mask_view(mask: &Mask<F>, x: &[F]) -> Vec<F> {
        let mut out = vec![mask.sum()];
        for i in 0..x.len() {
            for j in 0..=mask.degree {
                out.push(mask.round(&x[..i], F::from(j as u64)));
            }
        }
        out.push(mask.eval(x));
        out
    }

    fn outer_view(mut abc: [Vec<F>; 3], eta: [F; 3], r: &[F], x: &[F]) -> Vec<F> {
        let mut e = build_eq_table(r);
        let mut out = vec![F::zero()];
        for i in 0..x.len() {
            for j in 0..=OUTER_DEG {
                let z = F::from(j as u64);
                let mut f = abc.each_ref().map(|a| fold(a, z));
                if i + 1 == x.len() {
                    for k in 0..3 {
                        f[k][0] += eta[k] * z * (z - F::one());
                    }
                }
                out.push(
                    fold(&e, z)
                        .iter()
                        .enumerate()
                        .map(|(k, &e)| e * (f[0][k] * f[1][k] - f[2][k]))
                        .sum(),
                );
            }
            abc = abc.each_ref().map(|v| fold(v, x[i]));
            e = fold(&e, x[i]);
        }
        out.push(F::zero());
        out
    }

    #[test]
    fn independent_constant_makes_the_full_mask_transcript_hide_witness_changes() {
        // For fixed interactive challenges, exhibit the exact bijection between
        // randomness for two different satisfying witnesses, keeping the sums,
        // masked Q tuple, ALL round polynomials and terminal mask value fixed.
        let mut rng = StdRng::seed_from_u64(992);
        for rows in 1..=4 {
            let r: Vec<_> = (0..rows).map(|_| F::rand(&mut rng)).collect();
            let x: Vec<_> = (0..rows).map(|_| F::rand(&mut rng)).collect();
            let phi = x[rows - 1] * (x[rows - 1] - F::one());
            let make = |rng: &mut StdRng| {
                let a: Vec<_> = (0..1 << rows).map(|_| F::rand(rng)).collect();
                let b: Vec<_> = (0..1 << rows).map(|_| F::rand(rng)).collect();
                let c = a.iter().zip(&b).map(|(&a, &b)| a * b).collect();
                [a, b, c]
            };
            let first = make(&mut rng);
            let second = make(&mut rng);
            let eta = [F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng)];
            let eq = build_eq_table(&x);
            let adjusted = std::array::from_fn(|i| {
                eta[i]
                    + (crate::zk_pcs::dot(&first[i], &eq) - crate::zk_pcs::dot(&second[i], &eq))
                        / phi
            });
            let a = outer_view(first, eta, &r, &x);
            let b = outer_view(second, adjusted, &r, &x);
            let rhs: Vec<_> = a.iter().zip(b).map(|(&a, b)| a - b).collect();
            let coeffs = 1 + OUTER_DEG * rows;
            let columns: Vec<_> = (0..coeffs)
                .map(|i| {
                    let mut c = vec![F::zero(); coeffs];
                    c[i] = F::one();
                    mask_view(
                        &Mask {
                            coeffs: c,
                            vars: rows,
                            degree: OUTER_DEG,
                        },
                        &x,
                    )
                })
                .collect();
            let matrix = (0..rhs.len())
                .map(|i| columns.iter().map(|c| c[i]).collect())
                .collect();
            let shift =
                solve(matrix, rhs.clone()).expect("a witness change must lie in the mask image");
            assert_eq!(
                mask_view(
                    &Mask {
                        coeffs: shift,
                        vars: rows,
                        degree: OUTER_DEG
                    },
                    &x
                ),
                rhs
            );
        }
    }

    #[test]
    fn mask_rounds_and_sums_match_boolean_enumeration() {
        let mut rng = StdRng::seed_from_u64(71);
        for degree in [INNER_DEG, OUTER_DEG] {
            let mask = Mask::<F>::sample(3, degree, &mut rng);
            let values: Vec<_> = (0..8)
                .map(|i| {
                    mask.eval(
                        &(0..3)
                            .map(|j| F::from(((i >> j) & 1) as u64))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            assert_eq!(mask.sum(), values.iter().copied().sum::<F>());
            let x = F::from(7);
            let y = F::from(9);
            assert_eq!(
                mask.round(&[x], y),
                mask.eval(&[x, y, F::zero()]) + mask.eval(&[x, y, F::one()])
            );
        }
    }
}
