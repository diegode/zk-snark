//! Native masked Spartan reduction. No witness or inner-mask evaluation is sent.
//! Chapter: Section 7.2, (2) Masking the sumchecks; Construction 13 (Masked sumcheck),
//! with Propositions 8 and 9.
use crate::{
    piop::{build_l_table, build_q_table},
    r1cs::{ConstraintMatrices, build_eq_table, mle_of_matrix_at, public_mle},
    sumcheck::{SumcheckProof, lagrange_eval},
    transcript::Transcript,
    zk_pcs::{LinearForm, absorb, fold, nonzero},
};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use rand::{CryptoRng, Rng};

/// Individual degree of the masked outer summand eq·(Q̄_A·Q̄_B − Q̄_C).
pub(crate) const OUTER_DEG: usize = 5;
pub(crate) const INNER_DEG: usize = 2;

/// Z(X) = c_0 + Σ_i Σ_{j=1}^{degree} c_{i,j}·X_i^j, the Z_out / Z_in of Section 7.2 (2).
/// `coeffs` = [c_0, c_{1,1}, …, c_{1,degree}, c_{2,1}, …].
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
    /// S = Σ_{x ∈ {0,1}^ℓ} Z(x) = 2^ℓ·c_0 + 2^{ℓ-1}·Σ c_{i,j} (proof of Proposition 8).
    pub fn sum(&self) -> F {
        F::from(2u64).pow([self.vars as u64]) * self.coeffs[0]
            + F::from(2u64).pow([(self.vars - 1) as u64])
                * self.coeffs[1..].iter().copied().sum::<F>()
    }
    /// The mask's contribution to the round polynomial: Σ over the trailing cube of Z(prefix, x, ·).
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

/// The prover's masking randomness: row masks η_M and the sumcheck masks Z_out, Z_in.
pub(crate) struct Secrets<F: PrimeField> {
    pub eta: [F; 3],
    pub z_out: Mask<F>,
    pub z_in: Mask<F>,
}

impl<F: PrimeField> Secrets<F> {
    pub fn sample<R: Rng + CryptoRng>(rows: usize, cols: usize, rng: &mut R) -> Self {
        Self {
            eta: [F::rand(rng), F::rand(rng), F::rand(rng)],
            z_out: Mask::sample(rows, OUTER_DEG, rng),
            z_in: Mask::sample(cols, INNER_DEG, rng),
        }
    }
    /// Construction 13, Setup: pack the private trace, η_M and mask coefficients into `a`.
    pub fn pack(&self, witness: &[F]) -> Vec<F> {
        let mut a = witness.to_vec();
        a.extend(self.eta);
        a.extend(&self.z_out.coeffs);
        a.extend(&self.z_in.coeffs);
        a.resize(a.len().next_power_of_two(), F::zero());
        a
    }
}

#[derive(CanonicalSerialize)]
pub struct ZkPiopProof<F: PrimeField> {
    /// Step 3: outer sumcheck on F_r + τ_out·Z_out.
    pub outer_sc: SumcheckProof<F>,
    /// Step 4: v_M = Q̄_M(x*) = Q_M(x*) + η_M·x*_ℓ(x*_ℓ − 1), for M ∈ {A, B, C}.
    pub v_m: [F; 3],
    /// Step 4: inner sumcheck on L·w̃ + 2^{-ℓ}Δ + τ_in·Z_in.
    pub inner_sc: SumcheckProof<F>,
    /// Matrix values are public, certified by the sparse-matrix proofs.
    pub matrix_evals: [F; 3],
    /// Step 1: declared mask sums S_out and S_in.
    pub s_out: F,
    pub s_in: F,
    /// Step 4: Z_out(x*), authenticated by the batched private opening.
    pub z_out_eval: F,
}

pub(crate) struct Context<F: PrimeField> {
    pub x_star: Vec<F>,
    pub y_star: Vec<F>,
    varrho: F,
    gamma: F,
    tau_in: F,
    /// g_ℓ(y*_ℓ): the inner sumcheck's terminal value.
    terminal: F,
}

/// The last outer challenge x*_ℓ is resampled outside {0, 1}, so that the row masks hide
/// the v_M; Proposition 10's simulator would instead abort on x*_ℓ ∈ {0, 1}.
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

/// Construction 13, Steps 1–4, up to the batched private opening (`relation`).
/// The private commitment to `a` has already been absorbed by the caller.
pub(crate) fn prove<F: PrimeField>(
    matrices: &ConstraintMatrices<F>,
    assignment: &[F],
    rows: usize,
    cols: usize,
    secrets: &Secrets<F>,
    t: &mut Transcript,
) -> (ZkPiopProof<F>, Context<F>) {
    // Step 1: declare the mask sums.
    let s_out = secrets.z_out.sum();
    let s_in = secrets.z_in.sum();
    t.absorb_field(s_out);
    t.absorb_field(s_in);
    // Step 2: bind.
    let tau_out = nonzero::<F>(t);
    let tau_in = nonzero::<F>(t);
    let r: Vec<_> = (0..rows).map(|_| t.squeeze_field::<F>()).collect();
    // Step 3: outer sumcheck on F_r + τ_out·Z_out, where Q̄_M = Q_M + η_M·x_ℓ(x_ℓ − 1)
    // differs from Q_M only in the last round.
    let mut eq_tbl = build_eq_table(&r);
    let mut q_tbls: Vec<_> = [&matrices.a, &matrices.b, &matrices.c]
        .iter()
        .map(|m| build_q_table(m, assignment, rows))
        .collect();
    let mut outer_polys = Vec::new();
    let mut x_star = Vec::new();
    t.absorb_field(tau_out * s_out);
    for i in 0..rows {
        let g_i: Vec<_> = (0..=OUTER_DEG)
            .map(|j| {
                let z = F::from(j as u64);
                let eq_z = fold(&eq_tbl, z);
                let q_z: Vec<_> = q_tbls
                    .iter()
                    .enumerate()
                    .map(|(m, q)| {
                        let mut q = fold(q, z);
                        if i + 1 == rows {
                            q[0] += secrets.eta[m] * z * (z - F::one());
                        }
                        q
                    })
                    .collect();
                eq_z.iter()
                    .enumerate()
                    .map(|(k, &e)| e * (q_z[0][k] * q_z[1][k] - q_z[2][k]))
                    .sum::<F>()
                    + tau_out * secrets.z_out.round(&x_star, z)
            })
            .collect();
        absorb(t, &g_i);
        let x_i = challenge(i + 1 == rows, t);
        x_star.push(x_i);
        outer_polys.push(g_i);
        eq_tbl = fold(&eq_tbl, x_i);
        q_tbls = q_tbls.iter().map(|q| fold(q, x_i)).collect();
    }
    // Step 4: send v_M = Q̄_M(x*) and Z_out(x*), then run the inner sumcheck on
    // L·w̃ + 2^{-ℓ}Δ + τ_in·Z_in.
    let phi = x_star[rows - 1] * (x_star[rows - 1] - F::one());
    let v_m = std::array::from_fn(|m| q_tbls[m][0] + secrets.eta[m] * phi);
    let z_out_eval = secrets.z_out.eval(&x_star);
    absorb(t, &v_m);
    t.absorb_field(z_out_eval);
    let varrho = t.squeeze_field::<F>();
    let gamma = t.squeeze_field::<F>();
    let delta = phi * (secrets.eta[0] + varrho * secrets.eta[1] + gamma * secrets.eta[2]);
    let mut l = build_l_table(
        &matrices.a,
        &matrices.b,
        &matrices.c,
        &x_star,
        varrho,
        gamma,
        cols,
    );
    let mut w = assignment.to_vec();
    let mut inner_polys = Vec::new();
    let mut y_star = Vec::new();
    t.absorb_field(v_m[0] + varrho * v_m[1] + gamma * v_m[2] + tau_in * s_in);
    let inv2 = F::from(2u64).inverse().unwrap();
    // The constant 2^{-ℓ}Δ summed over the trailing cube adds 2^{-i}Δ to g_i.
    let mut delta_round = delta;
    for _ in 0..cols {
        delta_round *= inv2;
        let g_i: Vec<_> = (0..=INNER_DEG)
            .map(|j| {
                let z = F::from(j as u64);
                fold(&l, z)
                    .iter()
                    .zip(fold(&w, z))
                    .map(|(&a, b)| a * b)
                    .sum::<F>()
                    + delta_round
                    + tau_in * secrets.z_in.round(&y_star, z)
            })
            .collect();
        absorb(t, &g_i);
        let y_i = t.squeeze_field::<F>();
        y_star.push(y_i);
        inner_polys.push(g_i);
        l = fold(&l, y_i);
        w = fold(&w, y_i);
    }
    let terminal = lagrange_eval(inner_polys.last().unwrap(), *y_star.last().unwrap());
    let matrix_evals =
        [&matrices.a, &matrices.b, &matrices.c].map(|m| mle_of_matrix_at(m, &x_star, &y_star));
    absorb(t, &matrix_evals);
    (
        ZkPiopProof {
            outer_sc: SumcheckProof {
                round_polys: outer_polys,
            },
            v_m,
            inner_sc: SumcheckProof {
                round_polys: inner_polys,
            },
            matrix_evals,
            s_out,
            s_in,
            z_out_eval,
        },
        Context {
            x_star,
            y_star,
            varrho,
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
    // Steps 1–2.
    t.absorb_field(p.s_out);
    t.absorb_field(p.s_in);
    let tau_out = nonzero::<F>(t);
    let tau_in = nonzero::<F>(t);
    let r: Vec<_> = (0..rows).map(|_| t.squeeze_field::<F>()).collect();
    // Step 3, with claimed sum τ_out·S_out.
    let mut expected = tau_out * p.s_out;
    t.absorb_field(expected);
    let mut x_star = Vec::new();
    for (i, g_i) in p.outer_sc.round_polys.iter().enumerate() {
        if g_i.len() != OUTER_DEG + 1 || g_i[0] + g_i[1] != expected {
            return None;
        }
        absorb(t, g_i);
        let x_i = challenge(i + 1 == rows, t);
        x_star.push(x_i);
        expected = lagrange_eval(g_i, x_i);
    }
    // Step 4: g_ℓ(x*_ℓ) − τ_out·Z_out(x*) = eq(x*, r)·(v_A·v_B − v_C).
    if expected != eq(&r, &x_star) * (p.v_m[0] * p.v_m[1] - p.v_m[2]) + tau_out * p.z_out_eval {
        return None;
    }
    absorb(t, &p.v_m);
    t.absorb_field(p.z_out_eval);
    let varrho = t.squeeze_field::<F>();
    let gamma = t.squeeze_field::<F>();
    expected = p.v_m[0] + varrho * p.v_m[1] + gamma * p.v_m[2] + tau_in * p.s_in;
    t.absorb_field(expected);
    let mut y_star = Vec::new();
    for g_i in &p.inner_sc.round_polys {
        if g_i.len() != INNER_DEG + 1 || g_i[0] + g_i[1] != expected {
            return None;
        }
        absorb(t, g_i);
        let y_i = t.squeeze_field::<F>();
        y_star.push(y_i);
        expected = lagrange_eval(g_i, y_i);
    }
    absorb(t, &p.matrix_evals);
    // The inner terminal check is deferred to the private opening built by `relation`.
    Some(Context {
        x_star,
        y_star,
        varrho,
        gamma,
        tau_in,
        terminal: expected,
    })
}

/// Construction 13's single private opening: batch the four linear claims on `a` with
/// independent coefficients β, sampled after every claimed value is fixed:
///   β_0: S_out,  β_1: Z_out(x*),  β_2: S_in,
///   β_3: g_ℓ(y*_ℓ) = L(y*)·w̃(y*) + 2^{-ℓ}Δ + τ_in·Z_in(y*).
/// Returns the weight and the claimed value v of Section 7.2 (1).
pub(crate) fn relation<F: PrimeField>(
    p: &ZkPiopProof<F>,
    c: &Context<F>,
    public: &[F],
    t: &mut Transcript,
) -> (LinearForm<F>, F) {
    let rows = c.x_star.len();
    let cols = c.y_star.len();
    // Layout of `a` (see `Secrets::pack`): private w | η_A, η_B, η_C | Z_out | Z_in.
    let half = 1usize << (cols - 1);
    let outer_start = half + 3;
    let inner_start = outer_start + 1 + OUTER_DEG * rows;
    let len = (inner_start + 1 + INNER_DEG * cols).next_power_of_two();
    let mut form = LinearForm::new(len.ilog2() as usize);
    t.absorb(b"zk-private-relations/independent-batch-v2");
    let beta: [F; 4] = std::array::from_fn(|_| t.squeeze_field::<F>());
    let count_out = F::from(2u64).pow([rows as u64]);
    let count_in = F::from(2u64).pow([cols as u64]);
    let inv2 = F::from(2u64).inverse().unwrap();
    // S_out = 2^ℓ·c_0 + 2^{ℓ-1}·Σ c_{i,j} and Z_out(x*) = c_0 + Σ c_{i,j}·(x*_i)^j.
    form.unit(outer_start, beta[0] * count_out + beta[1]);
    for (i, &x) in c.x_star.iter().enumerate() {
        let mut xp = F::one();
        for j in 0..OUTER_DEG {
            xp *= x;
            form.unit(
                outer_start + 1 + i * OUTER_DEG + j,
                beta[0] * count_out * inv2 + beta[1] * xp,
            );
        }
    }
    // S_in, and the τ_in·Z_in(y*) term of the inner terminal relation.
    form.unit(inner_start, beta[2] * count_in + beta[3] * c.tau_in);
    for (i, &y) in c.y_star.iter().enumerate() {
        let mut yp = F::one();
        for j in 0..INNER_DEG {
            yp *= y;
            form.unit(
                inner_start + 1 + i * INNER_DEG + j,
                beta[2] * count_in * inv2 + beta[3] * c.tau_in * yp,
            );
        }
    }
    // 2^{-ℓ}Δ = 2^{-ℓ}·x*_ℓ(x*_ℓ − 1)·(η_A + ϱ·η_B + γ·η_C).
    let phi = c.x_star[rows - 1] * (c.x_star[rows - 1] - F::one());
    let scale = beta[3] * phi * count_in.inverse().unwrap();
    for (i, coeff) in [F::one(), c.varrho, c.gamma].into_iter().enumerate() {
        form.unit(half + i, scale * coeff);
    }
    // L(y*)·w̃(y*): the private half stays in the weight; the public half y_last·ĩo(y_w)
    // is known to the verifier and moves to the right-hand side.
    let l = p.matrix_evals[0] + c.varrho * p.matrix_evals[1] + c.gamma * p.matrix_evals[2];
    let y_last = c.y_star[cols - 1];
    let y_w = &c.y_star[..cols - 1];
    let mut factors: Vec<_> = y_w.iter().map(|&y| [F::one() - y, y]).collect();
    factors.resize(form.vars, [F::one(), F::zero()]);
    form.terms
        .push((beta[3] * l * (F::one() - y_last), factors));
    let v = beta[0] * p.s_out
        + beta[1] * p.z_out_eval
        + beta[2] * p.s_in
        + beta[3] * (c.terminal - l * y_last * public_mle(public, y_w));
    (form, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{Field, One, UniformRand, Zero};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn private_batch_matches_four_independently_weighted_claims() {
        let mut rng = StdRng::seed_from_u64(714);
        for rows in 1..=3 {
            for cols in 2..=4 {
                let half = 1 << (cols - 1);
                let c = Context {
                    x_star: (0..rows).map(|_| F::rand(&mut rng)).collect(),
                    y_star: (0..cols).map(|_| F::rand(&mut rng)).collect(),
                    varrho: F::rand(&mut rng),
                    gamma: F::rand(&mut rng),
                    tau_in: F::rand(&mut rng),
                    terminal: F::rand(&mut rng),
                };
                let p = ZkPiopProof {
                    outer_sc: SumcheckProof {
                        round_polys: vec![],
                    },
                    inner_sc: SumcheckProof {
                        round_polys: vec![],
                    },
                    v_m: [F::zero(); 3],
                    matrix_evals: std::array::from_fn(|_| F::rand(&mut rng)),
                    s_out: F::rand(&mut rng),
                    s_in: F::rand(&mut rng),
                    z_out_eval: F::rand(&mut rng),
                };
                let public: Vec<_> = (0..half).map(|_| F::rand(&mut rng)).collect();
                let mut t = Transcript::new(b"batch-test");
                let (form, target) = relation(&p, &c, &public, &mut t);
                let mut replay = Transcript::new(b"batch-test");
                replay.absorb(b"zk-private-relations/independent-batch-v2");
                let beta: [F; 4] = std::array::from_fn(|_| replay.squeeze_field());
                assert_eq!(t.squeeze_field::<F>(), replay.squeeze_field::<F>());
                let l =
                    p.matrix_evals[0] + c.varrho * p.matrix_evals[1] + c.gamma * p.matrix_evals[2];
                let io = crate::r1cs::mle_of_vector(&public, cols - 1, &c.y_star[..cols - 1]);
                assert_eq!(
                    target,
                    beta[0] * p.s_out
                        + beta[1] * p.z_out_eval
                        + beta[2] * p.s_in
                        + beta[3] * (c.terminal - l * c.y_star[cols - 1] * io)
                );

                // Enumerate each separate linear claim on every basis vector;
                // this also checks that unused packed entries have zero weight.
                let weights = form.dense();
                let outer_start = half + 3;
                let inner_start = outer_start + 1 + OUTER_DEG * rows;
                for index in 0..weights.len() {
                    let mut basis = vec![F::zero(); weights.len()];
                    basis[index] = F::one();
                    let outer = Mask {
                        coeffs: basis[outer_start..inner_start].to_vec(),
                        vars: rows,
                        degree: OUTER_DEG,
                    };
                    let inner = Mask {
                        coeffs: basis[inner_start..inner_start + 1 + INNER_DEG * cols].to_vec(),
                        vars: cols,
                        degree: INNER_DEG,
                    };
                    let cube_sum = |mask: &Mask<F>| -> F {
                        (0..1 << mask.vars)
                            .map(|i| {
                                let point: Vec<_> = (0..mask.vars)
                                    .map(|j| F::from(((i >> j) & 1) as u64))
                                    .collect();
                                mask.eval(&point)
                            })
                            .sum()
                    };
                    let w =
                        crate::r1cs::mle_of_vector(&basis[..half], cols - 1, &c.y_star[..cols - 1]);
                    let phi = c.x_star[rows - 1] * (c.x_star[rows - 1] - F::one());
                    let delta = phi
                        * (basis[half] + c.varrho * basis[half + 1] + c.gamma * basis[half + 2]);
                    let terminal = l * (F::one() - c.y_star[cols - 1]) * w
                        + delta / F::from((1 << cols) as u64)
                        + c.tau_in * inner.eval(&c.y_star);
                    assert_eq!(
                        weights[index],
                        beta[0] * cube_sum(&outer)
                            + beta[1] * outer.eval(&c.x_star)
                            + beta[2] * cube_sum(&inner)
                            + beta[3] * terminal
                    );
                }
            }
        }
    }

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

    fn inner_view(mut l: Vec<F>, mut w: Vec<F>, mut delta: F, y: &[F]) -> Vec<F> {
        let mut out = vec![F::zero()]; // Keep the declared inner mask sum fixed.
        for &yi in y {
            delta /= F::from(2);
            for j in 0..=INNER_DEG {
                let z = F::from(j as u64);
                out.push(crate::zk_pcs::dot(&fold(&l, z), &fold(&w, z)) + delta);
            }
            l = fold(&l, yi);
            w = fold(&w, yi);
        }
        // The inner mask evaluation is private, so it need not stay fixed.
        out
    }

    fn compensate(mask: &Mask<F>, first: &[F], second: &[F], x: &[F], tau: F) -> Mask<F> {
        let rhs: Vec<_> = first
            .iter()
            .zip(second)
            .map(|(&a, &b)| (a - b) / tau)
            .collect();
        let columns: Vec<_> = (0..mask.coeffs.len())
            .map(|i| {
                let mut coeffs = vec![F::zero(); mask.coeffs.len()];
                coeffs[i] = F::one();
                mask_view(
                    &Mask {
                        coeffs,
                        vars: mask.vars,
                        degree: mask.degree,
                    },
                    x,
                )
            })
            .collect();
        let matrix = (0..rhs.len())
            .map(|i| columns.iter().map(|c| c[i]).collect())
            .collect();
        let shift = solve(matrix, rhs).expect("witness change must be absorbed by the masks");
        Mask {
            coeffs: mask.coeffs.iter().zip(shift).map(|(&a, b)| a + b).collect(),
            vars: mask.vars,
            degree: mask.degree,
        }
    }

    #[test]
    fn joint_view_and_private_relation_survive_a_witness_change() {
        // C14: one change of randomness preserves BOTH sumchecks, the masked
        // terminal tuple, and the final private relation. Separate marginal
        // uniformity tests would not establish this composition property.
        let mut rng = StdRng::seed_from_u64(14013);
        for rows in 1..=3 {
            let cols = rows + 1;
            let n = 1 << rows;
            // Public instance is 1. Constraints w_i * 1 = w_i admit both traces.
            let a: Vec<_> = (0..n).map(|i| vec![(F::one(), i)]).collect();
            let matrices = ConstraintMatrices {
                a: a.clone(),
                b: vec![vec![(F::one(), n)]; n],
                c: a,
                num_instance_variables: 1,
                num_witness_variables: n,
                num_constraints: n,
            };
            let first_w: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
            let second_w: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
            let assignment = |w: &[F]| {
                let mut v = w.to_vec();
                v.resize(2 * n, F::zero());
                v[n] = F::one();
                v
            };
            let r: Vec<_> = (0..rows).map(|_| F::rand(&mut rng)).collect();
            let mut x: Vec<_> = (0..rows).map(|_| F::rand(&mut rng)).collect();
            x[rows - 1] = F::from(7); // Last outer challenge excludes 0 and 1.
            let y: Vec<_> = (0..cols).map(|_| F::rand(&mut rng)).collect();
            let (varrho, gamma, tau_out, tau_in) =
                (F::from(11), F::from(13), F::from(17), F::from(19));
            let first = Secrets::sample(rows, cols, &mut rng);
            let phi = x[rows - 1] * (x[rows - 1] - F::one());
            let eq = build_eq_table(&x);
            let difference =
                (crate::zk_pcs::dot(&first_w, &eq) - crate::zk_pcs::dot(&second_w, &eq)) / phi;
            let eta = [
                first.eta[0] + difference,
                first.eta[1],
                first.eta[2] + difference,
            ];
            let outer_a = outer_view(
                [first_w.clone(), vec![F::one(); n], first_w.clone()],
                first.eta,
                &r,
                &x,
            );
            let outer_b = outer_view(
                [second_w.clone(), vec![F::one(); n], second_w.clone()],
                eta,
                &r,
                &x,
            );
            let outer = compensate(&first.z_out, &outer_a, &outer_b, &x, tau_out);
            let masked = |base: &[F], mask: &Mask<F>, point: &[F], tau: F| -> Vec<F> {
                base.iter()
                    .zip(mask_view(mask, point))
                    .map(|(&a, b)| a + tau * b)
                    .collect()
            };
            let outer_view = masked(&outer_a, &first.z_out, &x, tau_out);
            assert_eq!(outer_view, masked(&outer_b, &outer, &x, tau_out));
            let l = build_l_table(
                &matrices.a,
                &matrices.b,
                &matrices.c,
                &x,
                varrho,
                gamma,
                cols,
            );
            let delta = |eta: [F; 3]| phi * (eta[0] + varrho * eta[1] + gamma * eta[2]);
            let inner_a = inner_view(l.clone(), assignment(&first_w), delta(first.eta), &y);
            let inner_b = inner_view(l, assignment(&second_w), delta(eta), &y);
            let inner = compensate(&first.z_in, &inner_a, &inner_b, &y, tau_in);
            let inner_view = masked(&inner_a, &first.z_in, &y, tau_in);
            assert_eq!(inner_view, masked(&inner_b, &inner, &y, tau_in));
            let second = Secrets {
                eta,
                z_out: outer,
                z_in: inner,
            };
            let q = |w: &[F], eta: [F; 3]| {
                let w = crate::zk_pcs::dot(w, &eq);
                [w + eta[0] * phi, F::one() + eta[1] * phi, w + eta[2] * phi]
            };
            assert_eq!(q(&first_w, first.eta), q(&second_w, second.eta));
            let inner_sc = SumcheckProof {
                round_polys: inner_view[1..]
                    .chunks_exact(INNER_DEG + 1)
                    .map(<[F]>::to_vec)
                    .collect(),
            };
            let terminal = lagrange_eval(inner_sc.round_polys.last().unwrap(), y[cols - 1]);
            let p = ZkPiopProof {
                outer_sc: SumcheckProof {
                    round_polys: outer_view[1..1 + rows * (OUTER_DEG + 1)]
                        .chunks_exact(OUTER_DEG + 1)
                        .map(<[F]>::to_vec)
                        .collect(),
                },
                v_m: q(&first_w, first.eta),
                inner_sc,
                matrix_evals: [&matrices.a, &matrices.b, &matrices.c]
                    .map(|m| mle_of_matrix_at(m, &x, &y)),
                s_out: first.z_out.sum(),
                s_in: first.z_in.sum(),
                z_out_eval: first.z_out.eval(&x),
            };
            let context = Context {
                x_star: x,
                y_star: y,
                varrho,
                gamma,
                tau_in,
                terminal,
            };
            let (form, target) = relation(
                &p,
                &context,
                &[F::one()],
                &mut Transcript::new(b"joint-view"),
            );
            let weights = form.dense();
            assert_eq!(crate::zk_pcs::dot(&first.pack(&first_w), &weights), target);
            assert_eq!(
                crate::zk_pcs::dot(&second.pack(&second_w), &weights),
                target
            );
        }
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
            for i in 0..3 {
                assert_eq!(
                    crate::zk_pcs::dot(&first[i], &eq) + eta[i] * phi,
                    crate::zk_pcs::dot(&second[i], &eq) + adjusted[i] * phi
                );
            }
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
