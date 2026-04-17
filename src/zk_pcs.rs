//! Single-use hiding linear-relation openings by random table padding.
//! Chapter: Section 7.2, (1) Blinding the witness oracle, with hiding from Proposition 7.
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;
use rand::{CryptoRng, Rng};

use crate::{
    merkle::{
        BLOWUP, Hash, MerkleDomain, PRIVACY_SALT_BYTES, PrivacySalt, build_tree_in_domain,
        num_queries, private_leaf,
    },
    pcs::{self, Commitment, FoldProver, PcsError, rs_encode, verify_leaf_bytes},
    sumcheck::{SumcheckProof, lagrange_eval},
    transcript::Transcript,
};

/// Table expansion 2^k for the least k with 2^(k-1) > t = num_queries().
pub fn padding_factor() -> usize {
    2 * (num_queries() + 1).next_power_of_two()
}

/// Resamples until nonzero, where Proposition 10's simulator would instead abort.
pub(crate) fn nonzero<F: PrimeField>(t: &mut Transcript) -> F {
    loop {
        let x = t.squeeze_field::<F>();
        if !x.is_zero() {
            return x;
        }
    }
}

pub(crate) fn absorb<F: PrimeField>(t: &mut Transcript, xs: &[F]) {
    for &x in xs {
        t.absorb_field(x);
    }
}

#[cfg(test)]
pub(crate) fn dot<F: PrimeField>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn fold<F: PrimeField>(xs: &[F], x: F) -> Vec<F> {
    xs.chunks_exact(2)
        .map(|p| p[0] + x * (p[1] - p[0]))
        .collect()
}

/// The weight that replaces `eq(·, z*)` for a linear claim (Section 7.2 (1)), kept in
/// tensor form for succinct verification; bit zero folds first.
#[derive(Clone, CanonicalSerialize)]
pub(crate) struct LinearForm<F: PrimeField> {
    pub vars: usize,
    pub terms: Vec<(F, Vec<[F; 2]>)>,
}

impl<F: PrimeField> LinearForm<F> {
    pub fn new(vars: usize) -> Self {
        Self {
            vars,
            terms: Vec::new(),
        }
    }

    pub fn unit(&mut self, i: usize, scale: F) {
        assert!(i < 1usize << self.vars);
        self.terms.push((
            scale,
            (0..self.vars)
                .map(|j| {
                    if (i >> j) & 1 == 0 {
                        [F::one(), F::zero()]
                    } else {
                        [F::zero(), F::one()]
                    }
                })
                .collect(),
        ));
    }

    fn valid(&self) -> bool {
        self.vars < usize::BITS as usize
            && self
                .terms
                .iter()
                .all(|(_, factors)| factors.len() == self.vars)
    }

    fn eval(&self, point: &[F]) -> F {
        self.terms
            .iter()
            .map(|(scale, factors)| {
                *scale
                    * factors
                        .iter()
                        .zip(point)
                        .map(|(a, &x)| a[0] + x * (a[1] - a[0]))
                        .product::<F>()
            })
            .sum()
    }

    pub fn dense(&self) -> Vec<F> {
        let mut out = vec![F::zero(); 1 << self.vars];
        for (scale, factors) in &self.terms {
            let mut term = vec![*scale];
            for f in factors {
                let n = term.len();
                term.resize(2 * n, F::zero());
                for i in 0..n {
                    term[n + i] = term[i] * f[1];
                    term[i] *= f[0];
                }
            }
            for (a, b) in out.iter_mut().zip(term) {
                *a += b;
            }
        }
        out
    }
}

fn padded_len<F: FftField>(vars: usize) -> Result<usize, PcsError> {
    let n = 1usize
        .checked_shl(vars.try_into().map_err(|_| PcsError::InvalidShape)?)
        .and_then(|n| n.checked_mul(padding_factor()))
        .ok_or(PcsError::InvalidShape)?;
    let size = n.checked_mul(BLOWUP).ok_or(PcsError::InvalidDomain)?;
    Radix2EvaluationDomain::<F>::new(size).ok_or(PcsError::InvalidDomain)?;
    Ok(n)
}

/// Not Clone: consuming this witness enforces one opening per random padding.
pub(crate) struct Witness<F: PrimeField> {
    inner: pcs::Witness<F>,
    salts: Vec<Vec<PrivacySalt>>,
}

/// Commit to the rate-ρ encoding of `ã(z) + R(u, z)`, with `R` uniform subject to `R(0, z) = 0`.
pub(crate) fn commit<F: PrimeField + FftField, R: Rng + CryptoRng>(
    a: Vec<F>,
    rng: &mut R,
) -> Result<(Hash, Witness<F>), PcsError> {
    if !a.len().is_power_of_two() {
        return Err(PcsError::InvalidShape);
    }
    let n = padded_len::<F>(a.len().ilog2() as usize)?;
    let factor = padding_factor();
    // The padding variables u are the LOW bits, so the PCS folds them before z.
    // On the cube, u = 0 holds ã(z) and every u ≠ 0 holds a uniform value.
    let evals: Vec<_> = (0..n)
        .map(|i| {
            if i % factor == 0 {
                a[i / factor]
            } else {
                F::rand(rng)
            }
        })
        .collect();
    let codeword = rs_encode(&evals);
    // Sample independent leaf salts for every committed layer. Only salts for
    // opened positions are revealed.
    let salts: Vec<Vec<_>> = (0..n.ilog2())
        .map(|layer| {
            (0..codeword.len() >> layer)
                .map(|_| {
                    let mut salt = [0; PRIVACY_SALT_BYTES];
                    rng.fill_bytes(&mut salt);
                    salt
                })
                .collect()
        })
        .collect();
    let tree = build_tree_in_domain(
        codeword
            .iter()
            .zip(&salts[0])
            .map(|(&v, salt)| private_leaf(v, salt)),
        &MerkleDomain::private(0),
    );
    Ok((
        tree.root(),
        Witness {
            inner: pcs::Witness {
                evals,
                codeword,
                tree,
            },
            salts,
        },
    ))
}

#[derive(CanonicalSerialize)]
pub struct RelationProof<F: PrimeField> {
    pub sc: SumcheckProof<F>,
    pub folds: pcs::Proof<F>,
    /// Per query, per layer, salts for both authenticated leaves.
    pub leaf_salts: Vec<Vec<[PrivacySalt; 2]>>,
}

fn bind<F: PrimeField>(root: &[u8], form: &LinearForm<F>, v: F, t: &mut Transcript) {
    t.absorb(b"zk-linear-relation/padded-v2");
    t.absorb(root);
    t.absorb(&(padding_factor() as u64).to_le_bytes());
    t.absorb(&(num_queries() as u64).to_le_bytes());
    t.absorb(&(BLOWUP as u64).to_le_bytes());
    let mut bytes = Vec::new();
    form.serialize_compressed(&mut bytes).unwrap();
    t.absorb(&bytes);
    t.absorb_field(v);
}

/// Evaluations at 0, 1, 2 of sum_z A(X,z) B(X,z).
fn round_polynomial<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
    let mut g = vec![F::zero(); 3];
    for (a, b) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        g[0] += a[0] * b[0];
        g[1] += a[1] * b[1];
        g[2] += (a[1].double() - a[0]) * (b[1].double() - b[0]);
    }
    g
}

fn fold_in_place<F: PrimeField>(a: &mut Vec<F>, x: F) {
    let half = a.len() / 2;
    for i in 0..half {
        a[i] = a[2 * i] + x * (a[2 * i + 1] - a[2 * i]);
    }
    a.truncate(half);
}

/// Prove `Σ_{u,z} [ã(z) + R(u,z)]·eq(u, 0)·weight(z) = v`, folding along each challenge
/// (the opening procedure of Construction 12, Step 4, processing u before z).
pub(crate) fn prove<F: PrimeField + FftField>(
    witness: Witness<F>,
    form: LinearForm<F>,
    v: F,
    t: &mut Transcript,
) -> Result<RelationProof<F>, PcsError> {
    if !form.valid() || witness.inner.evals.len() != padded_len::<F>(form.vars)? {
        return Err(PcsError::InvalidShape);
    }
    bind(&witness.inner.tree.root(), &form, v, t);
    // Table of ã(z) + R(u, z), and of eq(u, 0)·weight(z).
    let mut table = witness.inner.evals.clone();
    let mut weight = vec![F::zero(); table.len()];
    for (i, value) in form.dense().into_iter().enumerate() {
        weight[i * padding_factor()] = value;
    }
    let rounds = table.len().ilog2() as usize;
    let mut folds = FoldProver::new(&witness.inner.codeword)?;
    let mut round_polys = Vec::with_capacity(rounds);
    for i in 0..rounds {
        let g_i = round_polynomial(&table, &weight);
        absorb(t, &g_i);
        // Proposition 7 needs α_1 ≠ 0.
        let challenge = if i == 0 {
            nonzero(t)
        } else {
            t.squeeze_field()
        };
        folds.advance_private(challenge, witness.salts.get(i + 1).map(Vec::as_slice));
        folds.absorb_latest(t);
        fold_in_place(&mut table, challenge);
        fold_in_place(&mut weight, challenge);
        round_polys.push(g_i);
    }
    let folds = folds.finish(&witness.inner, t);
    let leaf_salts = folds
        .queries
        .iter()
        .map(|q| {
            (0..rounds)
                .map(|i| {
                    [
                        witness.salts[i][q.a_paths[i].leaf_index],
                        witness.salts[i][q.b_paths[i].leaf_index],
                    ]
                })
                .collect()
        })
        .collect();
    Ok(RelationProof {
        sc: SumcheckProof { round_polys },
        folds,
        leaf_salts,
    })
}

pub(crate) fn verify<F: PrimeField + FftField>(
    root: &[u8],
    form: LinearForm<F>,
    v: F,
    proof: &RelationProof<F>,
    t: &mut Transcript,
) -> Result<bool, PcsError> {
    if !form.valid() {
        return Ok(false);
    }
    let rounds = padded_len::<F>(form.vars)?.ilog2() as usize;
    if root.len() != 32
        || proof.sc.round_polys.len() != rounds
        || proof.folds.intermediate_roots.len() != rounds - 1
        || proof.leaf_salts.len() != num_queries()
        || proof.leaf_salts.iter().any(|salts| salts.len() != rounds)
    {
        return Ok(false);
    }
    bind(root, &form, v, t);
    let mut expected = v;
    let mut point = Vec::with_capacity(rounds);
    for (i, g_i) in proof.sc.round_polys.iter().enumerate() {
        if g_i.len() != 3 || g_i[0] + g_i[1] != expected {
            return Ok(false);
        }
        absorb(t, g_i);
        let challenge = if i == 0 {
            nonzero(t)
        } else {
            t.squeeze_field()
        };
        expected = lagrange_eval(g_i, challenge);
        point.push(challenge);
        if proof.folds.absorb_round(i, rounds, t).is_none() {
            return Ok(false);
        }
    }
    // Terminal point (α, s): check g_{k+m}(s_m) = p·eq(α, 0)·weight(s), p the folded value.
    let k = padding_factor().ilog2() as usize;
    let (alpha, s) = point.split_at(k);
    let eq_alpha_0: F = alpha.iter().map(|&a| F::one() - a).product();
    if expected != proof.folds.final_value * eq_alpha_0 * form.eval(s) {
        return Ok(false);
    }
    let commitment = Commitment {
        root: root.to_vec(),
    };
    pcs::verify_queries_with_authentication(
        &point,
        proof.folds.final_value,
        &proof.folds,
        t,
        |qi, layer, low, size, q| {
            let root = if layer == 0 {
                &commitment.root
            } else {
                &proof.folds.intermediate_roots[layer - 1]
            };
            let domain = MerkleDomain::private(layer);
            verify_leaf_bytes(
                &private_leaf(q.a_vals[layer], &proof.leaf_salts[qi][layer][0]),
                low,
                size,
                &q.a_paths[layer],
                root,
                &domain,
            ) && verify_leaf_bytes(
                &private_leaf(q.b_vals[layer], &proof.leaf_salts[qi][layer][1]),
                low + size / 2,
                size,
                &q.b_paths[layer],
                root,
                &domain,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr as F;
    use ark_ff::{Field, One, UniformRand, Zero};
    use rand::{SeedableRng, rngs::StdRng};

    fn rng() -> StdRng {
        StdRng::seed_from_u64(718)
    }

    #[test]
    fn padded_relations_bind_the_statement_and_all_opening_messages() {
        let mut rng = rng();
        for vars in [0, 2, 4] {
            let message: Vec<_> = (0..1 << vars).map(|_| F::rand(&mut rng)).collect();
            let mut form = LinearForm::new(vars);
            form.unit(0, F::from(9));
            form.terms
                .push((F::from(3), vec![[F::from(2), F::from(7)]; vars]));
            let target = dot(&message, &form.dense());
            let (root, witness) = commit(message.clone(), &mut rng).unwrap();
            assert_eq!(
                witness
                    .inner
                    .evals
                    .iter()
                    .step_by(padding_factor())
                    .copied()
                    .collect::<Vec<_>>(),
                message
            );
            let mut pt = Transcript::new(b"padded-test");
            let mut proof = prove(witness, form.clone(), target, &mut pt).unwrap();
            let mut vt = Transcript::new(b"padded-test");
            assert!(verify(&root, form.clone(), target, &proof, &mut vt).unwrap());
            assert_eq!(pt.squeeze_field::<F>(), vt.squeeze_field::<F>());
            let accepts = |p: &RelationProof<F>| {
                verify(
                    &root,
                    form.clone(),
                    target,
                    p,
                    &mut Transcript::new(b"padded-test"),
                )
                .unwrap()
            };
            assert!(
                !verify(
                    &root,
                    form.clone(),
                    target + F::one(),
                    &proof,
                    &mut Transcript::new(b"padded-test")
                )
                .unwrap()
            );
            let mut other = form.clone();
            other.unit(0, F::one());
            assert!(
                !verify(
                    &root,
                    other,
                    target,
                    &proof,
                    &mut Transcript::new(b"padded-test")
                )
                .unwrap()
            );
            let mut other_root = root.clone();
            other_root[0] ^= 1;
            assert!(
                !verify(
                    &other_root,
                    form.clone(),
                    target,
                    &proof,
                    &mut Transcript::new(b"padded-test")
                )
                .unwrap()
            );
            assert!(
                !verify(
                    &root,
                    form.clone(),
                    target,
                    &proof,
                    &mut Transcript::new(b"old-protocol")
                )
                .unwrap()
            );

            proof.sc.round_polys[0][2] += F::one();
            assert!(!accepts(&proof));
            proof.sc.round_polys[0][2] -= F::one();
            proof.folds.final_value += F::one();
            assert!(!accepts(&proof));
            proof.folds.final_value -= F::one();
            let first_two_challenges = |p: &RelationProof<F>| {
                let mut t = Transcript::new(b"padded-test");
                bind(&root, &form, target, &mut t);
                absorb(&mut t, &p.sc.round_polys[0]);
                let first = nonzero::<F>(&mut t);
                p.folds
                    .absorb_round(0, p.sc.round_polys.len(), &mut t)
                    .unwrap();
                absorb(&mut t, &p.sc.round_polys[1]);
                [first, t.squeeze_field::<F>()]
            };
            let original = first_two_challenges(&proof);
            assert!(!original[0].is_zero());
            proof.folds.intermediate_roots[0][0] ^= 1;
            let changed = first_two_challenges(&proof);
            assert_eq!(original[0], changed[0]);
            assert_ne!(original[1], changed[1]);
            assert!(!accepts(&proof));
            proof.folds.intermediate_roots[0][0] ^= 1;

            for layer in [0, 1] {
                proof.folds.queries[0].a_vals[layer] += F::one();
                assert!(!accepts(&proof));
                proof.folds.queries[0].a_vals[layer] -= F::one();
                proof.folds.queries[0].b_vals[layer] += F::one();
                assert!(!accepts(&proof));
                proof.folds.queries[0].b_vals[layer] -= F::one();
                proof.folds.queries[0].a_paths[layer].leaf_index ^= 1;
                assert!(!accepts(&proof));
                proof.folds.queries[0].a_paths[layer].leaf_index ^= 1;
                let digest = proof.folds.queries[0].b_paths[layer]
                    .auth_path
                    .pop()
                    .unwrap();
                assert!(!accepts(&proof));
                proof.folds.queries[0].b_paths[layer].auth_path.push(digest);
                let byte = proof.folds.queries[0].a_paths[layer]
                    .leaf_sibling_hash
                    .pop()
                    .unwrap();
                assert!(!accepts(&proof));
                proof.folds.queries[0].a_paths[layer]
                    .leaf_sibling_hash
                    .push(byte);
            }
            for layer in 0..proof.leaf_salts[0].len() {
                for side in 0..2 {
                    proof.leaf_salts[0][layer][side][0] ^= 1;
                    assert!(!accepts(&proof));
                    proof.leaf_salts[0][layer][side][0] ^= 1;
                }
            }
            let salt = proof.leaf_salts[0].pop().unwrap();
            assert!(!accepts(&proof));
            proof.leaf_salts[0].push(salt);
            proof.leaf_salts[0].push(salt);
            assert!(!accepts(&proof));
            proof.leaf_salts[0].pop();
            let salts = proof.leaf_salts.pop().unwrap();
            assert!(!accepts(&proof));
            proof.leaf_salts.push(salts);
            let query = proof.folds.queries.pop().unwrap();
            assert!(!accepts(&proof));
            proof.folds.queries.push(query);
            let round = proof.sc.round_polys.pop().unwrap();
            assert!(!accepts(&proof));
            proof.sc.round_polys.push(round);
            let root = proof.folds.intermediate_roots.pop().unwrap();
            assert!(!accepts(&proof));
            proof.folds.intermediate_roots.push(root);
            assert!(accepts(&proof));
        }
    }

    #[test]
    fn padding_hides_joint_sumcheck_and_fold_view_at_full_query_budget() {
        // Witness-change bijection preserving sumchecks, queried values and all folds.
        let mut rng = rng();
        let factor = padding_factor();
        assert!(factor / 2 > num_queries());
        assert!(factor / 4 <= num_queries());
        let a: Vec<_> = (0..4).map(|_| F::rand(&mut rng)).collect();
        let delta = [F::from(2), -F::one(), F::zero(), F::zero()];
        let weights = [F::one(), F::from(2), F::from(3), F::from(4)];
        assert_eq!(dot(&delta, &weights), F::zero());
        let (_, witness) = commit(a, &mut rng).unwrap();
        let domain = Radix2EvaluationDomain::<F>::new(witness.inner.codeword.len()).unwrap();
        let mut q = vec![F::one()];
        for i in 0..num_queries() {
            let square = domain.element(i).square();
            let mut next = vec![F::zero(); q.len() + 1];
            for (j, &c) in q.iter().enumerate() {
                next[j] -= square * c;
                next[j + 1] += c;
            }
            q = next;
        }
        let inv_q0 = q[0].inverse().unwrap();
        q.resize(factor / 2, F::zero());
        // Multilinear coefficient lift of q, converted to Boolean evaluations.
        for bit in 0..(factor / 2).ilog2() {
            for i in 0..q.len() {
                if i & (1 << bit) != 0 {
                    let lo = q[i ^ (1 << bit)];
                    q[i] += lo;
                }
            }
        }
        for alpha in [F::one(), F::from(13)] {
            let mut original = witness.inner.evals.clone();
            let mut changed = original.clone();
            for (i, value) in changed.iter_mut().enumerate() {
                let u = i % factor;
                *value += delta[i / factor]
                    * (F::one() - F::from((u & 1) as u64) / alpha)
                    * q[u >> 1]
                    * inv_q0;
            }
            for i in 0..4 {
                assert_eq!(changed[i * factor] - original[i * factor], delta[i]);
            }
            let encoded = rs_encode(&changed);
            for i in 0..num_queries() {
                assert_eq!(encoded[i], witness.inner.codeword[i]);
                let j = i + encoded.len() / 2;
                assert_eq!(encoded[j], witness.inner.codeword[j]);
            }
            let mut public = vec![F::zero(); original.len()];
            for i in 0..4 {
                public[i * factor] = weights[i];
            }
            let rounds = original.len().ilog2();
            for round in 0..rounds {
                assert_eq!(
                    round_polynomial(&original, &public),
                    round_polynomial(&changed, &public)
                );
                let x = if round == 0 { alpha } else { F::rand(&mut rng) };
                fold_in_place(&mut original, x);
                fold_in_place(&mut changed, x);
                fold_in_place(&mut public, x);
                assert_eq!(original, changed);
            }
        }
    }

    #[test]
    fn invalid_shapes_and_unsupported_domains_fail_before_padding() {
        assert!(matches!(
            commit(Vec::<F>::new(), &mut rng()),
            Err(PcsError::InvalidShape)
        ));
        assert!(matches!(
            commit(vec![F::one(); 3], &mut rng()),
            Err(PcsError::InvalidShape)
        ));
        assert_eq!(
            padded_len::<F>(usize::BITS as usize),
            Err(PcsError::InvalidShape)
        );
        assert_eq!(
            padded_len::<F>(F::TWO_ADICITY as usize),
            Err(PcsError::InvalidDomain)
        );
        let (_, witness) = commit(vec![F::one(); 4], &mut rng()).unwrap();
        let mut form = LinearForm::new(2);
        form.terms.push((F::one(), vec![[F::one(), F::one()]]));
        assert!(matches!(
            prove(witness, form, F::one(), &mut Transcript::new(b"shape")),
            Err(PcsError::InvalidShape)
        ));
    }
}
