//! Single-use, hiding linear-relation openings over randomized RS encodings.
//!
//! Code switching, masked one-round sumchecks and a masked-response base case
//! specialize Constructions 9.7, 6.3 and 7.2 of ePrint 2026/391. See README.md
//! for the construction overview, query budget, and limits of the security claim.
use ark_crypto_primitives::merkle_tree::{MerkleTree, Path};
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;
use rand::{CryptoRng, Rng};

use crate::{
    merkle::{BLOWUP, Hash, MerkleConfig, build_tree, num_queries},
    pcs::PcsError,
    sumcheck::lagrange_eval,
    transcript::Transcript,
};

fn aux_len() -> usize {
    (num_queries() + 1).next_power_of_two()
}

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

pub(crate) fn dot<F: PrimeField>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn fold<F: PrimeField>(xs: &[F], x: F) -> Vec<F> {
    xs.chunks_exact(2)
        .map(|p| p[0] + x * (p[1] - p[0]))
        .collect()
}

fn powers<F: PrimeField>(x: F, n: usize) -> Vec<F> {
    let mut p = F::one();
    (0..n)
        .map(|_| {
            let v = p;
            p *= x;
            v
        })
        .collect()
}

/// A sum of tensor products. The verifier folds this representation without
/// allocating a vector proportional to the witness size. Bit zero folds first.
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
    pub fn power(&mut self, x: F, scale: F) {
        let mut p = x;
        let factors = (0..self.vars)
            .map(|_| {
                let v = [F::one(), p];
                p.square_in_place();
                v
            })
            .collect();
        self.terms.push((scale, factors));
    }
    fn folded(&mut self, x: F, scale: F) {
        for (s, factors) in &mut self.terms {
            let a = factors.remove(0);
            *s *= scale * (a[0] + x * (a[1] - a[0]));
        }
        self.vars -= 1;
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

fn domain<F: FftField>(n: usize) -> Result<Radix2EvaluationDomain<F>, PcsError> {
    let size = n
        .checked_add(num_queries())
        .and_then(|n| n.checked_mul(BLOWUP))
        .and_then(usize::checked_next_power_of_two)
        .ok_or(PcsError::InvalidDomain)?;
    Radix2EvaluationDomain::new(size).ok_or(PcsError::InvalidDomain)
}

/// Message entries are the low univariate coefficients; a fresh random tail
/// hides any `num_queries()` distinct nonzero evaluation positions.
fn encode<F: FftField>(f: &[F], r: &[F]) -> Result<Vec<F>, PcsError> {
    if r.len() != num_queries() {
        return Err(PcsError::InvalidShape);
    }
    let d = domain::<F>(f.len())?;
    let mut a = f.to_vec();
    a.extend_from_slice(r);
    a.resize(d.size(), F::zero());
    d.fft_in_place(&mut a);
    Ok(a)
}

fn leaf<F: PrimeField>(values: &[F], salt: &[u8]) -> Vec<u8> {
    let mut b = b"zk-rs-leaf-v1".to_vec();
    values.serialize_compressed(&mut b).unwrap();
    b.extend_from_slice(salt);
    b
}

struct Oracle<F: PrimeField> {
    columns: Vec<Vec<F>>,
    salts: Vec<Vec<u8>>,
    tree: MerkleTree<MerkleConfig>,
}

impl<F: PrimeField> Oracle<F> {
    fn new<R: Rng + CryptoRng>(columns: Vec<Vec<F>>, rng: &mut R) -> Self {
        let salts: Vec<Vec<u8>> = (0..columns[0].len())
            .map(|_| {
                let mut s = vec![0; 32];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();
        let leaves: Vec<_> = salts
            .iter()
            .enumerate()
            .map(|(i, s)| leaf(&columns.iter().map(|c| c[i]).collect::<Vec<_>>(), s))
            .collect();
        Self {
            columns,
            salts,
            tree: build_tree(&leaves),
        }
    }
    fn open(&self, i: usize) -> Opening<F> {
        Opening {
            values: self.columns.iter().map(|c| c[i]).collect(),
            salt: self.salts[i].clone(),
            path: self.tree.generate_proof(i).unwrap(),
        }
    }
}

#[derive(CanonicalSerialize)]
pub struct Opening<F: PrimeField> {
    pub values: Vec<F>,
    pub salt: Vec<u8>,
    pub path: Path<MerkleConfig>,
}

struct Source {
    root: Hash,
    size: usize,
    columns: usize,
}

fn check<F: PrimeField>(s: &Source, p: &Opening<F>, i: usize) -> bool {
    s.root.len() == 32
        && p.values.len() == s.columns
        && p.salt.len() == 32
        && p.path.leaf_index == i
        && p.path.auth_path.len() + 1 == s.size.ilog2() as usize
        && p.path
            .verify(&(), &(), &s.root, leaf(&p.values, &p.salt).as_slice())
            .unwrap_or(false)
}

fn value<F: PrimeField>(p: &Opening<F>, alpha: Option<F>) -> F {
    match alpha {
        None => p.values[0],
        Some(a) => p.values[0] + a * (p.values[1] - p.values[0]),
    }
}

/// Deliberately not Clone: each randomized commitment may be opened only once.
/// Reusing its encoding randomness across proofs exceeds the hiding budget.
pub(crate) struct Witness<F: PrimeField> {
    f: Vec<F>,
    r: Vec<F>,
    oracle: Oracle<F>,
}

pub(crate) fn commit<F: PrimeField + FftField, R: Rng + CryptoRng>(
    f: Vec<F>,
    rng: &mut R,
) -> Result<(Hash, Witness<F>), PcsError> {
    if !f.len().is_power_of_two() {
        return Err(PcsError::InvalidShape);
    }
    let r = (0..num_queries()).map(|_| F::rand(rng)).collect::<Vec<_>>();
    let oracle = Oracle::new(vec![encode(&f, &r)?], rng);
    Ok((oracle.tree.root(), Witness { f, r, oracle }))
}

#[derive(CanonicalSerialize)]
pub struct Step<F: PrimeField> {
    pub switched_root: Hash,
    pub randomness_root: Hash,
    pub ood: F,
    pub queries: Vec<Opening<F>>,
    pub mask_root: Hash,
    pub polynomial: Vec<F>,
}

#[derive(CanonicalSerialize)]
pub struct Response<F: PrimeField> {
    pub message: Vec<F>,
    pub randomness: Vec<F>,
    pub source_queries: Vec<Opening<F>>,
    pub mask_queries: Vec<Opening<F>>,
}

#[derive(CanonicalSerialize)]
pub struct RelationProof<F: PrimeField> {
    pub steps: Vec<Step<F>>,
    pub mask_roots: Vec<Hash>,
    pub mask_target: F,
    pub responses: Vec<Response<F>>,
}

fn bind<F: PrimeField>(root: &[u8], form: &LinearForm<F>, target: F, t: &mut Transcript) {
    t.absorb(b"zk-linear-relation-v1");
    t.absorb(root);
    let mut bytes = Vec::new();
    form.serialize_compressed(&mut bytes).unwrap();
    t.absorb(&bytes);
    t.absorb_field(target);
}

fn ood_point<F: PrimeField>(size: usize, t: &mut Transcript) -> F {
    loop {
        let x = nonzero::<F>(t);
        if x.pow([size as u64]) != F::one() {
            return x;
        }
    }
}

fn switch_weights<F: PrimeField>(
    form: &mut LinearForm<F>,
    dense: Option<&mut Vec<F>>,
    u: F,
    xs: &[F],
    beta: F,
) -> Vec<F> {
    let n = 1usize << form.vars;
    let mut tail = vec![F::zero(); aux_len()];
    let mut dense = dense;
    let mut scale = beta;
    for (i, x) in std::iter::once(u).chain(xs.iter().copied()).enumerate() {
        form.power(x, scale);
        if let Some(d) = dense.as_mut() {
            for (a, p) in d.iter_mut().zip(powers(x, n)) {
                *a += scale * p;
            }
        }
        let limit = if i == 0 { aux_len() } else { num_queries() };
        let s = scale * x.pow([n as u64]);
        for (a, p) in tail[..limit].iter_mut().zip(powers(x, limit)) {
            *a += s * p;
        }
        scale *= beta;
    }
    tail
}

fn switch_target<F: PrimeField>(
    target: F,
    y: F,
    qs: &[Opening<F>],
    alpha: Option<F>,
    beta: F,
) -> F {
    let mut result = target + beta * y;
    let mut scale = beta.square();
    for q in qs {
        result += scale * value(q, alpha);
        scale *= beta;
    }
    result
}

pub(crate) fn prove<F: PrimeField + FftField, R: Rng + CryptoRng>(
    witness: Witness<F>,
    mut form: LinearForm<F>,
    mut target: F,
    t: &mut Transcript,
    rng: &mut R,
) -> Result<RelationProof<F>, PcsError> {
    let Witness {
        mut f,
        mut r,
        mut oracle,
    } = witness;
    if f.len() != 1usize << form.vars {
        return Err(PcsError::InvalidShape);
    }
    bind(&oracle.tree.root(), &form, target, t);
    let mut weights = form.dense();
    let mut alpha = None;
    let mut aux: Vec<Witness<F>> = Vec::new();
    let mut aux_weights: Vec<Vec<F>> = Vec::new();
    let mut steps = Vec::new();
    let inv2 = F::from(2u64).inverse().ok_or(PcsError::DivisionByZero)?;
    while f.len() > aux_len() {
        let n = f.len();
        let even: Vec<_> = f.iter().step_by(2).copied().collect();
        let odd: Vec<_> = f.iter().skip(1).step_by(2).copied().collect();
        let re: Vec<_> = (0..num_queries()).map(|_| F::rand(rng)).collect();
        let ro: Vec<_> = (0..num_queries()).map(|_| F::rand(rng)).collect();
        let switched = Oracle::new(vec![encode(&even, &re)?, encode(&odd, &ro)?], rng);
        let mut s = r.clone();
        s.resize_with(aux_len(), || F::rand(rng));
        let (s_root, s_wit) = commit(s, rng)?;
        t.absorb(&switched.tree.root());
        t.absorb(&s_root);
        let d = domain::<F>(n)?;
        let u = ood_point::<F>(d.size(), t);
        let y = dot(&f, &powers(u, n)) + u.pow([n as u64]) * dot(&s_wit.f, &powers(u, aux_len()));
        t.absorb_field(y);
        let indices = t.squeeze_indices(d.size(), num_queries());
        let queries: Vec<_> = indices.iter().map(|&i| oracle.open(i)).collect();
        for q in &queries {
            absorb(t, &q.values);
        }
        let beta = nonzero::<F>(t);
        let xs: Vec<_> = indices.iter().map(|&i| d.element(i)).collect();
        target = switch_target(target, y, &queries, alpha, beta);
        aux_weights.push(switch_weights(&mut form, Some(&mut weights), u, &xs, beta));
        aux.push(s_wit);

        // A uniform quadratic in the two-dimensional zero-Boolean-sum space.
        // Encoding this parametrization makes its zero sum true by construction.
        let z: Vec<_> = (0..aux_len()).map(|_| F::rand(rng)).collect();
        let (z_root, z_wit) = commit(z, rng)?;
        t.absorb(&z_root);
        let epsilon = nonzero::<F>(t);
        let aux_value: F = aux
            .iter()
            .zip(&aux_weights)
            .map(|(a, w)| dot(&a.f, w))
            .sum();
        let polynomial: Vec<_> = (0..3)
            .map(|i| {
                let x = F::from(i as u64);
                epsilon * (dot(&fold(&f, x), &fold(&weights, x)) + inv2 * aux_value)
                    + z_wit.f[0] * (x - inv2)
                    + z_wit.f[1] * (x.square() - inv2)
            })
            .collect();
        debug_assert_eq!(polynomial[0] + polynomial[1], epsilon * target);
        absorb(t, &polynomial);
        let a = t.squeeze_field::<F>();
        target = lagrange_eval(&polynomial, a);
        f = fold(&f, a);
        r = re.iter().zip(&ro).map(|(&e, &o)| e + a * (o - e)).collect();
        weights = fold(&weights, a).into_iter().map(|v| epsilon * v).collect();
        form.folded(a, epsilon);
        for w in &mut aux_weights {
            for v in w {
                *v *= epsilon * inv2;
            }
        }
        let mut zw = vec![F::zero(); aux_len()];
        zw[0] = a - inv2;
        zw[1] = a.square() - inv2;
        aux_weights.push(zw);
        aux.push(z_wit);
        steps.push(Step {
            switched_root: switched.tree.root(),
            randomness_root: s_root,
            ood: y,
            queries,
            mask_root: z_root,
            polynomial,
        });
        oracle = switched;
        alpha = Some(a);
    }

    let mut sources = vec![Witness { f, r, oracle }];
    sources.extend(aux);
    let mut all_weights = vec![weights];
    all_weights.extend(aux_weights);
    let mut masks = Vec::new();
    let mut mask_roots = Vec::new();
    let mut mask_target = F::zero();
    for (source, w) in sources.iter().zip(&all_weights) {
        let g = (0..source.f.len()).map(|_| F::rand(rng)).collect();
        let (root, mask) = commit(g, rng)?;
        mask_target += dot(&mask.f, w);
        t.absorb(&root);
        mask_roots.push(root);
        masks.push(mask);
    }
    t.absorb_field(mask_target);
    let gamma = nonzero::<F>(t);
    let mut responses: Vec<_> = sources
        .iter()
        .zip(&masks)
        .map(|(s, g)| {
            let message: Vec<_> = s.f.iter().zip(&g.f).map(|(&s, &g)| g + gamma * s).collect();
            let randomness: Vec<_> = s.r.iter().zip(&g.r).map(|(&s, &g)| g + gamma * s).collect();
            absorb(t, &message);
            absorb(t, &randomness);
            Response {
                message,
                randomness,
                source_queries: Vec::new(),
                mask_queries: Vec::new(),
            }
        })
        .collect();
    for ((response, source), mask) in responses.iter_mut().zip(&sources).zip(&masks) {
        let indices = t.squeeze_indices(source.oracle.columns[0].len(), num_queries());
        response.source_queries = indices.iter().map(|&i| source.oracle.open(i)).collect();
        response.mask_queries = indices.iter().map(|&i| mask.oracle.open(i)).collect();
    }
    Ok(RelationProof {
        steps,
        mask_roots,
        mask_target,
        responses,
    })
}

pub(crate) fn verify<F: PrimeField + FftField>(
    root: &[u8],
    mut form: LinearForm<F>,
    mut target: F,
    proof: &RelationProof<F>,
    t: &mut Transcript,
) -> Result<bool, PcsError> {
    let mut n = 1usize
        .checked_shl(form.vars as u32)
        .ok_or(PcsError::InvalidShape)?;
    let rounds = form.vars.saturating_sub(aux_len().ilog2() as usize);
    if proof.steps.len() != rounds
        || proof.responses.len() != 1 + 2 * rounds
        || proof.mask_roots.len() != proof.responses.len()
    {
        return Ok(false);
    }
    bind(root, &form, target, t);
    let mut source = Source {
        root: root.to_vec(),
        size: domain::<F>(n)?.size(),
        columns: 1,
    };
    let mut alpha = None;
    let mut aux_sources = Vec::new();
    let mut aux_weights = Vec::new();
    let inv2 = F::from(2u64).inverse().ok_or(PcsError::DivisionByZero)?;
    for step in &proof.steps {
        if step.switched_root.len() != 32
            || step.randomness_root.len() != 32
            || step.mask_root.len() != 32
            || step.polynomial.len() != 3
            || step.queries.len() != num_queries()
        {
            return Ok(false);
        }
        t.absorb(&step.switched_root);
        t.absorb(&step.randomness_root);
        let u = ood_point::<F>(source.size, t);
        t.absorb_field(step.ood);
        let indices = t.squeeze_indices(source.size, num_queries());
        for (&i, q) in indices.iter().zip(&step.queries) {
            if !check(&source, q, i) {
                return Ok(false);
            }
            absorb(t, &q.values);
        }
        let beta = nonzero::<F>(t);
        let d = domain::<F>(n)?;
        let xs: Vec<_> = indices.iter().map(|&i| d.element(i)).collect();
        target = switch_target(target, step.ood, &step.queries, alpha, beta);
        aux_weights.push(switch_weights(&mut form, None, u, &xs, beta));
        aux_sources.push(Source {
            root: step.randomness_root.clone(),
            size: domain::<F>(aux_len())?.size(),
            columns: 1,
        });
        t.absorb(&step.mask_root);
        let epsilon = nonzero::<F>(t);
        if step.polynomial[0] + step.polynomial[1] != epsilon * target {
            return Ok(false);
        }
        absorb(t, &step.polynomial);
        let a = t.squeeze_field::<F>();
        target = lagrange_eval(&step.polynomial, a);
        form.folded(a, epsilon);
        for w in &mut aux_weights {
            for v in w {
                *v *= epsilon * inv2;
            }
        }
        let mut zw = vec![F::zero(); aux_len()];
        zw[0] = a - inv2;
        zw[1] = a.square() - inv2;
        aux_weights.push(zw);
        aux_sources.push(Source {
            root: step.mask_root.clone(),
            size: domain::<F>(aux_len())?.size(),
            columns: 1,
        });
        n /= 2;
        source = Source {
            root: step.switched_root.clone(),
            size: domain::<F>(n)?.size(),
            columns: 2,
        };
        alpha = Some(a);
    }
    let mut sources = vec![source];
    sources.extend(aux_sources);
    let mut weights = vec![form.dense()];
    weights.extend(aux_weights);
    for root in &proof.mask_roots {
        if root.len() != 32 {
            return Ok(false);
        }
        t.absorb(root);
    }
    t.absorb_field(proof.mask_target);
    let gamma = nonzero::<F>(t);
    let mut response_target = F::zero();
    for (i, response) in proof.responses.iter().enumerate() {
        if response.message.len() != weights[i].len()
            || response.randomness.len() != num_queries()
            || response.source_queries.len() != num_queries()
            || response.mask_queries.len() != num_queries()
        {
            return Ok(false);
        }
        absorb(t, &response.message);
        absorb(t, &response.randomness);
        response_target += dot(&response.message, &weights[i]);
    }
    if response_target != proof.mask_target + gamma * target {
        return Ok(false);
    }
    for (i, response) in proof.responses.iter().enumerate() {
        let s = &sources[i];
        let g = Source {
            root: proof.mask_roots[i].clone(),
            size: s.size,
            columns: 1,
        };
        let encoded = encode(&response.message, &response.randomness)?;
        let indices = t.squeeze_indices(s.size, num_queries());
        for ((&j, sq), gq) in indices
            .iter()
            .zip(&response.source_queries)
            .zip(&response.mask_queries)
        {
            if !check(s, sq, j) || !check(&g, gq, j) {
                return Ok(false);
            }
            if encoded[j] != gq.values[0] + gamma * value(sq, if i == 0 { alpha } else { None }) {
                return Ok(false);
            }
        }
    }
    Ok(true)
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
    fn tensor_folding_matches_dense_linear_form() {
        let mut form = LinearForm::<F>::new(5);
        form.unit(7, F::from(9));
        form.power(F::from(3), F::from(11));
        let expected = fold(&form.dense(), F::from(8))
            .into_iter()
            .map(|v| F::from(2) * v)
            .collect::<Vec<_>>();
        form.folded(F::from(8), F::from(2));
        assert_eq!(form.dense(), expected);
    }

    #[test]
    fn base_and_recursive_relations_bind_all_messages() {
        for n in [16usize, 1024] {
            let mut rng = rng();
            let message: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
            let mut form = LinearForm::new(n.ilog2() as usize);
            form.unit(3, F::from(9));
            form.power(F::from(7), F::one());
            let target = dot(&message, &form.dense());
            let (root, witness) = commit(message, &mut rng).unwrap();
            let mut tp = Transcript::new(b"test");
            let mut proof = prove(witness, form.clone(), target, &mut tp, &mut rng).unwrap();
            let mut tv = Transcript::new(b"test");
            assert!(verify(&root, form.clone(), target, &proof, &mut tv).unwrap());
            assert_eq!(tp.squeeze_field::<F>(), tv.squeeze_field::<F>());
            let check = |p: &RelationProof<F>| {
                verify(
                    &root,
                    form.clone(),
                    target,
                    p,
                    &mut Transcript::new(b"test"),
                )
                .unwrap()
            };
            assert!(
                !verify(
                    &root,
                    form.clone(),
                    target + F::one(),
                    &proof,
                    &mut Transcript::new(b"test")
                )
                .unwrap()
            );
            proof.responses[0].message[0] += F::one();
            assert!(!check(&proof));
            proof.responses[0].message[0] -= F::one();
            proof.responses[0].source_queries[0].values[0] += F::one();
            assert!(!check(&proof));
            proof.responses[0].source_queries[0].values[0] -= F::one();
            proof.responses[0].mask_queries[0].path.leaf_index ^= 1;
            assert!(!check(&proof));
            proof.responses[0].mask_queries[0].path.leaf_index ^= 1;
            if n > aux_len() {
                assert_eq!(proof.steps.len(), 2);
                proof.steps[0].ood += F::one();
                assert!(!check(&proof));
                proof.steps[0].ood -= F::one();
                proof.steps[1].polynomial[2] += F::one();
                assert!(!check(&proof));
                proof.steps[1].polynomial[2] -= F::one();
                proof.steps[0].switched_root[0] ^= 1;
                assert!(!check(&proof));
                proof.steps[0].switched_root[0] ^= 1;
                proof.steps.pop();
                assert!(!check(&proof));
            } else {
                proof.responses[0].randomness.pop();
                assert!(!check(&proof));
            }
        }
    }

    #[test]
    fn rs_queries_can_be_identical_for_any_two_messages() {
        // Explicit affine bijection of randomness: interpolate a tail shift
        // hiding an arbitrary message change at the full query budget.
        let n = 16;
        let q = num_queries();
        let d = domain::<F>(n).unwrap();
        let mut rng = rng();
        let a: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let b: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let r: Vec<_> = (0..q).map(|_| F::rand(&mut rng)).collect();
        let delta: Vec<_> = a.iter().zip(&b).map(|(&a, &b)| a - b).collect();
        let xs: Vec<_> = (0..q).map(|i| d.element(i)).collect();
        let ys: Vec<_> = xs
            .iter()
            .map(|&x| dot(&delta, &powers(x, n)) * x.pow([n as u64]).inverse().unwrap())
            .collect();
        let mut shift = vec![F::zero(); q];
        for i in 0..q {
            let mut basis = vec![F::one()];
            let mut denominator = F::one();
            for j in 0..q {
                if i != j {
                    let mut next = vec![F::zero(); basis.len() + 1];
                    for (k, &c) in basis.iter().enumerate() {
                        next[k] -= xs[j] * c;
                        next[k + 1] += c;
                    }
                    basis = next;
                    denominator *= xs[i] - xs[j];
                }
            }
            let scale = ys[i] * denominator.inverse().unwrap();
            for (v, c) in shift.iter_mut().zip(basis) {
                *v += scale * c;
            }
        }
        let rb: Vec<_> = r.iter().zip(shift).map(|(&r, s)| r + s).collect();
        let ea = encode(&a, &r).unwrap();
        let eb = encode(&b, &rb).unwrap();
        assert_eq!(&ea[..q], &eb[..q]);
        assert_ne!(ea[q], eb[q]);
    }

    #[test]
    fn masked_response_hides_the_message_given_its_linear_target() {
        let mut rng = rng();
        let n = 16;
        let a: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let mut b = a.clone();
        b[0] += F::one();
        b[1] -= F::one();
        let weights = vec![F::one(); n];
        assert_eq!(dot(&a, &weights), dot(&b, &weights));
        let mask: Vec<_> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let gamma = F::from(13);
        let shifted: Vec<_> = mask
            .iter()
            .zip(a.iter().zip(&b))
            .map(|(&g, (&a, &b))| g + gamma * (a - b))
            .collect();
        assert_eq!(dot(&mask, &weights), dot(&shifted, &weights));
        for i in 0..n {
            assert_eq!(mask[i] + gamma * a[i], shifted[i] + gamma * b[i]);
        }
    }
}
