# zk-snark

A Rust implementation of a zero-knowledge SNARK for arithmetic circuits.

> **Academic use only.** This code was built as a teaching tool companion to a book chapter on probabilistic proof systems and zero knowledge. It has not been audited for security and should not be used in production.

## Book chapter construction

The chapter follows [Spartan](https://eprint.iacr.org/2019/550), replacing Spark
with a direct sumcheck over the matrix nonzeros. This is simpler but costs an
extra logarithmic factor.

Its polynomial commitment scheme specializes [BaseFold](https://eprint.iacr.org/2023/1705)
to Reed–Solomon codes, interleaving FRI folding with sumcheck to prove
multilinear evaluations.

Zero knowledge adapts [Libra](https://eprint.iacr.org/2019/317)'s additive sumcheck masks,
replacing pairing-based commitments with random table padding and salted
Merkle commitments.
Terminal checks adapt [Chiesa–Fenzi–Weissenberg](https://eprint.iacr.org/2026/391),
replacing separate mask oracles and code switching with one BaseFold opening
for the witness and masks.

## Complexity

Let

- *N* — a bound on constraints, variables, and total matrix nonzeros
- *λ = 128* — the target in bits for each opening's proximity-query term
- *ρ = 1/8* — the Reed–Solomon rate bound
- *t = ⌈λ / log₂(3/(2+ρ))⌉ = 258* — queries per opening

With security parameters and the number of public inputs fixed, the asymptotic
bounds in N remain as below.

| Phase    | Complexity |
|----------|------------|
| `setup`  | O( (1/ρ)·N·log² N ) |
| `prove`  | O( (1/ρ)·N·log² N ) |
| `verify` | O( t·log³ N ) |
| proof size | O( t·log³ N ) |

## Running the Tests

```bash
cargo test --release --all-targets -- --test-threads=1
```

## Usage steps

1. Implement `ConstraintSynthesizer<F>` from `ark-relations`.
2. Run a one-time `setup(circuit, zk)` and handle its `Result` to obtain the
   prover and verifier parameters `(pp, vp)`. Only the circuit's structure is read,
   so leave both the public-input and witness fields empty. When the circuit
   layout depends on data, pass that as a separate field (e.g. the clue positions
   in the Sudoku example). Pass
   `zk: true` for zero-knowledge proofs or `zk: false` for non-hiding proofs.
3. Call `prove(&pp, circuit, rng)` with the same circuit but the
   public-input and witness fields filled in and a cryptographic RNG.
4. Build the `public_inputs` vector and call `verify(&vp, &public_inputs, &proof)`.

## Examples

| Example | What it proves |
|---------|---------------|
| `sudoku_solution` | Knowledge of a valid Sudoku solution without revealing it |
| `merkle_tree_inclusion` | Membership of a secret leaf in a Poseidon Merkle tree, given only the root |

```bash
cargo run --release --example sudoku_solution
cargo run --release --example merkle_tree_inclusion
```
