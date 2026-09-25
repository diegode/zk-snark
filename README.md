# zk-snark

A Rust implementation of a zero-knowledge SNARK for arithmetic circuits.

> **Academic use only.** This code was built as a teaching tool companion to a book chapter on probabilistic proof systems and zero knowledge. It has not been audited for security and should not be used in production.

## Book chapter construction

The chapter follows [Spartan](https://eprint.iacr.org/2019/550), replacing Spark
with a direct sumcheck over the matrix nonzeros. This is simpler but costs an
extra logarithmic factor.

Its polynomial commitment scheme simplifies [BaseFold](https://eprint.iacr.org/2023/1705),
using Reed–Solomon encodings, Merkle commitments, and folding along the evaluation
point. The zero-knowledge layer combines witness blinding with
[Libra](https://eprint.iacr.org/2019/317)-style additive sumcheck masks.
Fiat–Shamir makes the protocol non-interactive.

## Errata

The implementation includes the following corrections to the chapter:

1. **Commitment order.** Each folded root must be committed before the next
   folding challenge. Openings at an already known point use a fresh degree-two
   evaluation sumcheck.
2. **Sumcheck masking.** Mask all three outer terminal claims and include
   independent constants in the additive masks. The outer sumcheck has degree
   five; the inner sumcheck remains degree two.
3. **Witness openings.** Prove the final witness/mask identity as one private
   linear relation, without revealing the witness or inner-mask evaluation.
   The hiding opening follows [Chiesa–Fenzi–Weissenberg](https://eprint.iacr.org/2026/391),
   using randomized Reed–Solomon encodings, masked sumchecks, and masked responses.
   Each private encoding is opened once, with fresh randomness covering its query budget.

## Complexity

Let

- *N* — a bound on constraints, variables, and total matrix nonzeros
- *λ = 128* — the security target in bits
- *ρ = 1/8* — the Reed–Solomon rate bound
- *t = ⌈λ / log₂(2/(1+ρ))⌉ = 155* — queries per opening

With security parameters and the number of public inputs fixed, the asymptotic
bounds in N remain as below. Field and hash operations count at unit cost;
proof size is measured in field elements and hash digests.

| Phase    | Complexity |
|----------|------------|
| `setup`  | O( (1/ρ)·N·log² N ) |
| `prove`  | O( (1/ρ)·N·log² N ) |
| `verify` | O( t·log³ N ) |
| proof size | O( t·log³ N ) |

## Running the Tests

```bash
cargo test
```

## Usage steps

1. Implement `ConstraintSynthesizer<F>` from `ark-relations`.
2. Run a one-time `setup(circuit, zk)` and handle its `Result` to obtain the
   prover and verifier parameters `(pp, vp)`. Only canonical R1CS constraints are
   supported. Only the circuit's structure is read, so leave both the public-input
   and witness fields empty. When the circuit layout depends on data, pass that
   as a separate field (e.g. the clue positions in the Sudoku example). Pass
   `zk: true` if you intend to produce zero-knowledge proofs.
3. Call `prove(&pp, circuit, zk, rng)` with the same R1CS circuit but the
   public-input and witness fields filled in and a cryptographic RNG. Pass
   `zk: false` for a non-hiding proof.
4. Build the `public_inputs` vector and call `verify(&vp, &public_inputs, &proof)`.

## Examples

| Example | What it proves |
|---------|---------------|
| `sudoku_solution` | Knowledge of a valid Sudoku solution without revealing it |
| `merkle_tree_inclusion` | Membership of a secret leaf in a Poseidon Merkle tree, given only the root |
| `eddsa_verification` | Knowledge of a Schnorr signature under a publicly committed key, revealing neither the key nor the signature |

```bash
cargo run --release --example sudoku_solution
cargo run --release --example merkle_tree_inclusion
cargo run --release --example eddsa_verification
```
