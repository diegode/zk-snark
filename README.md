# zk-snark

A Rust implementation of a simple zero-knowledge SNARK for arithmetic circuits.

> **Academic use only.** This code was built as a teaching tool companion to a book chapter on probabilistic proof systems and zero knowledge. It has not been audited for security and should not be used in production.

## Construction

The overall argument system follows [Spartan](https://eprint.iacr.org/2019/550).
We depart from the original in that the succinct matrix evaluations use a direct sumcheck over the nonzero set rather than Spark. This is simpler but costs an extra logarithmic factor.

The polynomial commitment scheme follows [BaseFold](https://eprint.iacr.org/2023/1705),
which we simplify by folding directly along the evaluation point instead of running its
interleaved sumcheck. This is sound, as the evaluation point is the sumcheck's random challenge, but confines it to the unique-decoding regime, implying
a larger *t* (as defined below).

The zero-knowledge layer follows [Libra](https://eprint.iacr.org/2019/317)'s additive
sumcheck masks, with two departures:
1. Libra's standalone constant-term mask is dropped: the sumcheck consistency check
already pins that coefficient.
2. Our hash-based evaluation proofs open raw codeword symbols, which Libra's
pairing-based commitments never do, so every committed table carries enough uniform
randomness to make the opened symbols uniform too (via an [Aurora](https://eprint.iacr.org/2018/828)-style randomized encoding).

## Complexity

Let

- *N* — the number of constraints
- *λ = 128* — the security parameter
- *ρ = 1/8* — the Reed–Solomon rate
- *t = ⌈λ / log₂(2/(1+ρ))⌉ = 155* - queries per evaluation proof

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
2. Run a one-time `setup(circuit, zk)` to obtain the prover and verifier parameters
   `(pp, vp)`. Only the circuit's structure is read,
   so leave both the public-input and witness
   fields empty. When the circuit layout depends on data, pass that as a
   separate field (e.g. the clue positions in the Sudoku example). Pass
   `zk: true` if you intend to produce zero-knowledge proofs.
3. Call `prove(&pp, circuit, zk, rng)` with the same circuit but the
   public-input and witness fields filled in. Pass `zk: false` for a non-hiding proof.
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
