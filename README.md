# zk-snark

A Rust implementation of a simple zero-knowledge SNARK for arithmetic circuits.

> **Academic use only.** This code was built mostly via vibe-coding as a teaching tool to accompany a book chapter on probabilistic proof systems and zero knowledge. It has not been audited for security and should not be used in production.

## Implementation Notes

Compared to the [reference WHIR implementation](https://github.com/WizardOfMenlo/whir),
our WHIR-based PCS is roughly equivalent to running that implementation with
`--sec UniqueDecoding --fold 1`.

## Requirements

- Rust 1.65 or later (edition 2021)
- Cargo

## Running the Tests

```bash
cargo test
```

## Usage

Implement `ConstraintSynthesizer<F>` from `ark-relations`, then call `prove` / `verify`. Pass `zk: false` for a non-hiding proof.

## Examples

| Example | What it proves |
|---------|---------------|
| `sudoku_solution` | Knowledge of a valid 4×4 Sudoku solution without revealing any cell |
| `merkle_tree_inclusion` | Membership of a secret leaf in a Poseidon Merkle tree, given only the root |
| `eddsa_verification` | Knowledge of a Schnorr (EdDSA-style) signature over Jubjub without revealing the key or signature |

```bash
cargo run --example sudoku_solution
cargo run --example merkle_tree_inclusion
cargo run --example eddsa_verification
```
