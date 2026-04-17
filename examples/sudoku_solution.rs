//! ZK proof that a 4×4 Sudoku puzzle has a known solution, without revealing it.
//!
//! Public  : the non-zero clue cells (8 field elements for the puzzle below).
//! Private : the full 16-cell solution.
//!
//! Constraints
//! -----------
//!   1. Range  — each solution cell ∈ {1,2,3,4} via 2-bit decomposition.
//!   2. Clues  — solution matches the given puzzle digits.
//!   3. Unique — every row, column, and 2×2 box has pairwise-distinct cells
//!               (non-zero-inverse trick: (a − b) · inv = 1).
//!
//! Puzzle              Solution
//!   1  2  .  .          1  2  3  4
//!   .  .  2  1    →     3  4  2  1
//!   2  1  .  .          2  1  4  3
//!   .  .  1  2          4  3  1  2

use ark_ff::{Field, PrimeField, Zero};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable};
use ark_bls12_381::Fr as F;
use rand::{rngs::StdRng, SeedableRng};
use zk_snark::snark::{prove, verify};

// ---------------------------------------------------------------------------
// Groups: rows, columns, 2×2 boxes — each is 4 cell indices into the flat grid
// ---------------------------------------------------------------------------

const GROUPS: [[usize; 4]; 12] = [
    [0, 1, 2, 3],   [4, 5, 6, 7],   [8, 9, 10, 11],  [12, 13, 14, 15], // rows
    [0, 4, 8, 12],  [1, 5, 9, 13],  [2, 6, 10, 14],  [3, 7, 11, 15],   // cols
    [0, 1, 4, 5],   [2, 3, 6, 7],   [8, 9, 12, 13],  [10, 11, 14, 15], // boxes
];

// ---------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------

struct SudokuCircuit {
    /// Puzzle clues: 0 = empty, 1–4 = given digit.
    /// Non-zero entries become public inputs.
    clues: [u8; 16],
    /// Full 4×4 solution — Some(_) for the prover, None for the verifier.
    solution: Option<[u8; 16]>,
}

impl ConstraintSynthesizer<F> for SudokuCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // Cached field values for all 16 solution cells.
        let sol_vals: Vec<F> = (0..16)
            .map(|i| self.solution.map(|s| F::from(s[i] as u64)).unwrap_or_default())
            .collect();

        // --- 1. Public clue variables (non-zero cells) -----------------------
        let mut clue_vars: [Option<Variable>; 16] = [None; 16];
        for (i, &c) in self.clues.iter().enumerate() {
            if c != 0 {
                let val = F::from(c as u64);
                clue_vars[i] = Some(cs.new_input_variable(|| Ok(val))?);
            }
        }

        // --- 2. Private solution cells + 2-bit range decomposition -----------
        // For sol ∈ {1,2,3,4}: encode as sol = b0 + 2·b1 + 1
        // so that b0,b1 ∈ {0,1} and sol ∈ {1,2,3,4} is fully covered.
        let mut sol_vars = Vec::with_capacity(16);
        for i in 0..16 {
            let v = sol_vals[i];
            let sol = cs.new_witness_variable(|| Ok(v))?;

            // Extract bit decomposition of (v − 1) ∈ {0,1,2,3}.
            let raw_minus1: u64 = if v == F::zero() {
                0 // verifier placeholder
            } else {
                v.into_bigint().as_ref()[0].saturating_sub(1)
            };
            let b0_val = F::from(raw_minus1 & 1);
            let b1_val = F::from((raw_minus1 >> 1) & 1);

            let b0 = cs.new_witness_variable(|| Ok(b0_val))?;
            let b1 = cs.new_witness_variable(|| Ok(b1_val))?;

            // b0 · (b0 − 1) = 0
            cs.enforce_constraint(
                ark_relations::lc!() + b0,
                ark_relations::lc!() + b0 - Variable::One,
                ark_relations::lc!(),
            )?;
            // b1 · (b1 − 1) = 0
            cs.enforce_constraint(
                ark_relations::lc!() + b1,
                ark_relations::lc!() + b1 - Variable::One,
                ark_relations::lc!(),
            )?;
            // (b0 + 2·b1 + 1) · 1 = sol
            cs.enforce_constraint(
                ark_relations::lc!() + b0 + (F::from(2u64), b1) + Variable::One,
                ark_relations::lc!() + Variable::One,
                ark_relations::lc!() + sol,
            )?;

            sol_vars.push(sol);
        }

        // --- 3. Clue consistency: (sol[i] − clue[i]) · 1 = 0 ----------------
        for (i, &c) in self.clues.iter().enumerate() {
            if c != 0 {
                let clue_var = clue_vars[i].unwrap();
                cs.enforce_constraint(
                    ark_relations::lc!() + sol_vars[i] - clue_var,
                    ark_relations::lc!() + Variable::One,
                    ark_relations::lc!(),
                )?;
            }
        }

        // --- 4. Pairwise non-zero difference in every group ------------------
        // For each pair (a, b) in the group: (sol[a] − sol[b]) · inv = 1.
        // For the verifier all sol values are 0, so diff = 0 and inv = 0;
        // the constraint becomes unsatisfied for that (placeholder) witness,
        // but the matrix structure is identical to the prover's — that is all
        // the SNARK verifier needs.
        for group in &GROUPS {
            for gi in 0..4 {
                for gj in (gi + 1)..4 {
                    let a = group[gi];
                    let b = group[gj];
                    let diff = sol_vals[a] - sol_vals[b];
                    let inv_val = diff.inverse().unwrap_or_default();
                    let inv = cs.new_witness_variable(|| Ok(inv_val))?;
                    cs.enforce_constraint(
                        ark_relations::lc!() + sol_vars[a] - sol_vars[b],
                        ark_relations::lc!() + inv,
                        ark_relations::lc!() + Variable::One,
                    )?;
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    // The public puzzle and the hidden solution.
    let clues: [u8; 16] = [1, 2, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 0, 1, 2];
    let solution: [u8; 16] = [1, 2, 3, 4, 3, 4, 2, 1, 2, 1, 4, 3, 4, 3, 1, 2];

    let mut rng = StdRng::seed_from_u64(0);

    println!("Proving knowledge of 4×4 Sudoku solution (ZK mode)…");
    let proof = prove(SudokuCircuit { clues, solution: Some(solution) }, true, &mut rng).expect("failed to prove");

    let ok = verify(SudokuCircuit { clues, solution: None }, &proof).expect("failed to verify");
    println!("Correct puzzle verified : {ok}");
    assert!(ok, "valid Sudoku proof was rejected");

    // A different puzzle should be rejected.
    let mut wrong_clues = clues;
    wrong_clues[0] = 2; // flip the first given cell
    let rejected = !verify(SudokuCircuit { clues: wrong_clues, solution: None }, &proof).expect("failed to verify");
    println!("Wrong puzzle rejected   : {rejected}");
    assert!(rejected, "tampered puzzle was accepted");
}
