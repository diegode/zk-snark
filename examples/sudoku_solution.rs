//! ZK proof that a 9×9 Sudoku puzzle has a known solution, without revealing it.
//!
//! Public  : the non-zero clue cells (the 30 givens for the puzzle below).
//! Private : the full 81-cell solution.
//!
//! Constraints
//! -----------
//!   1. Range  — each solution cell ∈ {1,…,9} via ∏_{k=1}^{9}(cell − k) = 0,
//!      built as a chain of degree-2 (R1CS) multiplications.
//!   2. Clues  — solution matches the given puzzle digits.
//!   3. Unique — every row, column, and 3×3 box has pairwise-distinct cells
//!      (non-zero-inverse trick: (a − b) · inv = 1).

use ark_ff::{Field, One};
use ark_relations::gr1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable};
use ark_bls12_381::Fr as F;
use rand::{rngs::StdRng, SeedableRng};
use zk_snark::snark::{prove, setup, verify};

const N: usize = 81;

fn sudoku_groups() -> Vec<[usize; 9]> {
    let mut groups = Vec::with_capacity(27);
    for r in 0..9 {
        let mut g = [0usize; 9];
        for c in 0..9 {
            g[c] = r * 9 + c;
        }
        groups.push(g);
    }
    for c in 0..9 {
        let mut g = [0usize; 9];
        for r in 0..9 {
            g[r] = r * 9 + c;
        }
        groups.push(g);
    }
    for br in 0..3 {
        for bc in 0..3 {
            let mut g = [0usize; 9];
            let mut idx = 0;
            for i in 0..3 {
                for j in 0..3 {
                    g[idx] = (br * 3 + i) * 9 + (bc * 3 + j);
                    idx += 1;
                }
            }
            groups.push(g);
        }
    }
    groups
}

struct SudokuCircuit {
    /// Which of the 81 cells are given. This is the structural part of the
    /// puzzle: it sets how many public input variables exist and which solution
    /// cells get pinned, so it is required at both setup and proving.
    given: [bool; N],
    /// The digit (1–9) in each given cell — the public values. `None` at setup
    /// (only the structure is read); `Some(_)` when proving.
    clues: Option<[u8; N]>,
    /// Full 9×9 solution — the private witness. `None` at setup, `Some(_)` when
    /// proving.
    solution: Option<[u8; N]>,
}

impl ConstraintSynthesizer<F> for SudokuCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let sol_vals: Vec<F> = (0..N)
            .map(|i| self.solution.map(|s| F::from(s[i] as u64)).unwrap_or_default())
            .collect();

        let mut clue_vars: [Option<Variable>; N] = [None; N];
        for i in 0..N {
            if self.given[i] {
                let val = self.clues.map(|c| F::from(c[i] as u64)).unwrap_or_default();
                clue_vars[i] = Some(cs.new_input_variable(|| Ok(val))?);
            }
        }

        // Enforce ∏_{k=1}^{9} (sol − k) = 0, which holds iff sol is one of the
        // nine digits. The product is accumulated with degree-2 multiplications,
        // each introducing one intermediate witness; the last factor is folded
        // into a constraint whose output wire is fixed to 0.
        let mut sol_vars = Vec::with_capacity(N);
        for i in 0..N {
            let v = sol_vals[i];
            let sol = cs.new_witness_variable(|| Ok(v))?;

            let factor_lc = |k: u64| ark_relations::lc!() + sol + (-F::from(k), Variable::One);
            let factor_val = |k: u64| v - F::from(k);

            let mut acc_val = factor_val(1) * factor_val(2);
            let mut acc_var = cs.new_witness_variable(|| Ok(acc_val))?;
            cs.enforce_r1cs_constraint(
                || factor_lc(1),
                || factor_lc(2),
                || ark_relations::lc!() + acc_var,
            )?;

            for k in 3..=9u64 {
                if k < 9 {
                    let next_val = acc_val * factor_val(k);
                    let next_var = cs.new_witness_variable(|| Ok(next_val))?;
                    cs.enforce_r1cs_constraint(
                        || ark_relations::lc!() + acc_var,
                        || factor_lc(k),
                        || ark_relations::lc!() + next_var,
                    )?;
                    acc_val = next_val;
                    acc_var = next_var;
                } else {
                    cs.enforce_r1cs_constraint(
                        || ark_relations::lc!() + acc_var,
                        || factor_lc(k),
                        || ark_relations::lc!(),
                    )?;
                }
            }

            sol_vars.push(sol);
        }

        for i in 0..N {
            if self.given[i] {
                let clue_var = clue_vars[i].unwrap();
                cs.enforce_r1cs_constraint(
                    || ark_relations::lc!() + sol_vars[i] - clue_var,
                    || ark_relations::lc!() + Variable::One,
                    || ark_relations::lc!(),
                )?;
            }
        }

        // For each pair (a, b) in the group: (sol[a] − sol[b]) · inv = 1.
        for group in sudoku_groups() {
            for gi in 0..9 {
                for gj in (gi + 1)..9 {
                    let a = group[gi];
                    let b = group[gj];
                    let diff = sol_vals[a] - sol_vals[b];
                    let inv_val = diff.inverse().unwrap_or_default();
                    let inv = cs.new_witness_variable(|| Ok(inv_val))?;
                    cs.enforce_r1cs_constraint(
                        || ark_relations::lc!() + sol_vars[a] - sol_vars[b],
                        || ark_relations::lc!() + inv,
                        || ark_relations::lc!() + Variable::One,
                    )?;
                }
            }
        }

        Ok(())
    }
}

fn main() {
    #[rustfmt::skip]
    let clues: [u8; N] = [
        5, 3, 0,  0, 7, 0,  0, 0, 0,
        6, 0, 0,  1, 9, 5,  0, 0, 0,
        0, 9, 8,  0, 0, 0,  0, 6, 0,
        8, 0, 0,  0, 6, 0,  0, 0, 3,
        4, 0, 0,  8, 0, 3,  0, 0, 1,
        7, 0, 0,  0, 2, 0,  0, 0, 6,
        0, 6, 0,  0, 0, 0,  2, 8, 0,
        0, 0, 0,  4, 1, 9,  0, 0, 5,
        0, 0, 0,  0, 8, 0,  0, 7, 9,
    ];
    #[rustfmt::skip]
    let solution: [u8; N] = [
        5, 3, 4,  6, 7, 8,  9, 1, 2,
        6, 7, 2,  1, 9, 5,  3, 4, 8,
        1, 9, 8,  3, 4, 2,  5, 6, 7,
        8, 5, 9,  7, 6, 1,  4, 2, 3,
        4, 2, 6,  8, 5, 3,  7, 9, 1,
        7, 1, 3,  9, 2, 4,  8, 5, 6,
        9, 6, 1,  5, 3, 7,  2, 8, 4,
        2, 8, 7,  4, 1, 9,  6, 3, 5,
        3, 4, 5,  2, 8, 6,  1, 7, 9,
    ];

    let given: [bool; N] = clues.map(|c| c != 0);

    let mut rng = StdRng::seed_from_u64(0);

    let (pp, vp) = setup::<F, _>(SudokuCircuit { given, clues: None, solution: None }, true);

    println!("Proving knowledge of 9x9 Sudoku solution (ZK mode)…");
    let proof = prove(
        &pp,
        SudokuCircuit { given, clues: Some(clues), solution: Some(solution) },
        true,
        &mut rng,
    ).expect("failed to prove");

    let public_inputs = |clues: [u8; N]| -> Vec<F> {
        std::iter::once(F::one())
            .chain(clues.iter().filter(|&&c| c != 0).map(|&c| F::from(c as u64)))
            .collect()
    };
    let public = public_inputs(clues);
    let ok = verify(&vp, &public, &proof).expect("failed to verify");
    println!("Correct puzzle verified : {ok}");
    assert!(ok, "valid Sudoku proof was rejected");

    let mut wrong_clues = clues;
    wrong_clues[0] = 6;
    let public_wrong = public_inputs(wrong_clues);
    let rejected = !verify(&vp, &public_wrong, &proof).expect("failed to verify");
    println!("Wrong puzzle rejected   : {rejected}");
    assert!(rejected, "tampered puzzle was accepted");
}
