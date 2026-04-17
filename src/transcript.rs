//! Fiat--Shamir transcript.
//!
//! Prover and verifier absorb the same messages in the same order, reproducing
//! the public-coin challenges non-interactively.
use ark_ff::PrimeField;

use sha2::{Digest, Sha256};

pub struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(label);
        Self { hasher }
    }

    /// Absorb length-prefixed bytes, preventing ambiguity between message splits.
    pub fn absorb(&mut self, data: &[u8]) {
        self.hasher.update(&(data.len() as u64).to_le_bytes());
        self.hasher.update(data);
    }

    /// Absorb a field element.
    pub fn absorb_field<F: PrimeField>(&mut self, v: F) {
        let mut buf = Vec::new();
        v.serialize_compressed(&mut buf).unwrap();
        self.absorb(&buf);
    }

    /// Derive a field challenge from two chained SHA-256 digests.
    pub fn squeeze_field<F: PrimeField>(&mut self) -> F {
        let h1 = self.hasher.clone().finalize();
        self.hasher.update(&h1);
        let h2 = self.hasher.clone().finalize();
        self.hasher.update(&h2);
        let mut bytes = h1.to_vec();
        bytes.extend_from_slice(&h2);
        F::from_le_bytes_mod_order(&bytes)
    }

    /// Squeeze `count` indices in `0..n`, used to sample Merkle leaf positions.
    pub fn squeeze_indices(&mut self, n: usize, count: usize) -> Vec<usize> {
        (0..count)
            .map(|_| {
                let hash = self.hasher.clone().finalize();
                self.hasher.update(&hash);
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&hash[..8]);
                usize::from_le_bytes(bytes) % n
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn test_transcript_deterministic() {
        let mut t1 = Transcript::new(b"test_label");
        let mut t2 = Transcript::new(b"test_label");

        t1.absorb(b"some data");
        t2.absorb(b"some data");

        let f_absorb = Fr::from(42u64);
        t1.absorb_field(f_absorb);
        t2.absorb_field(f_absorb);

        let f1: Fr = t1.squeeze_field();
        let f2: Fr = t2.squeeze_field();
        assert_eq!(f1, f2);

        let i1 = t1.squeeze_indices(100, 5);
        let i2 = t2.squeeze_indices(100, 5);
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_domain_separation() {
        let mut t1 = Transcript::new(b"test_label");
        let mut t2 = Transcript::new(b"test_label");

        t1.absorb(b"hello");
        t1.absorb(b"world");

        t2.absorb(b"helloworld");

        let f1: Fr = t1.squeeze_field();
        let f2: Fr = t2.squeeze_field();
        assert_ne!(f1, f2);
    }

    #[test]
    fn test_squeeze_indices() {
        let mut t = Transcript::new(b"test_label");
        let n = 42;
        let count = 100;
        let indices = t.squeeze_indices(n, count);

        assert_eq!(indices.len(), count);
        for &index in &indices {
            assert!(index < n, "Index {} is not less than {}", index, n);
        }
    }
}
