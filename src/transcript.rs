//! Fiat--Shamir challenges from a domain-separated, length-delimited transcript.
//! Chapter: Section 6 (Fiat–Shamir), ρ_i = H(x, α_1, …, α_i) over all prior messages.
//! The transcript retains every absorbed message. For ZK proofs, the freshly
//! salted private commitment root supplies randomness before the first challenge.
use ark_ff::{BigInteger, PrimeField};
use sha2::{Digest, Sha256};

pub struct Transcript {
    hasher: Sha256,
    round: u64,
}

impl Transcript {
    /// Prover and verifier derive the same challenges from the same messages.
    pub fn new(label: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"zk-snark/transcript/v3");
        let mut t = Self { hasher, round: 0 };
        t.absorb(label);
        t
    }

    pub fn absorb(&mut self, data: &[u8]) {
        self.hasher.update([0]);
        self.hasher.update((data.len() as u64).to_le_bytes());
        self.hasher.update(data);
    }

    pub fn absorb_field<F: PrimeField>(&mut self, v: F) {
        let mut buf = Vec::new();
        v.serialize_compressed(&mut buf).unwrap();
        self.absorb(&buf);
    }

    fn begin_challenge(&mut self, kind: &[u8]) {
        self.hasher.update([1]);
        self.hasher.update(self.round.to_le_bytes());
        self.absorb(kind);
        self.round += 1;
    }

    fn block(&self, counter: &mut u64) -> [u8; 32] {
        let mut h = self.hasher.clone();
        h.update([2]);
        h.update(counter.to_le_bytes());
        *counter += 1;
        h.finalize().into()
    }

    /// Rejection sampling gives independent uniform field elements, including
    /// zero, rather than a biased reduction modulo the field characteristic.
    pub fn squeeze_field<F: PrimeField>(&mut self) -> F {
        self.absorb(&F::MODULUS.to_bytes_le());
        self.begin_challenge(b"field");
        let bytes_len = (F::MODULUS_BIT_SIZE as usize).div_ceil(8);
        let mut counter = 0;
        loop {
            let mut bytes = Vec::with_capacity(bytes_len + 31);
            while bytes.len() < bytes_len {
                bytes.extend_from_slice(&self.block(&mut counter));
            }
            bytes.truncate(bytes_len);
            if let Some(value) = F::from_random_bytes(&bytes) {
                self.absorb_field(value);
                return value;
            }
        }
    }

    pub fn squeeze_indices(&mut self, n: usize, count: usize) -> Vec<usize> {
        assert!(n > 0);
        self.absorb(&(n as u64).to_le_bytes());
        self.absorb(&(count as u64).to_le_bytes());
        self.begin_challenge(b"indices");
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        let mut counter = 0;
        let mut indices = Vec::with_capacity(count);
        while indices.len() < count {
            let hash = self.block(&mut counter);
            let value = u64::from_le_bytes(hash[..8].try_into().unwrap());
            if value >= threshold {
                indices.push((value % n) as usize);
            }
        }
        for &index in &indices {
            self.absorb(&(index as u64).to_le_bytes());
        }
        indices
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

    #[test]
    fn commitment_root_binds_all_challenges() {
        let replay = |root: &[u8]| {
            let mut t = Transcript::new(b"root-binding");
            t.absorb(root);
            t.absorb(b"first message");
            let a: Fr = t.squeeze_field();
            t.absorb(b"second message");
            let b: Fr = t.squeeze_field();
            let indices = t.squeeze_indices(17, 100);
            (a, b, indices)
        };
        let root = [7; 32];
        let original = replay(&root);
        assert_eq!(replay(&root), original);
        let mut other_root = root;
        other_root[0] ^= 1;
        let changed = replay(&other_root);
        assert_ne!(original.0, changed.0);
        assert_ne!(original.1, changed.1);
        assert_ne!(original.2, changed.2);
    }

    #[test]
    fn labels_and_challenge_types_have_separate_domains() {
        let mut a = Transcript::new(b"a");
        let mut b = Transcript::new(b"b");
        assert_ne!(a.squeeze_field::<Fr>(), b.squeeze_field::<Fr>());
        let mut a = Transcript::new(b"a");
        let mut b = Transcript::new(b"a");
        a.squeeze_field::<Fr>();
        b.squeeze_indices(17, 1);
        assert_ne!(a.squeeze_field::<Fr>(), b.squeeze_field::<Fr>());
    }
}
