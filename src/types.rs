//! The reference STARK configuration: Goldilocks field with a degree-2
//! binomial extension, Blake3 hashing, and a FRI-based PCS.
//!
//! The generic protocol lives in [`crate::config`], [`crate::system`],
//! [`crate::prover`] and [`crate::verifier`]; this module only provides a
//! concrete, batteries-included instantiation.

use crate::config::StarkGenericConfig;
use crate::hiding_mmcs::SyncHidingMmcs;
use p3_blake3::Blake3;
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, TwoAdicField, extension::BinomialExtensionField};
use p3_fri::{FriParameters as InnerFriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::{Rng, RngExt};
use std::sync::{Arc, Mutex};

pub type Val = Goldilocks;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 2>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, Blake3, 32>>;
pub type Mmcs =
    MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

pub type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <Pcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <Pcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> = <Pcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <Pcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

/// The reference [`StarkGenericConfig`] implementation.
pub struct GoldilocksBlake3Config {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// Seed for fresh challengers: a domain-separation tag followed by a
    /// digest of all protocol parameters.
    challenger_seed: Vec<u8>,
    /// Largest log2 degree the PCS can commit to and open.
    max_log_degree: usize,
    /// Largest quotient degree the PCS can serve trace evaluations for
    /// (the FRI blowup factor).
    max_quotient_degree: usize,
}

impl GoldilocksBlake3Config {
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        let pcs = new_pcs(commitment_parameters, fri_parameters);
        // Seed the challenger with a protocol tag for domain separation,
        // followed by every protocol parameter. Binding the parameters into
        // the seed means transcripts produced under different parameters
        // never collide (see the transcript contract on
        // [`StarkGenericConfig::initialise_challenger`]).
        let mut challenger_seed = b"multi-stark/v0".to_vec();
        for parameter in [
            commitment_parameters.log_blowup,
            commitment_parameters.cap_height,
            fri_parameters.log_final_poly_len,
            fri_parameters.max_log_arity,
            fri_parameters.num_queries,
            fri_parameters.commit_proof_of_work_bits,
            fri_parameters.query_proof_of_work_bits,
        ] {
            let parameter = u64::try_from(parameter).expect("parameter exceeds u64");
            challenger_seed.extend_from_slice(&parameter.to_le_bytes());
        }
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
        }
    }
}

impl StarkGenericConfig for GoldilocksBlake3Config {
    type Pcs = Pcs;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Challenger {
        Challenger::from_hasher(self.challenger_seed.clone(), Blake3)
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }
}

/// A `Clone`-able handle to one shared random generator. Plonky3's hiding
/// MMCS and PCS each take an owned `R: Clone`; cloning a seeded generator
/// would give them identical streams, so every clone of this handle draws
/// from the same underlying generator instead.
#[derive(Debug)]
pub struct SharedRng<R>(Arc<Mutex<R>>);

impl<R> Clone for SharedRng<R> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<R: Rng> SharedRng<R> {
    pub fn new(rng: R) -> Self {
        Self(Arc::new(Mutex::new(rng)))
    }
}

impl<R: Rng> rand::TryRng for SharedRng<R> {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.0.lock().expect("rng poisoned").next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.0.lock().expect("rng poisoned").next_u64())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.0.lock().expect("rng poisoned").fill_bytes(dst);
        Ok(())
    }
}

impl<R: rand::CryptoRng> rand::TryCryptoRng for SharedRng<R> {}

/// Number of random codewords the hiding PCS appends to every commitment
/// (the value every Plonky3 zero-knowledge configuration uses).
pub const NUM_RANDOM_CODEWORDS: usize = 4;

/// Salt elements per Merkle leaf in the hiding MMCS (4 x 64 bits).
pub const SALT_ELEMS: usize = 4;

pub type HidingMmcs<R> = SyncHidingMmcs<
    Val,
    u8,
    SerializingHasher<Blake3>,
    Blake3CompressionFunction,
    R,
    2,
    32,
    SALT_ELEMS,
>;
pub type HidingExtMmcs<R> = ExtensionMmcs<Val, ExtVal, HidingMmcs<R>>;
pub type HidingPcs<R> = HidingFriPcs<Val, Dft, HidingMmcs<R>, HidingExtMmcs<R>, R>;

/// The zero-knowledge variant of [`GoldilocksBlake3Config`]: salted Merkle
/// leaves and Plonky3's `HidingFriPcs`, which interleaves every committed
/// trace with random rows, appends random columns, randomizes the quotient
/// chunks and adds a random FRI-batch polynomial. The prover and verifier
/// select the zero-knowledge protocol variant through
/// [`StarkGenericConfig::is_zk`].
///
/// `rng` seeds all blinding; use an OS-backed generator in production.
pub struct GoldilocksBlake3ZkConfig<R> {
    pcs: HidingPcs<SharedRng<R>>,
    challenger_seed: Vec<u8>,
    max_log_degree: usize,
    max_quotient_degree: usize,
    min_trace_height: usize,
    /// The same stream the PCS and MMCS draw from, for accumulator masks.
    mask_rng: SharedRng<R>,
}

impl<R: rand::CryptoRng + Send> GoldilocksBlake3ZkConfig<R> {
    /// `rng` must be a cryptographically secure generator. It is wrapped in a
    /// single [`SharedRng`] so the Merkle salts, the trace blinding and the
    /// FRI-batch polynomial all draw from one stream; giving the hiding MMCS
    /// and PCS independent clones of a seeded generator would make their
    /// draws identical and publish the blinding values in the opened salts.
    pub fn new(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        rng: R,
    ) -> Self {
        let shared = SharedRng::new(rng);
        let pcs = new_hiding_pcs(commitment_parameters, fri_parameters, shared.clone());
        // A distinct tag: zero-knowledge transcripts must never collide with
        // plain ones even under identical parameters.
        let mut challenger_seed = b"multi-stark/v0-zk".to_vec();
        for parameter in [
            commitment_parameters.log_blowup,
            commitment_parameters.cap_height,
            fri_parameters.log_final_poly_len,
            fri_parameters.max_log_arity,
            fri_parameters.num_queries,
            fri_parameters.commit_proof_of_work_bits,
            fri_parameters.query_proof_of_work_bits,
        ] {
            let parameter = u64::try_from(parameter).expect("parameter exceeds u64");
            challenger_seed.extend_from_slice(&parameter.to_le_bytes());
        }
        // The verifier adds the zero-knowledge doubling to the claimed degree
        // itself, so the bound is the plain one.
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        // `num_queries` FRI openings plus ζ and ζ·g per column.
        let min_trace_height = (fri_parameters.num_queries + 2).next_power_of_two();
        Self {
            pcs,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
            min_trace_height,
            mask_rng: shared,
        }
    }
}

impl<R: Rng + Send> StarkGenericConfig for GoldilocksBlake3ZkConfig<R> {
    type Pcs = HidingPcs<SharedRng<R>>;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &HidingPcs<SharedRng<R>> {
        &self.pcs
    }

    fn min_trace_height(&self) -> usize {
        self.min_trace_height
    }

    fn sample_mask(&self, n: usize) -> Vec<Val> {
        let mut rng = self.mask_rng.clone();
        (0..n).map(|_| rng.random()).collect()
    }

    fn initialise_challenger(&self) -> Challenger {
        Challenger::from_hasher(self.challenger_seed.clone(), Blake3)
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }
}

/// Parameters of the polynomial commitment: Reed-Solomon rate and Merkle
/// tree shape.
#[derive(Clone, Copy)]
pub struct CommitmentParameters {
    pub log_blowup: usize,
    /// Height of the Merkle cap (number of top layers included in the commitment).
    /// A cap height of 0 means only the root is committed.
    pub cap_height: usize,
}

/// Parameters controlling the FRI protocol.
///
/// These parameters determine the concrete security level. The FRI soundness
/// error is approximately `ρ^num_queries` (conjectured; `√ρ^num_queries`
/// proven) where `ρ = 2^(-log_blowup)` (set in [`CommitmentParameters`]).
/// See the verifier module docs for the full soundness argument.
#[derive(Clone, Copy)]
pub struct FriParameters {
    /// Log2 of the degree of the final polynomial (0 means a constant).
    pub log_final_poly_len: usize,
    /// Maximum folding arity per FRI round (log2). A value of 1 means binary folding.
    pub max_log_arity: usize,
    /// Number of query repetitions for soundness amplification.
    pub num_queries: usize,
    /// Number of bits for the PoW phase before sampling _each_ batching challenge.
    pub commit_proof_of_work_bits: usize,
    /// Number of bits for the PoW phase before sampling the queries.
    pub query_proof_of_work_bits: usize,
}

type Blake3CompressionFunction = CompressionFunctionFromHasher<Blake3, 2, 32>;
type Dft = Radix2DitParallel<Val>;

fn new_mmcs(cap_height: usize) -> Mmcs {
    let byte_hash = Blake3;
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = Blake3CompressionFunction::new(byte_hash);
    Mmcs::new(field_hash, compress, cap_height)
}

fn new_pcs(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Pcs {
    let val_mmcs = new_mmcs(commitment_parameters.cap_height);
    let mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let inner_parameters = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        max_log_arity: fri_parameters.max_log_arity,
        num_queries: fri_parameters.num_queries,
        commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
        mmcs,
    };
    let dft = Dft::default();
    Pcs::new(dft, val_mmcs, inner_parameters)
}

fn new_hiding_pcs<R: Rng + Clone>(
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    rng: R,
) -> HidingPcs<R> {
    let byte_hash = Blake3;
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = Blake3CompressionFunction::new(byte_hash);
    let val_mmcs = HidingMmcs::new(
        field_hash,
        compress,
        commitment_parameters.cap_height,
        rng.clone(),
    );
    let mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let inner_parameters = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        max_log_arity: fri_parameters.max_log_arity,
        num_queries: fri_parameters.num_queries,
        commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
        mmcs,
    };
    HidingPcs::new(
        Dft::default(),
        val_mmcs,
        inner_parameters,
        NUM_RANDOM_CODEWORDS,
        rng,
    )
}

/// Zero-knowledge configuration for tests, with a deterministic seed.
#[cfg(test)]
pub(crate) type ZkTestConfig = GoldilocksBlake3ZkConfig<rand::rngs::StdRng>;

#[cfg(test)]
pub(crate) fn zk_test_config(log_blowup: usize, num_queries: usize, seed: u64) -> ZkTestConfig {
    use rand::SeedableRng;
    GoldilocksBlake3ZkConfig::new(
        CommitmentParameters {
            log_blowup,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
        rand::rngs::StdRng::seed_from_u64(seed),
    )
}

#[cfg(test)]
mod zk_config_tests {
    /// Aiur shares the system across rayon threads, so the zero-knowledge
    /// configuration must be `Send + Sync`.
    #[test]
    fn zk_config_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::ZkTestConfig>();
    }
}

#[cfg(test)]
mod pcs_ref_gen {
    use super::*;
    use p3_commit::Mmcs as _;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

    fn limbs(d: [u8; 32]) -> [u64; 4] {
        core::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap()))
    }
    fn dig(xs: [u64; 4]) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..4 {
            o[i * 8..i * 8 + 8].copy_from_slice(&xs[i].to_le_bytes());
        }
        o
    }

    /// Generates the Blake3 reference values asserted by `Ix/MultiStark/Tests.lean`
    /// (`pcs_hash_test`, `pcs_merkle_test`). Run with `--nocapture` and copy.
    #[test]
    fn gen_pcs_refs() {
        let f = Val::from_u32;
        let fh = SerializingHasher::new(Blake3);
        for n in [3u32, 17, 22, 20] {
            let row: Vec<Val> = (1..=n).map(f).collect();
            println!("LEAF{} {:?}", n, limbs(fh.hash_iter(row)));
        }
        let comp = Blake3CompressionFunction::new(Blake3);
        println!(
            "COMPRESS {:?}",
            limbs(comp.compress([dig([1, 2, 3, 4]), dig([5, 6, 7, 8])]))
        );

        // Merkle tree: matrices of heights 8/4/2 and widths 2/3/1, opened at index 5.
        let mut m0 = vec![f(0); 16];
        m0[10] = f(11);
        m0[11] = f(12); // row 5 = [11, 12]
        let mut m1 = vec![f(0); 12];
        m1[6] = f(107);
        m1[7] = f(108);
        m1[8] = f(109); // row 2 = [107, 108, 109]
        let mut m2 = vec![f(0); 2];
        m2[1] = f(202); // row 1 = [202]
        let mmcs = new_mmcs(0);
        let (commit, pd) = mmcs.commit(vec![
            RowMajorMatrix::new(m0, 2),
            RowMajorMatrix::new(m1, 3),
            RowMajorMatrix::new(m2, 1),
        ]);
        let bo = mmcs.open_batch(5, &pd);
        println!("OPENED {:?}", bo.opened_values);
        for (i, s) in bo.opening_proof.iter().enumerate() {
            println!("SIB{} {:?}", i, limbs(*s));
        }
        println!("COMMIT {:?}", commit);
    }

    /// Regenerates the Blake3-challenger reference values for `sample_bits_test`
    /// and `pcs_challenger4_test`.
    #[test]
    fn gen_challenger_refs() {
        use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
        use p3_field::{BasedVectorSpace, PrimeField64};
        let g = Val::from_u64;
        fn el(e: ExtVal) -> (u64, u64) {
            let s: &[Val] = e.as_basis_coefficients_slice();
            (s[0].as_canonical_u64(), s[1].as_canonical_u64())
        }
        // sample_bits_test: observe 0x0102030405060708, sample_bits(20).
        let mut ch = Challenger::from_hasher(vec![], Blake3);
        ch.observe(g(0x0102030405060708));
        println!(
            "SAMPLE_BITS {}",
            CanSampleBits::<usize>::sample_bits(&mut ch, 20)
        );
        // pcs_challenger4_test: the α_pcs/α_fri/β/index continuation.
        let mut ch = Challenger::from_hasher(vec![], Blake3);
        ch.observe(g(0x0102030405060708));
        ch.observe(g(0x1122334455667788));
        let apcs: ExtVal = ch.sample_algebra_element();
        let afri: ExtVal = ch.sample_algebra_element();
        println!("APCS {:?}", el(apcs));
        println!("AFRI {:?}", el(afri));
        ch.observe(g(0x00000000deadbeef));
        let beta: ExtVal = ch.sample_algebra_element();
        println!("BETA {:?}", el(beta));
        ch.observe(g(0x0a0b0c0d01020304));
        ch.observe(g(0x0000000000000002));
        println!(
            "SAMPLE_BITS2 {}",
            CanSampleBits::<usize>::sample_bits(&mut ch, 20)
        );
    }
}
