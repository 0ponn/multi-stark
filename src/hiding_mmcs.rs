//! A salted (hiding) Merkle-tree MMCS that is `Sync`.
//!
//! This is Plonky3's `MerkleTreeHidingMmcs` (p3-merkle-tree, rev e9d7561,
//! MIT OR Apache-2.0) with one change: the salt generator sits behind a
//! `Mutex` instead of a `RefCell`, so the MMCS, and therefore the whole
//! zero-knowledge configuration, can be shared across threads. The salting
//! scheme itself is unchanged: every committed row is extended with
//! `SALT_ELEMS` fresh random field elements before hashing (Section 3 of
//! <https://eprint.iacr.org/2016/116>), and openings carry the salts.

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::HorizontalPair;
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::{MerkleCap, MerkleTree, MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::zip_eq::zip_eq;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// See the module documentation.
#[derive(Debug)]
pub struct SyncHidingMmcs<
    P,
    PW,
    H,
    C,
    R,
    const N: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
> {
    inner: MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>,
    rng: Mutex<R>,
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    SyncHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Creates the MMCS. `rng` must be a cryptographically secure generator;
    /// it supplies the leaf salts.
    pub const fn new(hash: H, compress: C, cap_height: usize, rng: R) -> Self {
        Self {
            inner: MerkleTreeMmcs::new(hash, compress, cap_height),
            rng: Mutex::new(rng),
        }
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Clone
    for SyncHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>: Clone,
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rng: Mutex::new(self.rng.lock().expect("salt rng poisoned").clone()),
        }
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<P::Value> for SyncHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: Rng + Clone,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type ProverData<M> = MerkleTree<
        P::Value,
        PW::Value,
        HorizontalPair<M, RowMajorMatrix<P::Value>>,
        N,
        DIGEST_ELEMS,
    >;
    type Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>;
    /// The salts, then the usual Merkle proof (sibling digests).
    type Proof = (Vec<Vec<P::Value>>, Vec<[PW::Value; DIGEST_ELEMS]>);
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let salted_inputs = {
            let mut rng = self.rng.lock().expect("salt rng poisoned");
            inputs
                .into_iter()
                .map(|mat| {
                    let salts = RowMajorMatrix::rand(&mut *rng, mat.height(), SALT_ELEMS);
                    HorizontalPair::new(mat, salts)
                })
                .collect()
        };
        self.inner.commit(salted_inputs)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        let (salted_openings, siblings) = self.inner.open_batch(index, prover_data).unpack();
        let (openings, salts) = salted_openings
            .into_iter()
            .map(|row| {
                let (a, b) = row.split_at(row.len() - SALT_ELEMS);
                (a.to_vec(), b.to_vec())
            })
            .unzip();
        BatchOpening::new(openings, (salts, siblings))
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| &mat.left)
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, (salts, siblings)) = batch_opening.unpack();
        let opened_salted_values = zip_eq(opened_values, salts, MerkleTreeError::WrongBatchSize)?
            .map(|(opened, salt)| {
                opened
                    .iter()
                    .chain(salt.iter())
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.inner.verify_batch(
            commit,
            dimensions,
            index,
            BatchOpeningRef::new(&opened_salted_values, siblings),
        )
    }
}
