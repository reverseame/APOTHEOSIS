// AUTHORS: Daniel Huici and Ricardo J. Rodríguez
// Copyright (c) 2025 - 2026
// GPLv3 License
// reverseame@unizar.es

use crate::controllers::hnsw::Hnsw;
use crate::controllers::radix_tree::RadixNode;
use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::record::{ApotheosisRecord, RadixKeyMapping};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Last path segment of the type name, e.g.
/// "apotheosis2::datalayer::algorithms::TlshDistance" -> "TlshDistance".
fn distance_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "R: ApotheosisRecord + serde::Serialize, D: DistanceAlgorithm<R::MetricId> + serde::Serialize, R::MetricId: serde::Serialize",
    deserialize = "R: ApotheosisRecord + serde::Deserialize<'de>, D: DistanceAlgorithm<R::MetricId> + serde::Deserialize<'de>, R::MetricId: serde::Deserialize<'de>"
))]
pub struct Apotheosis<
    R,
    D,
    const M: usize = 16,
    const M0: usize = 32,
    const EF: usize = 400,
    const HEURISTIC: bool = false,
> where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    hnsw: Hnsw<D, R::MetricId, M, M0, EF, HEURISTIC>,
    radix: RadixNode<u8, Option<usize>>,
    records: Vec<R>,
}

impl<R, D, const M: usize, const M0: usize, const EF: usize, const HEURISTIC: bool>
    Apotheosis<R, D, M, M0, EF, HEURISTIC>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    pub fn new() -> Self {
        Self {
            hnsw: Hnsw::new(),
            radix: RadixNode::<u8, Option<usize>>::new(vec![], None),
            records: vec![],
        }
    }
}

impl<R, D, const M: usize, const M0: usize, const EF: usize, const HEURISTIC: bool> Default
    for Apotheosis<R, D, M, M0, EF, HEURISTIC>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R, D, const M: usize, const M0: usize, const EF: usize, const HEURISTIC: bool>
    Apotheosis<R, D, M, M0, EF, HEURISTIC>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    /// Inserts an ApotheosisRecord into the Apotheosis Model (radix tree + HNSW + vector metadata).
    /// The record's `search_id` is stored in HNSW for similarity search, the `radix_key` is mapped
    /// to the record's index, and the record itself is stored for later retrieval.
    ///
    /// # Parameters
    /// * `item` - The ApotheosisRecord to insert
    ///
    /// # Returns
    /// * `true` if the record was successfully inserted
    /// * `false` if the record's key already exists in the model
    pub fn insert(&mut self, item: R) -> bool {
        let radix_key = item.search_id().to_radix_key();

        // A radix node can exist without holding a value: splitting the tree on a
        // shared prefix creates internal nodes whose `data` is None. Only a node
        // that actually holds an index is a real duplicate.
        if let Some(ref key) = radix_key
            && self.radix.find(key).is_some_and(|node| node.data.is_some())
        {
            tracing::warn!("Key already exists in radix tree: {:?}", key);
            return false;
        }

        let hnsw_node_index = self.hnsw.insert(item.search_id());
        if let Some(key) = radix_key {
            self.radix.insert(key, Some(hnsw_node_index));
        }

        self.records.push(item);
        true
    }

    /// Number of records currently indexed.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the model has no records indexed yet.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// HNSW nodes and edges per layer, layer 0 first.
    #[allow(clippy::type_complexity)]
    pub fn draw_model(&self) -> Vec<(Vec<usize>, Vec<(String, String, f32)>)> {
        self.hnsw.draw_model()
    }

    /// Record at `index`; resolves the node indices returned by `draw_model()`.
    pub fn record(&self, index: usize) -> Option<&R> {
        self.records.get(index)
    }

    /// Performs an approximate k-NN search. If the `MetricId` natively maps to a Radix Tree key
    /// (e.g. TLSH string), it jumps directly to the HNSW node to retrieve neighbors, bypassing full traversal.
    /// Otherwise, it performs a standard HNSW k-NN search using the `query`.
    ///
    /// # Parameters
    /// * `query` - The native HNSW numerical or hash representation
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// * `Vec<(u32, &R)>` - List of tuples containing:
    ///   - `u32`: Distance/score to the query
    ///   - `&R`: Reference to the actual retrieved Record item
    pub fn search(
        &self,
        query: &R::MetricId,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(u32, &R)> {
        if self.records.is_empty() {
            return vec![];
        }

        let ef_search = ef_search.unwrap_or(24);

        let hnsw_results: Vec<(u32, usize, &R::MetricId)> = if let Some(key) = query.to_radix_key()
        {
            if let Some(radix_node) = self.radix.find(&key) {
                if let Some(Some(node_index)) = radix_node.data {
                    // Exact match found! Jump directly to neighbors in HNSW
                    self.hnsw.get_neighbors_node(node_index)
                } else {
                    self.hnsw.knn_search(query, k, ef_search)
                }
            } else {
                self.hnsw.knn_search(query, k, ef_search)
            }
        } else {
            self.hnsw.knn_search(query, k, ef_search)
        };

        let mut hnsw_results = hnsw_results;
        hnsw_results.sort_by_key(|(d, _, _)| *d);
        hnsw_results
            .into_iter()
            .take(k)
            .map(|(distance, index, _id)| (distance, &self.records[index]))
            .collect()
    }

    pub fn dump<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: serde::Serialize,
    {
        let mut file = File::create(path)?;
        // Write magic bytes: "APOT" (4 bytes)
        file.write_all(b"APOT")?;
        // Write M, M0, EF as u32 (12 bytes)
        file.write_all(&(M as u32).to_le_bytes())?;
        file.write_all(&(M0 as u32).to_le_bytes())?;
        file.write_all(&(EF as u32).to_le_bytes())?;
        file.write_all(&[HEURISTIC as u8])?;
        let name = distance_name::<D>().as_bytes();
        file.write_all(&[u8::try_from(name.len())?])?;
        file.write_all(name)?;

        // Then write the model data
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: serde::de::DeserializeOwned,
    {
        let mut file = File::open(path)?;

        // Read and verify header (17 bytes + distance type name)
        let mut header = [0u8; 17];
        file.read_exact(&mut header)?;

        if &header[0..4] != b"APOT" {
            return Err("Invalid Apotheosis model file (missing magic bytes)".into());
        }

        let m = u32::from_le_bytes(header[4..8].try_into()?);
        let m0 = u32::from_le_bytes(header[8..12].try_into()?);
        let ef = u32::from_le_bytes(header[12..16].try_into()?);
        let heuristic = header[16] != 0;

        if m != M as u32 || m0 != M0 as u32 || ef != EF as u32 || heuristic != HEURISTIC {
            return Err(format!(
                "Model parameter mismatch. File has M={}, M0={}, EF={}, HEURISTIC={} but code expects M={}, M0={}, EF={}, HEURISTIC={}",
                m, m0, ef, heuristic, M, M0, EF, HEURISTIC
            ).into());
        }

        // Explicit check of distance type (does not rely on bincode)
        let mut name_len = [0u8; 1];
        file.read_exact(&mut name_len)?;
        let mut name = vec![0u8; name_len[0] as usize];
        file.read_exact(&mut name)?;
        let file_distance = String::from_utf8_lossy(&name);

        let expected = distance_name::<D>();
        if file_distance != expected {
            return Err(format!(
                "Distance type mismatch. File was built with {} but is being loaded as {}",
                file_distance, expected
            )
            .into());
        }

        let decoded = bincode::deserialize_from(file)?;
        Ok(decoded)
    }
}
