use crate::controllers::radix_tree::RadixNode;
use crate::controllers::napp::{Napp, NappParams};
use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::record::{ApotheosisRecord, RadixKeyMapping};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "R: ApotheosisRecord + serde::Serialize, D: DistanceAlgorithm<R::MetricId> + serde::Serialize, R::MetricId: serde::Serialize",
    deserialize = "R: ApotheosisRecord + serde::Deserialize<'de>, D: DistanceAlgorithm<R::MetricId> + serde::Deserialize<'de>, R::MetricId: serde::Deserialize<'de>"
))]
pub struct Apotheosis<R, D>
where
    R: ApotheosisRecord,
    R::MetricId: Clone,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    pub napp: Napp<D, R::MetricId>,
    pub radix: RadixNode<u8, Option<usize>>,
    pub records: Vec<R>,
    pub config: NappParams,
}

impl<R, D> Apotheosis<R, D>
where
    R: ApotheosisRecord,
    R::MetricId: Clone,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    pub fn new(config: NappParams) -> Self {
        Self {
            napp: Napp::new(),
            radix: RadixNode::<u8, Option<usize>>::new(vec![], None),
            records: vec![],
            config,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.napp.set_seed(seed);
    }

    pub fn insert(&mut self, items: Vec<R>) -> usize {
        let mut features: Vec<R::MetricId> = Vec::with_capacity(items.len());
        for item in items {
            let radix_key = item.search_id().to_radix_key();
            if let Some(ref key) = radix_key {
                if self.radix.find(key).is_some() {
                    continue;
                }
            }
            let index = self.records.len();
            if let Some(key) = radix_key {
                self.radix.insert(key, Some(index));
            }
            features.push(item.search_id());
            self.records.push(item);
        }
        self.napp.set_seed(self.config.seed);
        if let Err(e) = self.napp.build(features, &self.config) {
            eprintln!("NAPP build failed: {e}");
        }
        self.records.len()
    }

    pub fn search(&mut self, query: &R::MetricId, k: usize, ef_search: Option<usize>) -> Vec<(u32, &R)> {
        let ef_search = ef_search.unwrap_or(64);
        let hits: Vec<(u32, usize)> = self
            .napp
            .knn_search(query, k, ef_search)
            .into_iter()
            .map(|(distance, index, _)| (distance, index))
            .collect();
        hits.into_iter()
            .map(|(distance, index)| (distance, &self.records[index]))
            .collect()
    }

    pub fn dump<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: serde::Serialize,
    {
        let mut file = File::create(path)?;
        file.write_all(b"APOP")?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: serde::de::DeserializeOwned,
    {
        let mut file = File::open(path)?;
        let mut header = [0u8; 4];
        file.read_exact(&mut header)?;
        if &header != b"APOP" {
            return Err("Invalid Apotheosis NAPP model file (missing magic bytes)".into());
        }
        Ok(bincode::deserialize_from(file)?)
    }
}
