use crate::datalayer::algorithms::DistanceAlgorithm;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::collections::BinaryHeap;
use tracing::debug;

fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// P:  pivots sampled from the data
// mi: pivots recorded per indexed point
// ms: pivots taken from the query at search time
// t:  minimum shared pivots for a point to become a candidate
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NappParams {
    pub num_pivots: usize,
    pub num_pivot_index: usize,
    pub num_pivot_search: usize,
    pub min_shared: usize,
    /// Candidate cap as a multiple of the search-time ef.
    pub candidate_scale: usize,
    /// PRNG seed for the whole construction. Only the pivot sample is random,
    /// so two seeds give two different pivot sets from the same data and the
    /// same parameters. Vary it to measure that variance (see the harness
    /// `--trials` flag); leave it at the default for a reproducible single run.
    pub seed: u64,
}

impl Default for NappParams {
    fn default() -> Self {
        Self {
            num_pivots: 512,
            num_pivot_index: 32,
            num_pivot_search: 32,
            min_shared: 2,
            candidate_scale: 4,
            seed: 42,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "ID: serde::Serialize, D: DistanceAlgorithm<ID> + serde::Serialize",
    deserialize = "ID: serde::Deserialize<'de>, D: DistanceAlgorithm<ID> + serde::Deserialize<'de>"
))]
pub struct Napp<D, ID>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    features: Vec<ID>,
    pivots: Vec<u32>,
    /// One posting list per pivot, ascending by point index.
    posting: Vec<Vec<u32>>,
    distance: D,
    num_pivot_search: usize,
    min_shared: usize,
    candidate_scale: usize,
    #[serde(skip, default = "default_rng")]
    prng: StdRng,
}

impl<D, ID> Default for Napp<D, ID>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<D, ID> Napp<D, ID>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            pivots: Vec::new(),
            posting: Vec::new(),
            distance: D::default(),
            num_pivot_search: NappParams::default().num_pivot_search,
            min_shared: NappParams::default().min_shared,
            candidate_scale: NappParams::default().candidate_scale,
            prng: default_rng(),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.prng = StdRng::seed_from_u64(seed);
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn pivot_count(&self) -> usize {
        self.pivots.len()
    }

    pub fn default_ef(&self) -> usize {
        64
    }

    pub fn posting_len(&self) -> usize {
        self.posting.iter().map(|p| p.len()).sum()
    }

    pub fn structure_bytes(&self) -> usize {
        self.pivots.len() * std::mem::size_of::<u32>()
            + self.posting.len() * std::mem::size_of::<Vec<u32>>()
            + self
                .posting
                .iter()
                .map(|p| p.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    pub fn build(&mut self, features: Vec<ID>, params: &NappParams) -> std::io::Result<()> {
        let n = features.len();
        self.features = features;
        self.num_pivot_search = params.num_pivot_search.max(1);
        self.min_shared = params.min_shared.max(1);
        self.candidate_scale = params.candidate_scale.max(1);
        self.pivots.clear();
        self.posting.clear();
        if n == 0 {
            return Ok(());
        }

        // Pivots are members of the corpus: there is no way to synthesise a
        // digest, so a reference point has to be an actual one.
        let num_pivots = params.num_pivots.max(1).min(n);
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for i in 0..num_pivots {
            let j = i + (self.prng.next_u64() % (n - i) as u64) as usize;
            perm.swap(i, j);
        }
        self.pivots = perm.into_iter().take(num_pivots).collect();

        let index_width = params.num_pivot_index.max(1).min(self.pivots.len());
        self.posting = vec![Vec::new(); self.pivots.len()];

        let mut scored: Vec<(u32, u32)> = Vec::with_capacity(self.pivots.len());
        for point in 0..n {
            scored.clear();
            for (slot, &pivot) in self.pivots.iter().enumerate() {
                let d = self
                    .distance
                    .calculate_distance(&self.features[point], &self.features[pivot as usize]);
                scored.push((d, slot as u32));
            }
            scored.sort_unstable();
            for &(_, slot) in scored.iter().take(index_width) {
                self.posting[slot as usize].push(point as u32);
            }
            if point % 20_000 == 19_999 {
                debug!("napp: indexed {}/{}", point + 1, n);
            }
        }

        debug!(
            "napp: {} pivots, {} postings, mean list {:.1}",
            self.pivots.len(),
            self.posting_len(),
            self.posting_len() as f64 / self.pivots.len() as f64
        );
        Ok(())
    }

    pub fn knn_search(&self, query_id: &ID, k: usize, ef: usize) -> Vec<(u32, usize, &ID)> {
        let n = self.features.len();
        if n == 0 || self.pivots.is_empty() {
            return Vec::new();
        }
        let k = k.max(1);

        let mut pivot_scores: Vec<(u32, u32)> = self
            .pivots
            .iter()
            .enumerate()
            .map(|(slot, &pivot)| {
                let d = self
                    .distance
                    .calculate_distance(&self.features[pivot as usize], query_id);
                (d, slot as u32)
            })
            .collect();
        pivot_scores.sort_unstable();

        let width = self.num_pivot_search.min(pivot_scores.len());
        // A point can share at most `width` pivots with the query, so a higher
        // threshold would reject everything.
        let min_shared = self.min_shared.min(width).max(1);

        let mut hits: Vec<u32> = Vec::new();
        for &(_, slot) in pivot_scores.iter().take(width) {
            hits.extend_from_slice(&self.posting[slot as usize]);
        }
        hits.sort_unstable();

        let mut candidates: Vec<(u16, u32)> = Vec::new();
        let mut i = 0;
        while i < hits.len() {
            let point = hits[i];
            let mut count = 0u16;
            while i < hits.len() && hits[i] == point {
                count = count.saturating_add(1);
                i += 1;
            }
            if count as usize >= min_shared {
                candidates.push((count, point));
            }
        }

        // Ranked by shared pivots, so the cap keeps the most promising points
        // rather than an arbitrary prefix of the posting lists.
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let cap = ef.max(k).saturating_mul(self.candidate_scale).max(k);
        candidates.truncate(cap);

        let mut top: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
        for &(_, point) in &candidates {
            let d = self
                .distance
                .calculate_distance(&self.features[point as usize], query_id);
            if top.len() < k {
                top.push((d, point));
            } else if let Some(&(worst, _)) = top.peek() {
                if d < worst {
                    top.pop();
                    top.push((d, point));
                }
            }
        }

        top.into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|(d, idx)| (d, idx as usize, &self.features[idx as usize]))
            .collect()
    }
}
