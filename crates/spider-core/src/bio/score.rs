//! Bio score: ((S × Ws × 100) + (F × Wf)) / (Δdays + 2)^G

pub use crate::schema::node::Node;

#[derive(Debug, Clone, Copy)]
pub struct BioParams {
    pub w_sig: f64,
    pub w_freq: f64,
    pub gravity: f64,
}

impl Default for BioParams {
    fn default() -> Self {
        Self { w_sig: 3.0, w_freq: 2.0, gravity: 1.5 }
    }
}

pub fn calculate(node: &Node) -> f64 {
    calculate_with_params(node, &BioParams::default())
}

pub fn calculate_with_params(node: &Node, params: &BioParams) -> f64 {
    let s = (node.significance as f64 / 255.0) * params.w_sig * 100.0;
    let f = (1.0 + node.access_count as f64).ln() * 10.0 * params.w_freq;
    let days = days_since(node.last_accessed_at);
    (s + f) / (days + 2.0).powf(params.gravity)
}

fn days_since(last_access: u32) -> f64 {
    let now = now_unix_secs();
    let diff = now.saturating_sub(last_access);
    (diff as f64) / 86400.0
}

fn now_unix_secs() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_calculation() {
        let node = Node {
            id: 1,
            first_edge_id: 0,
            first_prop_id: 0,
            labels: [0; 4],
            access_count: 10,
            created_at: 0,
            last_accessed_at: now_unix_secs() - 86400,
            significance: 128,
        };
        assert!(calculate(&node) > 0.0);
    }
}
