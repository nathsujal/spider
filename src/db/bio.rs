//! # Bio Scoring
//!
//! Implements the "Living Memory" formula for Spider nodes.
//!
//! ## Formula
//!
//! ```text
//! Life Score = ((S × Ws × 100) + (F × Wf)) / (Δdays + 2)^G
//! ```
//!
//! Where:
//! - S = Significance (0.0-1.0, stored as u8 0-255)
//! - F = Frequency (log-dampened access count)
//! - Δdays = Days since last access
//! - Ws, Wf = Tuning weights (default 1.0, future RL)
//! - G = Gravity (default 1.0, future RL)

/// Tuning parameters for the bio score formula.
///
/// These default to 1.0 and are designed to be tuned via
/// reinforcement learning in a future phase.
pub struct BioParams {
    /// Weight for significance component.
    pub w_sig: f64,
    /// Weight for frequency component.
    pub w_freq: f64,
    /// Decay exponent (higher = faster forgetting).
    pub gravity: f64,
}

impl Default for BioParams {
    fn default() -> Self {
        Self {
            w_sig: 1.0,
            w_freq: 1.0,
            gravity: 1.0,
        }
    }
}

/// Calculate the bio score for a node.
///
/// Uses log-dampened frequency to prevent "super-nodes" from
/// dominating the graph. Time decay is measured in days.
///
/// # Arguments
/// * `access_count` - Number of times the node was accessed
/// * `significance` - User-assigned importance (0-255)
/// * `last_accessed_at` - Unix timestamp (seconds) of last access
/// * `now` - Current Unix timestamp (seconds)
pub fn calculate_bio_score(
    access_count: u32,
    significance: u8,
    last_accessed_at: u32,
    now: u32,
    params: &BioParams,
) -> f64 {
    // 1. Frequency: Log-dampened to prevent super-nodes
    //    ln(1+1) = 0.69, ln(1+10) = 2.4, ln(1+100) = 4.6
    let freq_score = (access_count as f64).ln_1p() * 10.0;

    // 2. Significance: Normalized 0.0-1.0, scaled by 100
    let sig_score = (significance as f64 / 255.0) * 100.0;

    // 3. Time decay: Days since last access
    let elapsed_secs = now.saturating_sub(last_accessed_at);
    let days_since = elapsed_secs as f64 / 86400.0;

    // 4. The formula
    let numerator = (sig_score * params.w_sig) + (freq_score * params.w_freq);
    let denominator = (days_since + 2.0).powf(params.gravity);

    numerator / denominator
}