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
) -> f64 {
    calculate_bio_score_with_params(
        access_count,
        significance,
        last_accessed_at,
        now,
        &BioParams::default(),
    )
}

/// Calculate bio score with custom tuning parameters.
pub fn calculate_bio_score_with_params(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_node_has_positive_score() {
        let now = 1_700_000_000u32;
        let score = calculate_bio_score(1, 128, now, now);
        // Sig: (128/255)*100 = 50.2, Freq: ln(2)*10 = 6.93
        // Num: 57.1, Denom: (0+2)^1 = 2 → Score ≈ 28.5
        assert!(score > 25.0 && score < 35.0, "score was {}", score);
    }

    #[test]
    fn score_decays_over_days() {
        let start = 1_700_000_000u32;
        let score_now = calculate_bio_score(1, 128, start, start);
        let score_1d = calculate_bio_score(1, 128, start, start + 86400);
        let score_7d = calculate_bio_score(1, 128, start, start + 86400 * 7);

        assert!(score_now > score_1d, "1 day decay");
        assert!(score_1d > score_7d, "7 day decay");
    }

    #[test]
    fn frequency_boosts_score() {
        let now = 1_700_000_000u32;
        let score_1 = calculate_bio_score(1, 128, now, now);
        let score_100 = calculate_bio_score(100, 128, now, now);
        assert!(score_100 > score_1);
        // But log-dampened: 100x accesses ≠ 100x score
        assert!(score_100 < score_1 * 10.0, "log dampening works");
    }

    #[test]
    fn significance_boosts_score() {
        let now = 1_700_000_000u32;
        let score_low = calculate_bio_score(1, 0, now, now);
        let score_mid = calculate_bio_score(1, 128, now, now);
        let score_high = calculate_bio_score(1, 255, now, now);
        assert!(score_low < score_mid);
        assert!(score_mid < score_high);
    }

    #[test]
    fn custom_params_affect_score() {
        let now = 1_700_000_000u32;
        let default_score = calculate_bio_score(10, 128, now, now);

        let heavy_sig = BioParams { w_sig: 5.0, ..Default::default() };
        let sig_score = calculate_bio_score_with_params(10, 128, now, now, &heavy_sig);
        assert!(sig_score > default_score, "w_sig boost");

        let heavy_gravity = BioParams { gravity: 2.0, ..Default::default() };
        let one_day_later = now + 86400;
        let normal_decay = calculate_bio_score(10, 128, now, one_day_later);
        let heavy_decay = calculate_bio_score_with_params(10, 128, now, one_day_later, &heavy_gravity);
        assert!(heavy_decay < normal_decay, "higher gravity = faster decay");
    }
}
