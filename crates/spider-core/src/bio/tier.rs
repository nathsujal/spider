//! Storage tier classification based on bio score.
//!
//! [`BioTier`] is a stub for future storage tiering. Right now it is only
//! used as a return value from [`BioTier::from_score`] so higher layers can
//! make routing decisions without knowing the threshold values.
//!
//! ## Future behaviour (not yet implemented)
//!
//! ```text
//! HOT   (RAM)    — Bio Score > high_threshold   — instant access
//! WARM  (SSD)    — medium < score ≤ high        — fast I/O
//! COLD  (archive)— low < score ≤ medium         — slower, can be rehydrated
//! PRUNED         — score ≤ 0                    — eligible for deletion
//! ```

// BioTier

/// The storage tier a node belongs to, derived from its bio score.
///
/// Currently all nodes are treated as [`Hot`](BioTier::Hot) — tiering is a
/// future feature. The enum exists now so the scoring API can already return
/// a tier and callers can match on it without an API break when tiering lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BioTier {
    /// Eligible for deletion — score at or below zero.
    Pruned,
    /// Archived, slow access. Future: compressed off-SSD storage.
    Cold,
    /// On SSD, metadata in memory. Future: fast I/O path.
    Warm,
    /// In RAM. Future: embeddings cached, instant access.
    Hot,
}

impl BioTier {
    // Thresholds — will move to BioParams / meta.db when tiering is implemented.
    const HOT_THRESHOLD:  f64 = 20.0;
    const WARM_THRESHOLD: f64 = 5.0;
    const COLD_THRESHOLD: f64 = 0.0;

    /// Classify a bio score into a storage tier.
    ///
    /// Currently every live node (score > 0) returns [`Hot`](BioTier::Hot)
    /// because the storage backend does not yet implement tiering.
    /// The thresholds are defined now so the classification logic is in place
    /// for when tiering is built.
    #[inline]
    pub fn from_score(score: f64) -> Self {
        if score > Self::HOT_THRESHOLD {
            Self::Hot
        } else if score > Self::WARM_THRESHOLD {
            Self::Warm
        } else if score > Self::COLD_THRESHOLD {
            Self::Cold
        } else {
            Self::Pruned
        }
    }

    /// `true` if this node is eligible for pruning (score ≤ 0).
    #[inline]
    pub fn is_prunable(self) -> bool {
        self == Self::Pruned
    }

    /// `true` if this node is in active storage (Warm or Hot).
    #[inline]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Warm | Self::Hot)
    }
}

impl std::fmt::Display for BioTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot    => write!(f, "Hot"),
            Self::Warm   => write!(f, "Warm"),
            Self::Cold   => write!(f, "Cold"),
            Self::Pruned => write!(f, "Pruned"),
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_score_is_hot() {
        assert_eq!(BioTier::from_score(100.0), BioTier::Hot);
        assert_eq!(BioTier::from_score(20.1),  BioTier::Hot);
    }

    #[test]
    fn medium_score_is_warm() {
        assert_eq!(BioTier::from_score(20.0), BioTier::Warm);
        assert_eq!(BioTier::from_score(10.0), BioTier::Warm);
        assert_eq!(BioTier::from_score(5.1),  BioTier::Warm);
    }

    #[test]
    fn low_score_is_cold() {
        assert_eq!(BioTier::from_score(5.0), BioTier::Cold);
        assert_eq!(BioTier::from_score(0.1), BioTier::Cold);
    }

    #[test]
    fn zero_score_is_pruned() {
        assert_eq!(BioTier::from_score(0.0),  BioTier::Pruned);
        assert_eq!(BioTier::from_score(-1.0), BioTier::Pruned);
        assert_eq!(BioTier::from_score(f64::NEG_INFINITY), BioTier::Pruned);
    }

    #[test]
    fn is_prunable_only_for_pruned() {
        assert!( BioTier::Pruned.is_prunable());
        assert!(!BioTier::Cold.is_prunable());
        assert!(!BioTier::Warm.is_prunable());
        assert!(!BioTier::Hot.is_prunable());
    }

    #[test]
    fn is_active_for_warm_and_hot() {
        assert!(!BioTier::Pruned.is_active());
        assert!(!BioTier::Cold.is_active());
        assert!( BioTier::Warm.is_active());
        assert!( BioTier::Hot.is_active());
    }

    #[test]
    fn tier_ordering_pruned_lt_hot() {
        assert!(BioTier::Pruned < BioTier::Cold);
        assert!(BioTier::Cold   < BioTier::Warm);
        assert!(BioTier::Warm   < BioTier::Hot);
    }

    #[test]
    fn display_names() {
        assert_eq!(BioTier::Hot.to_string(),    "Hot");
        assert_eq!(BioTier::Warm.to_string(),   "Warm");
        assert_eq!(BioTier::Cold.to_string(),   "Cold");
        assert_eq!(BioTier::Pruned.to_string(), "Pruned");
    }
}