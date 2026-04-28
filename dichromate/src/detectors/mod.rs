use opencv::core::{Mat, Vec3b};
use petgraph::prelude::UnGraphMap;

use crate::solver::{GraphCell, GraphEdge};

pub mod cv;
pub mod ss;

//   0b 00000000 00000000
// & 0b 00000000 00000001 for RECTANGLE,
// & 0b 00000000 00000010 for WARPS,
// & 0b 00000000 00000100 for CHAINS,
// & 0b 00000000 00001000 for BRIDGE,
// & 0b 00000000 00010000 for SHAPES,
// & 0b 00000000 00100000 for WINDMILL,
// & 0b 00000000 01000000 for OVERPASS,
// & 0b 00000000 10000000 for UNDERPASS,
pub struct SupportedFeatures {
    pub features: u16
}

impl SupportedFeatures {
    pub const fn new() -> Self {
        SupportedFeatures { features: 0 }
    }
    pub const fn with_rectangle(mut self) -> Self {
        self.features |= 0b00000000_00000001;
        self
    }
    pub const fn with_warps(mut self) -> Self {
        self.features |= 0b00000000_00000010;
        self
    }
    pub const fn with_chains(mut self) -> Self {
        self.features |= 0b00000000_00000100;
        self
    }
    pub const fn with_bridge(mut self) -> Self {
        self.features |= 0b00000000_00001000;
        self
    }
    pub const fn with_shapes(mut self) -> Self {
        self.features |= 0b00000000_00010000;
        self
    }
    pub const fn with_windmill(mut self) -> Self {
        self.features |= 0b00000000_00100000;
        self
    }
    pub const fn with_overpass(mut self) -> Self {
        self.features |= 0b00000000_01000000;
        self
    }
    pub const fn with_underpass(mut self) -> Self {
        self.features |= 0b00000000_10000000;
        self
    }
}

pub trait CellDetector {
    const SUPPORTED_FEATURES: SupportedFeatures;

    fn get_affiliations(&self) -> &Vec<Vec3b>;

    fn detect_cells(
        &mut self,
        bit_mat: &Mat,
        bgr: bool
    ) -> opencv::error::Result<UnGraphMap<GraphCell, GraphEdge>>;
}