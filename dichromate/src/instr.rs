use crate::solver::{Coordinates, GraphCell, GraphCellHint, GraphEdge};
use opencv::core::Vec3b;
use petgraph::prelude::UnGraphMap;

pub enum Instruction {
    Hold,
    Goto(Coordinates),
    Release,
}

pub struct InstructionLinesIterator<T: Iterator<Item = Instruction>> {
    instructions: T,
    held: bool,
    prev: Option<Coordinates>
}

impl<T> Iterator for InstructionLinesIterator<T> 
where 
    T: Iterator<Item = Instruction> {
    type Item = (Coordinates, Coordinates);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(instr) = self.instructions.next() {
            match instr {
                Instruction::Hold => self.held = true,
                Instruction::Goto(b) => {
                    let keep = self.prev;
                    self.prev = Some(b);
                    let Some(a) = keep else { continue };
                    if !self.held { continue }
                    return Some((a, b));
                },
                Instruction::Release => { self.held = false; self.prev = None },
            };
        }
        None
    }
}

impl Instruction {
    /// Emit instructions for all affiliations in a solved Flow board
    pub fn create_vec_from_solved(
        solved: &UnGraphMap<GraphCell, GraphEdge>,
        affs: &Vec<Vec3b>
    ) -> Vec<Self> {
        let mut instructions = Vec::new();

        for aff in 1..=affs.len() {
            Self::emit_path(solved, aff, &mut instructions, affs);
        }

        instructions
    }

    pub fn to_lines_iter(this: Vec<Self>) -> InstructionLinesIterator<std::vec::IntoIter<Instruction>> {
        InstructionLinesIterator { instructions: this.into_iter(), held: false, prev: None }
    }

    /// Walk a single affiliation path and emit instructions directly
    fn emit_path(
        graph: &UnGraphMap<GraphCell, GraphEdge>,
        aff: usize,
        out: &mut Vec<Instruction>,
        affs: &Vec<Vec3b>
    ) {
        // 1. Find the start node (endpoint: exactly 1 neighbor of same affiliation)
        let start = graph
            .nodes()
            .find(|n| {
                n.affiliation == aff
                    && graph
                        .neighbors(*n)
                        .filter(|nbr| nbr.affiliation == aff)
                        .count()
                        == 1
            })
            .expect(&format!("Path must have an endpoint: affiliation is {aff}, {:?}", affs[aff - 1]));

        // 2. Initialize traversal
        let mut current = start;
        let mut prev = None;
        let get_next = |current: GraphCell, prev: Option<GraphCell>| {
            for n in graph.neighbors(current) {
                if n.affiliation == aff && Some(n) != prev {
                    return Some(n);
                }
            }
            None
        };
        out.push(Instruction::Goto(current.location));
        out.push(Instruction::Hold);
        
        // 3. Walk the path
        loop {
            let next = get_next(current, prev);
            let n = match next {
                Some(n) => n,
                None => break,
            };
            out.push(Instruction::Goto(n.location));
            prev = Some(current);
            current = n;
            if n.hint != GraphCellHint::Warp {
                continue
            }
            // Is a warp (warp 1) (n = warp1)
            out.push(Instruction::Release);
            let warp2 = get_next(current, prev)
                .expect("Went into a Warp and there was no exit node");
            assert_eq!(
                warp2.hint, GraphCellHint::Warp,
                "Went into a Warp and there was no exit warp"
            );
            prev = Some(current);
            current = warp2;
            out.push(Instruction::Goto(warp2.location));
            out.push(Instruction::Hold);
            let post_warp = get_next(current, prev)
                .expect("Went through paired warps and there was no exit node");
            assert_ne!(
                post_warp.hint, GraphCellHint::Warp,
                "Can't have more or less than 2 warps side by side"
            );
            prev = Some(current);
            current = post_warp;
            out.push(Instruction::Goto(post_warp.location));
        }
        out.push(Instruction::Release);
    }
}