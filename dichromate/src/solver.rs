use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::RangeInclusive;

use itertools::Itertools;
use petgraph::graphmap::{NodeTrait, UnGraphMap};
use rustsat::encodings::am1::{Encode, Ladder};
use unordered_pair::UnorderedPair;
use rustsat::types::{Assignment, Clause, Lit, TernaryVal, Var};
use rustsat::instances::{BasicVarManager, Cnf};
use rustsat::solvers::{Solve, SolveStats, SolverResult};

/// Constraint on node types given to [`GraphSolver`].
pub trait Terminus: NodeTrait /* constraints on GraphMap */ {
    fn is_terminus(&self) -> Option<NonZero<usize>>;
}

/// Reasons a [`GraphSolver`] may fail.
#[derive(Debug)]
pub enum SolverFailure {
    /// The SAT solver detected a logical inconsistency, i.e. the graph as stated is unsolvable.
    Inconsistent,
    /// The SAT solver could not solve the affiliation of at least one node and/or edge.
    /// If this happens, then the given board has no dots (all affs = 0)
    NoAffFound,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum HasAffiliation<N, E>
where
    N: Terminus,
{
    Node { node: N },
    Edge { edge: E, endpoints: UnorderedPair<N> },
}

impl<N, E> HasAffiliation<N, E>
where
    N: Terminus,
    E: Copy,
{
    pub(crate) fn from_node(node: N) -> Self {
        Self::Node { node }
    }

    pub(crate) fn from_edge(triple: (N, N, &E)) -> Self {
        let (n1, n2, e) = triple;
        Self::Edge { edge: *e, endpoints: UnorderedPair(n1, n2) }
    }
}

/// The most general implementation of the logic necessary to solve a graph in accordance with the rules for Numberlink.
/// Use [`Self::solve`] to attempt to find a solution.
///
/// The only requirement is that the node struct on the input graph implements [`Terminus`], so it may be noted as a Terminus.
struct GraphSolver<'gph, N, E, T>
where
    N: Terminus,
    E: PartialEq + Eq + Hash + Copy,
    T: Default + Solve + SolveStats
{
    graph: &'gph UnGraphMap<N, E>,
    affiliation_index: HashMap<HasAffiliation<N, E>, usize>,
    max_affiliation: usize,
    phantom: PhantomData<T>
}

impl<'gph, N, E, T> From<(&'gph UnGraphMap<N, E>, Option<usize>)> for GraphSolver<'gph, N, E, T>
where
    N: Terminus,
    E: PartialEq + Eq + Hash + Copy,
    T: Default + Solve + SolveStats
{
    fn from((graph, max_affiliation): (&'gph UnGraphMap<N, E>, Option<usize>)) -> Self {
        let max_affiliation = max_affiliation.unwrap_or_else(|| {
            match graph.nodes().filter_map(|node| node.is_terminus().and_then(Some)).max() {
                None => 0,
                Some(max) => max.get(),
            }
        });
        // Here, we give each node and each edge an id/index
        let affiliation_index = graph.nodes()
            .map(HasAffiliation::from_node)
            .chain(graph.all_edges().map(HasAffiliation::from_edge))
            .enumerate()
            .map(|(i, h)| (h, i))
            .collect::<HashMap<_, _>>();
        Self {
            graph,
            affiliation_index,
            max_affiliation,
            phantom: PhantomData
        }
    }
}

impl<N, E, T> GraphSolver<'_, N, E, T>
where
    N: Terminus,
    E: PartialEq + Eq + Hash + Copy,
    T: Default + Solve + SolveStats
{
    #[inline]
    fn valid_affiliations(&self) -> RangeInclusive<usize> {
        0..=self.max_affiliation
    }

    #[inline]
    fn valid_non_null_affiliations(&self) -> RangeInclusive<usize> {
        1..=self.max_affiliation
    }

    #[inline]
    fn affiliation_var(&self, subject: HasAffiliation<N, E>, affiliation: usize) -> Var {
        // max + 1 = number of affiliations beause if 0..=max, then len = max + 1
        // Each node and edge in the graph can have 0..=max affiliations, so there
        // are max + 1 possibilities for each node and edge. The +affiliation tells
        // us which affiliation we are interested about. Suppose edge was the 16th
        // element, and the board has max_affiliation = 9, and it's affiliation was 5.
        // Then we need to store x_160_5.pos_lit() and x_160_(0..=10 skip 5).neg_lit().
        // That is why we get the index, multiply it by the # of affiliations, and then
        // add the affiliation we want to get a var
        let idx = self.affiliation_index[&subject] * (self.max_affiliation + 1) + affiliation;
        Var::new(idx as u32)
    }

    #[inline]
    fn solved_affiliation_of(&self, model: &Assignment, subject: HasAffiliation<N, E>, allow_zero: bool) -> Option<usize> {
        // We go through each affiliation (including/excluding 0 depending on if we want to consider it)
        // and we find the affiliation for which the var for the subject holds true. That is the affiliation
        // of the subject
        (if allow_zero { self.valid_affiliations() } else { self.valid_non_null_affiliations() })
            .find(|aff| model.lit_value(self.affiliation_var(subject, *aff).pos_lit()) == TernaryVal::True)
    }

    /// Solve a Flow graph, returning [`Ok`] with a [`HashMap`] of solved affiliations for each edge and vertex or [`Err`] with a [`SolverFailure`] reason.
    ///
    /// # Logical setup
    /// Suppose this board is undirected graph G.
    ///
    /// ## Vertices
    /// Every vertex V on G must have exactly one nonzero affiliation.
    /// If V is a Terminus, its affiliation is known and all other affiliations are incorrect.
    /// Exactly one incident edge has the same affiliation (the edge by which the path exits this Terminus).
    /// Every other incident edge has no affiliation (i.e. affiliation 0).
    ///
    /// If V is not a Terminus, it must have exactly one (not yet known) affiliation A.
    /// Then V is on the path between the two termini with affiliation A and has two incident edges with affiliation A.
    /// Every other incident edge has no affiliation.
    ///
    /// ## Edges
    /// Every edge E on G has exactly one affiliation, which may be 0.
    ///
    /// The two endpoints of E have the same affiliation if and only if E has the same nonzero affiliation.
    /// So, by complement, the two endpoints of E have different affiliation if and only if E has no affiliation.
    /// We encode the former of these two biconditionals.
    pub fn solve(&self) -> Result<HashMap<HasAffiliation<N, E>, usize>, SolverFailure> {
        let num_aff_vars = self.affiliation_index.len() * (self.max_affiliation + 1);
        let mut cnf = Cnf::new();
        let mut vm = BasicVarManager::from_next_free(Var::new(num_aff_vars as u32));
        for vertex in self.graph.nodes() {
            // let this vertex be V
            if let Some(aff) = vertex.is_terminus() {
                // the affiliation of V is the one already assigned, and no other; we tell the solver to assume this is so
                self.valid_affiliations().map(|maybe_aff| {
                    let v = self.affiliation_var(HasAffiliation::from_node(vertex), maybe_aff);
                    if maybe_aff == aff.get() { v.pos_lit() } else { v.neg_lit() }
                }).for_each(|a| cnf.add_clause(Clause::from([a])));

                let incident = self.graph.edges(vertex).collect_vec();
                // exactly one incident edge E has the same affiliation
                let vars = incident.iter()
                    .map(|e_triple| self.affiliation_var(HasAffiliation::from_edge(*e_triple), aff.get()).pos_lit())
                    .collect_vec();
                // encoding "exactly 1" = enconding "at least 1 AND at most 1"
                cnf.add_clause(Clause::from(vars.as_slice()));
                Ladder::from(vars).encode(&mut cnf, &mut vm).unwrap();

                // V has deg(V) - 1 incident edges with affiliation 0 (unaffiliated)
                // or, equivalently, exactly 1 incident edge does *not* have affiliation 0
                let vars = incident.iter()
                    .map(|e_triple| self.affiliation_var(HasAffiliation::from_edge(*e_triple), 0).neg_lit())
                    .collect_vec();
                // encoding "exactly 1" = enconding "at least 1 AND at most 1"
                cnf.add_clause(Clause::from(vars.as_slice()));
                Ladder::from(vars).encode(&mut cnf, &mut vm).unwrap();
            } else {
                // V must have nonzero affiliation
                cnf.add_clause(Clause::from([self.affiliation_var(HasAffiliation::from_node(vertex), 0).neg_lit()]));
            
                // V has only one affiliation
                let vars = self.valid_non_null_affiliations()
                    .map(|aff| self.affiliation_var(HasAffiliation::from_node(vertex), aff).pos_lit())
                    .collect_vec();
                // encoding "exactly 1" = enconding "at least 1 AND at most 1"
                cnf.add_clause(Clause::from(vars.as_slice()));
                Ladder::from(vars).encode(&mut cnf, &mut vm).unwrap();

                let all_incident = self.graph.edges(vertex).collect_vec();
                for aff in self.valid_non_null_affiliations() {
                    {
                        let mut terms = Vec::with_capacity(1 + all_incident.len());
                        terms.push(self.affiliation_var(HasAffiliation::from_node(vertex), aff).neg_lit());
                        terms.extend(all_incident.iter()
                            .map(|e_triple| self.affiliation_var(HasAffiliation::from_edge(*e_triple), aff).pos_lit())
                        );
                        cnf.add_clause(Clause::from(terms.as_slice()));
                    }

                    for clause in all_incident.iter().map(|e1_triple| {
                        all_incident.iter()
                            .map(|e_triple| {
                                let v = self.affiliation_var(HasAffiliation::from_edge(*e_triple), aff);
                                if e1_triple != e_triple { v.pos_lit() } else { v.neg_lit() }
                            })
                            .collect_vec()
                    }) {
                        cnf.add_clause(Clause::from(clause.as_slice()));
                    }

                    for c in all_incident.iter()
                        .combinations(3)
                        .map(|sel| sel.iter()
                            .map(|e_triple| self.affiliation_var(HasAffiliation::from_edge(**e_triple), aff).neg_lit())
                            .collect_vec())
                    {
                        cnf.add_clause(Clause::from(c.as_slice()));
                    }
                }
            }
        }

        for edge_triple in self.graph.all_edges() {
            // this edge E has exactly one affiliation, which may be 0
            let vars = self.valid_affiliations()
                .map(|aff| self.affiliation_var(HasAffiliation::from_edge(edge_triple), aff).pos_lit())
                .collect_vec();
            // encoding "exactly 1" = enconding "at least 1 AND at most 1"
            cnf.add_clause(Clause::from(vars.as_slice()));
            Ladder::from(vars).encode(&mut cnf, &mut vm).unwrap();

            for aff in self.valid_non_null_affiliations() {
                // E having a non-null affiliation <=> its vertices have the same affiliation
                // let this be A <=> BC
                // A => BC = !A + BC = (!A + B)(!A + C)
                // BC => A = !(BC) + A = !B + !C + A
                // together, A <=> BC = (!A + B)(!A + C)(A + !B + !C)
                let a = self.affiliation_var(HasAffiliation::from_edge(edge_triple), aff);
                let b = self.affiliation_var(HasAffiliation::from_node(edge_triple.0), aff);
                let c = self.affiliation_var(HasAffiliation::from_node(edge_triple.1), aff);

                cnf.add_clause(Clause::from([a.neg_lit(), b.pos_lit()].as_slice()));
                cnf.add_clause(Clause::from([a.neg_lit(), c.pos_lit()].as_slice()));
                cnf.add_clause(Clause::from([a.pos_lit(), b.neg_lit(), c.neg_lit()].as_slice()));
            }
        }

        let mut solver = T::default();
        if solver.add_cnf(cnf).is_err() { return Err(SolverFailure::Inconsistent); }
        let Ok(SolverResult::Sat) = solver.solve() else { return Err(SolverFailure::Inconsistent) };
        let model = solver.full_solution().unwrap();
        let mut solved_affiliations = HashMap::new();
        for node in self.graph.nodes() {
            solved_affiliations.insert(
                HasAffiliation::from_node(node),
                match self.solved_affiliation_of(&model, HasAffiliation::from_node(node), false) {
                    None => return Err(SolverFailure::NoAffFound),
                    Some(aff) => aff
                });
        }

        for edge_triple in self.graph.all_edges() {
            solved_affiliations.insert(
                HasAffiliation::from_edge(edge_triple),
                match self.solved_affiliation_of(&model, HasAffiliation::from_edge(edge_triple), true) {
                    None => return Err(SolverFailure::NoAffFound),
                    Some(aff) => aff
                });
        }

        Ok(solved_affiliations)
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Coordinates {
    pub x: usize,
    pub y: usize
}

impl From<(usize, usize)> for Coordinates {
    fn from(value: (usize, usize)) -> Self {
        Self { x: value.0, y: value.1 }
    }
}

impl Into<(usize, usize)> for Coordinates {
    fn into(self) -> (usize, usize) {
        (self.x, self.y)
    }
}

impl Coordinates {
    pub fn dne() -> Self {
        return Self { x: usize::MAX, y: usize::MAX }
    }

    pub fn is_dne(&self) -> bool {
        return *self == Self::dne()
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GraphCellHint {
    Phantom, // Used for extreme alley,
    Warp, // Used in warps (we must let go and go to other side of board)
    // These are just here, but I'll probably never use them
    Empty,
    Terminus,
    Bridge,
    Windmill
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GraphCell {
    pub hint: GraphCellHint,
    pub affiliation: usize,
    pub location: Coordinates,
}

impl GraphCell {
    pub fn new_def(
        hint: GraphCellHint,
        affiliation: usize,
        location: Coordinates,
    ) -> Self {
        Self { hint, affiliation, location }
    }
}

impl Terminus for GraphCell {
    fn is_terminus(&self) -> Option<std::num::NonZero<usize>> {
        std::num::NonZero::new(self.affiliation)
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GraphEdge {
     pub affiliation: usize
}

pub struct ArbitraryGraphSolver {
    pub(crate) graph: UnGraphMap<GraphCell, GraphEdge>,
    pub(crate) max_affiliation: Option<usize>,
}

impl ArbitraryGraphSolver {
    pub fn new(graph: UnGraphMap<GraphCell, GraphEdge>, max_affiliation: Option<usize>) -> Self {
        Self { graph, max_affiliation }
    }

    /// Solves this board, deferring to a [`GraphSolver`](crate::solver::GraphSolver) and mutating and returning `self` accordingly.
    ///
    /// Returns according to the result of [`GraphSolver::solve`](crate::solver::GraphSolver::solve).
    pub fn solve<T: Default + Solve + SolveStats>(&self) -> Result<UnGraphMap<GraphCell, GraphEdge>, SolverFailure> {
        let solver = GraphSolver::<_, _, T>::from((&self.graph, self.max_affiliation));
        let solution = solver.solve()?;

        let mut solved_graph = UnGraphMap::with_capacity(self.graph.node_count(), self.graph.edge_count());
        for node in self.graph.nodes() {
            let mut new_node = node.clone();
            if node.affiliation == 0 {
                new_node.affiliation = *solution.get(&HasAffiliation::from_node(node)).unwrap();
            }
            // existing Terminus and path cells can stay as is
            solved_graph.add_node(new_node);
        }

        for triple in self.graph.all_edges() {
            let (n1, n2, e) = triple;

            let mut new_e = *e;
            new_e.affiliation = *solution.get(&HasAffiliation::from_edge(triple)).unwrap();

            solved_graph.add_edge(
                solved_graph.nodes().find(|n| n.location == n1.location).unwrap(),
                solved_graph.nodes().find(|n| n.location == n2.location).unwrap(),
                new_e
            );
        }

        Ok(solved_graph)
    }
}