use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct DisjointSet {
    parents: Vec<usize>,
    ranks: Vec<usize>,
}

impl DisjointSet {
    /// Create a new set.
    pub fn new() -> DisjointSet {
        DisjointSet {
            parents: Vec::new(),
            ranks: Vec::new(),
        }
    }

    /// Add a new node to the set.
    pub fn add(&mut self) -> usize {
        let id = self.parents.len();
        self.parents.push(id);
        self.ranks.push(0);
        id
    }

    /// Find the root ancestor of a node.
    pub fn find(&mut self, index: usize) -> usize {
        let mut current = index;
        let mut parent = self.parents[index];
        while current != parent {
            let grandfather = self.parents[parent];
            self.parents[index] = grandfather;
            current = parent;
            parent = grandfather;
        }
        current
    }

    /// Find the root ancestor node without compressing paths.
    pub fn find_immutable(&self, index: usize) -> usize {
        let mut current = index;
        let mut parent = self.parents[index];
        while current != parent {
            let grandfather = self.parents[parent];
            current = parent;
            parent = grandfather;
        }
        current
    }

    /// Merge the forests containing the two nodes.
    pub fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            // Already in same set
            return;
        }

        let rank_a = self.ranks[root_a];
        let rank_b = self.ranks[root_b];

        match rank_a.cmp(&rank_b) {
            Ordering::Equal => {
                self.parents[root_b] = root_a;
                self.ranks[root_a] += 1;
            }
            Ordering::Less => {
                self.parents[root_a] = root_b;
            }
            Ordering::Greater => {
                self.parents[root_b] = root_a;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_two_nodes() {
        let mut set = DisjointSet::new();
        let a = set.add();
        let b = set.add();

        assert_ne!(set.find(a), set.find(b));

        set.union(a, b);

        assert_eq!(set.find(a), set.find(b));
    }
}

impl Debug for DisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sets = BTreeMap::new();

        for i in 0..self.parents.len() {
            let set = self.find_immutable(i);
            sets.entry(set).or_insert_with(BTreeSet::new).insert(i);
        }

        struct Raw(String);

        impl Debug for Raw {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        let sets = sets
            .into_iter()
            .map(|(_, values)| {
                let numbers = values
                    .into_iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Raw(format!("{{ {} }}", numbers))
            })
            .collect::<Vec<_>>();

        f.debug_struct("DisjointSet").field("sets", &sets).finish()
    }
}
