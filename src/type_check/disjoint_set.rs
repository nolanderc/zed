#[derive(Debug, Clone)]
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

        if rank_a == rank_b {
            // merge b into a
            self.parents[root_b] = root_a;
            self.ranks[root_a] += 1;
        } else if rank_a < rank_b {
            // merge a into b
            self.parents[root_a] = root_b;
        } else {
            // rank_a > rank_b
            // merge b into a
            self.parents[root_b] = root_a;
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

