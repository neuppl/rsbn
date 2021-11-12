struct DAG<T> {
    nodes: Vec<T>,
    // edges from child to parent
    edges: Vec<(usize, usize)>
}

impl<T> DAG<T> {
    fn new() -> DAG<T> {
        DAG { nodes: Vec::new(), edges: Vec::new() }
    }
    fn add_edge(&mut self, child: usize, parent: usize) -> () {
        self.edges.push((child, parent));
    }
}