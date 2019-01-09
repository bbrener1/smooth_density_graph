use std::env;
use std::io::Write;
use std::fs::File;
use std::fs::OpenOptions;


extern crate rand;
extern crate num_cpus;
extern crate weighted_sampling;

#[macro_use]
extern crate ndarray;
extern crate rayon;


mod smooth_density_graph;
mod io;
mod clustering;

use io::Parameters;
use clustering::Cluster;
use smooth_density_graph::Graph;
use io::write_vec;

fn main() {

    let mut arg_iter = env::args();

    let mut parameters_raw = Parameters::read(&mut arg_iter);

    let distance = parameters_raw.distance;

    let mut distance_matrix = parameters_raw.distance_matrix.take();

    if distance_matrix.is_none() {
        let counts = parameters_raw.counts.as_ref().expect("No counts or distance matrix specified");
        parameters_raw.distance_matrix = Some(parameters_raw.distance.matrix(counts.view()));
    }

    let counts = parameters_raw.counts.take().unwrap();

    let mut graph = Graph::new(counts,parameters_raw);

    graph.connect();

    let densities = graph.smooth_densities();

    write_vec(densities, &None);

    // let labels = vec![Graph::wandernode(0,graph)];
    // let (final_positions,fuzz) = Graph::wanderlust(graph);
    // let labels = Cluster::quick_cluster(final_positions,fuzz,distance);
    // eprintln!("Wanderlust:{:?}",labels.iter().enumerate().collect::<Vec<(usize,&usize)>>());
    // write_vec(labels,&None);
}

#[cfg(test)]
mod tests {

    use super::*;
    use smooth_density_graph::Graph;
    use io::Distance;

    pub fn example_graph() -> Graph {
        let a = (0..100).map(|x| x as f64).collect();
        let mut p = Parameters::from_vec((25,4), a, 3, 5);
        let points = p.counts.clone().unwrap();
        p.distance_matrix = Some(p.distance.matrix(p.counts.take().unwrap().view()));
        let mut graph = Graph::new(points,p);
        graph.connect();
        graph
    }

    pub fn example_graph_euclidean() -> Graph {
        let a = (0..100).map(|x| x as f64).collect();
        let mut p = Parameters::from_vec((10,10), a, 3, 5);
        p.distance = Distance::Euclidean;
        let points = p.counts.clone().unwrap();
        p.distance_matrix = Some(p.distance.matrix(p.counts.take().unwrap().view()));
        let mut graph = Graph::new(points,p);
        graph.connect();
        graph
    }

    #[test]
    pub fn microtest() {
        let graph = example_graph();
        eprintln!("{:?}",graph.connectivity());
    }

    #[test]
    pub fn wandertest() {
        let graph = example_graph();
        let node = 0;
        eprintln!("{:?}",Graph::wandernode(node, graph));
    }

    #[test]
    pub fn wanderlust() {
        let graph = example_graph();
        let (final_positions,fuzz) = Graph::wanderlust(graph);
        Cluster::quick_cluster(final_positions, fuzz, Distance::Cosine);
        // panic!();
    }


}
