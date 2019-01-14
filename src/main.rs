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
mod follow_the_crowd;
mod connection_density;

use io::Parameters;
use clustering::Cluster;
use connection_density::Graph;
use io::write_vec;
use io::write_vector;
use io::write_array;

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

    graph.rapid_connection();

    // for _ in 0..5 {
    //     graph.smooth_density();
    //     graph.connect();
    // }

    let labels = Graph::fuzzy_cluster(&mut graph);

    write_vec(labels, &None);

    // write_vec(Graph::history_mode(&wanderers),&None);

    // let (final_positions,fuzz) = graph.fuzzy_positions();
    //
    // write_vec(Cluster::quick_cluster(final_positions, fuzz, distance),&None);

    // Graph::wanderlust(&mut graph);
    // write_vector(graph.populations(), &None);

    // let labels = vec![Graph::wandernode(0,graph)];
    // let wanderers = Graph::wanderlust(&mut graph);
    // let (final_positions,fuzz) = Graph::final_positions(wanderers, &mut graph);
    // let labels = Cluster::quick_cluster(final_positions,fuzz,distance);
    // write_vec(labels,&None);
    // eprintln!("Wanderlust:{:?}",labels.iter().enumerate().collect::<Vec<(usize,&usize)>>());
    // write_array(final_positions,&None);
}

#[cfg(test)]
mod tests {

    use super::*;
    use follow_the_crowd::Graph;
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
        p.steps = 3;
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
        let mut graph = example_graph();
        let wanderers = Graph::wanderlust(&mut graph);
        // Cluster::quick_cluster(final_positions, fuzz, Distance::Cosine);
        // panic!();
    }


}
