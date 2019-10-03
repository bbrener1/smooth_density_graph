
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


mod io;
mod clustering;
mod connection_density;

use io::Parameters;
use io::Command;
use connection_density::Graph;
use io::write_vec;
use io::write_vector;
use io::write_array;
use io::renumber;

fn main() {

    let mut arg_iter = env::args();

    let mut parameters_raw = Parameters::read(&mut arg_iter);

    eprintln!("Parameters read");

    parameters_raw.summary();

    let distance = parameters_raw.distance;

    let mut distance_matrix = parameters_raw.distance_matrix.take();

    if distance_matrix.is_none() {
        let counts = parameters_raw.counts.as_ref().expect("No counts or distance matrix specified");
        parameters_raw.distance_matrix = Some(parameters_raw.distance.matrix(counts.view()));
    }
    else {
        parameters_raw.distance_matrix = distance_matrix;
    }

    eprintln!("Distance matrix obtained, computing");

    let mut graph = Graph::new(parameters_raw);

    graph.rapid_connection();

    match graph.parameters.command {
        Command::FitPredict => {

            let mut labels = Graph::fuzzy_cluster(&mut graph);

            labels = renumber(&labels);

            write_vec(labels, &None);

        },
        Command::Density => {
            write_vector(graph.density().to_owned(),&None);
        }
    }


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
    use connection_density::Graph;
    use io::Distance;
    use io::sanitize;
    use ndarray::prelude::*;

    pub fn example_graph() -> Graph {
        let a = (0..100).map(|x| (x - 50) as f64).collect();
        let mut p = Parameters::from_vec((25,4), a, 3, 5);
        let points = p.counts.clone().unwrap();
        p.distance_matrix = Some(p.distance.matrix(p.counts.take().unwrap().view()));
        let mut graph = Graph::new(p);
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
        let mut graph = Graph::new(p);
        graph.connect();
        graph
    }

    #[test]
    pub fn microtest() {
        let graph = example_graph();
        eprintln!("{:?}",graph.connectivity());
    }

    #[test]
    pub fn wanderlust() {
        let mut graph = example_graph();
        let wanderers = Graph::fuzzy_cluster(&mut graph);
        // Cluster::quick_cluster(final_positions, fuzz, Distance::Cosine);
        // panic!();
    }

    #[test]
    pub fn renumber_test() {
        let a = vec![10,5,3,3,10];
        assert_eq!(renumber(&a),vec![0,1,2,2,0]);
    }

    #[test]
    pub fn sanitize_test() {
        let mut a = Array::from_shape_vec((5,5),(0..25).map(|x| x as f64).collect()).unwrap();
        let mut b = a.slice(s![..,1..]).to_owned();
        a.column_mut(0).fill(0.);
        a = sanitize(a);
        assert_eq!(sanitize(a),b);
    }

}
