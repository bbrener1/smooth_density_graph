
use std::env;
use std::io::Write;
use std::fs::File;
use std::fs::OpenOptions;


extern crate rand;
extern crate num_cpus;
// extern crate weighted_sampling;

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
    use io::cosine_similarity_matrix;
    use io::sanitize;
    use ndarray::prelude::*;

    pub fn iris_matrix() -> Array<f64,Ix2> {
        let iris = array!
        [[5.1,3.5,1.4,0.2],
        [4.9,3.,1.4,0.2],
        [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2],
        [5.,3.6,1.4,0.2],
        [5.4,3.9,1.7,0.4],
        [4.6,3.4,1.4,0.3],
        [5.,3.4,1.5,0.2],
        [4.4,2.9,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [5.4,3.7,1.5,0.2],
        [4.8,3.4,1.6,0.2],
        [4.8,3.,1.4,0.1],
        [4.3,3.,1.1,0.1],
        [5.8,4.,1.2,0.2],
        [5.7,4.4,1.5,0.4],
        [5.4,3.9,1.3,0.4],
        [5.1,3.5,1.4,0.3],
        [5.7,3.8,1.7,0.3],
        [5.1,3.8,1.5,0.3],
        [5.4,3.4,1.7,0.2],
        [5.1,3.7,1.5,0.4],
        [4.6,3.6,1.,0.2],
        [5.1,3.3,1.7,0.5],
        [4.8,3.4,1.9,0.2],
        [5.,3.,1.6,0.2],
        [5.,3.4,1.6,0.4],
        [5.2,3.5,1.5,0.2],
        [5.2,3.4,1.4,0.2],
        [4.7,3.2,1.6,0.2],
        [4.8,3.1,1.6,0.2],
        [5.4,3.4,1.5,0.4],
        [5.2,4.1,1.5,0.1],
        [5.5,4.2,1.4,0.2],
        [4.9,3.1,1.5,0.1],
        [5.,3.2,1.2,0.2],
        [5.5,3.5,1.3,0.2],
        [4.9,3.1,1.5,0.1],
        [4.4,3.,1.3,0.2],
        [5.1,3.4,1.5,0.2],
        [5.,3.5,1.3,0.3],
        [4.5,2.3,1.3,0.3],
        [4.4,3.2,1.3,0.2],
        [5.,3.5,1.6,0.6],
        [5.1,3.8,1.9,0.4],
        [4.8,3.,1.4,0.3],
        [5.1,3.8,1.6,0.2],
        [4.6,3.2,1.4,0.2],
        [5.3,3.7,1.5,0.2],
        [5.,3.3,1.4,0.2],
        [7.,3.2,4.7,1.4],
        [6.4,3.2,4.5,1.5],
        [6.9,3.1,4.9,1.5],
        [5.5,2.3,4.,1.3],
        [6.5,2.8,4.6,1.5],
        [5.7,2.8,4.5,1.3],
        [6.3,3.3,4.7,1.6],
        [4.9,2.4,3.3,1.],
        [6.6,2.9,4.6,1.3],
        [5.2,2.7,3.9,1.4],
        [5.,2.,3.5,1.],
        [5.9,3.,4.2,1.5],
        [6.,2.2,4.,1.],
        [6.1,2.9,4.7,1.4],
        [5.6,2.9,3.6,1.3],
        [6.7,3.1,4.4,1.4],
        [5.6,3.,4.5,1.5],
        [5.8,2.7,4.1,1.],
        [6.2,2.2,4.5,1.5],
        [5.6,2.5,3.9,1.1],
        [5.9,3.2,4.8,1.8],
        [6.1,2.8,4.,1.3],
        [6.3,2.5,4.9,1.5],
        [6.1,2.8,4.7,1.2],
        [6.4,2.9,4.3,1.3],
        [6.6,3.,4.4,1.4],
        [6.8,2.8,4.8,1.4],
        [6.7,3.,5.,1.7],
        [6.,2.9,4.5,1.5],
        [5.7,2.6,3.5,1.],
        [5.5,2.4,3.8,1.1],
        [5.5,2.4,3.7,1.],
        [5.8,2.7,3.9,1.2],
        [6.,2.7,5.1,1.6],
        [5.4,3.,4.5,1.5],
        [6.,3.4,4.5,1.6],
        [6.7,3.1,4.7,1.5],
        [6.3,2.3,4.4,1.3],
        [5.6,3.,4.1,1.3],
        [5.5,2.5,4.,1.3],
        [5.5,2.6,4.4,1.2],
        [6.1,3.,4.6,1.4],
        [5.8,2.6,4.,1.2],
        [5.,2.3,3.3,1.],
        [5.6,2.7,4.2,1.3],
        [5.7,3.,4.2,1.2],
        [5.7,2.9,4.2,1.3],
        [6.2,2.9,4.3,1.3],
        [5.1,2.5,3.,1.1],
        [5.7,2.8,4.1,1.3],
        [6.3,3.3,6.,2.5],
        [5.8,2.7,5.1,1.9],
        [7.1,3.,5.9,2.1],
        [6.3,2.9,5.6,1.8],
        [6.5,3.,5.8,2.2],
        [7.6,3.,6.6,2.1],
        [4.9,2.5,4.5,1.7],
        [7.3,2.9,6.3,1.8],
        [6.7,2.5,5.8,1.8],
        [7.2,3.6,6.1,2.5],
        [6.5,3.2,5.1,2.],
        [6.4,2.7,5.3,1.9],
        [6.8,3.,5.5,2.1],
        [5.7,2.5,5.,2.],
        [5.8,2.8,5.1,2.4],
        [6.4,3.2,5.3,2.3],
        [6.5,3.,5.5,1.8],
        [7.7,3.8,6.7,2.2],
        [7.7,2.6,6.9,2.3],
        [6.,2.2,5.,1.5],
        [6.9,3.2,5.7,2.3],
        [5.6,2.8,4.9,2.],
        [7.7,2.8,6.7,2.],
        [6.3,2.7,4.9,1.8],
        [6.7,3.3,5.7,2.1],
        [7.2,3.2,6.,1.8],
        [6.2,2.8,4.8,1.8],
        [6.1,3.,4.9,1.8],
        [6.4,2.8,5.6,2.1],
        [7.2,3.,5.8,1.6],
        [7.4,2.8,6.1,1.9],
        [7.9,3.8,6.4,2.],
        [6.4,2.8,5.6,2.2],
        [6.3,2.8,5.1,1.5],
        [6.1,2.6,5.6,1.4],
        [7.7,3.,6.1,2.3],
        [6.3,3.4,5.6,2.4],
        [6.4,3.1,5.5,1.8],
        [6.,3.,4.8,1.8],
        [6.9,3.1,5.4,2.1],
        [6.7,3.1,5.6,2.4],
        [6.9,3.1,5.1,2.3],
        [5.8,2.7,5.1,1.9],
        [6.8,3.2,5.9,2.3],
        [6.7,3.3,5.7,2.5],
        [6.7,3.,5.2,2.3],
        [6.3,2.5,5.,1.9],
        [6.5,3.,5.2,2.],
        [6.2,3.4,5.4,2.3],
        [5.9,3.,5.1,1.8]];
        iris
    }

    #[test]
    pub fn iris_cosine() {
        let iris = iris_matrix();
        let distances = cosine_similarity_matrix(iris.t());
        eprintln!("{:?}",distances);
        panic!();
    }

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
