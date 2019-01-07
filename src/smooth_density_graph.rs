use io::Distance;
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::sync::Arc;
use std::collections::{HashMap,VecDeque};
use io::Parameters;
use rand::{Rng,thread_rng};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
use weighted_sampling::WBST;
use std::hash::Hash;
use std::cmp::Eq;

#[derive(Debug)]
struct Node {
    id: usize,
    coordinates: Array<f64,Ix1>,
    neighbors: Vec<usize>,
    distances: Vec<f64>,
    sampler: WBST<f64,usize>,
    parameters: Arc<Parameters>,
}

impl Node {

    pub fn new(id: usize, view: ArrayView<f64,Ix1>,parameters:Arc<Parameters>) -> Node {
        Node {
            id: id,
            coordinates: view.to_owned(),
            neighbors: vec![],
            distances: vec![],
            sampler: WBST::empty(),
            parameters: parameters,
        }
    }

    pub fn connect_subsampled(&mut self, subsampled_indecies: Vec<usize>, distances: Vec<f64>) {
        let nearest_indecies = k_max(self.parameters.k,&distances);
        let nearest_neighbors = nearest_indecies.iter().map(|i| subsampled_indecies[*i]).collect();
        let mut nearest_distances: Vec<f64> = nearest_indecies.iter().map(|i| distances[*i]).collect();
        for d in nearest_distances.iter_mut() {
                *d = 1. / *d;
                if d.is_nan() {
                    *d = 0.
                }
        }
        self.sampler = WBST::<f64,usize>::index_tree(&nearest_distances);
        self.neighbors = nearest_neighbors;
        self.distances = nearest_distances;
    }

    pub fn pick_a_neighbor(&self, rng: &mut ThreadRng) -> usize {
        let index = self.sampler.draw_replace_rng(rng).unwrap().1;
        self.neighbors[index]
    }
}

pub struct Graph {
    arena: Vec<Node>,
    points: Arc<Array<f64,Ix2>>,
    parameters: Arc<Parameters>,
    distance_matrix: Arc<Array<f64,Ix2>>,
}

impl Graph {

    pub fn new(points: Array<f64,Ix2>,mut parameters: Parameters) -> Graph {
        let shared_points = Arc::new(points);
        let distance_matrix = Arc::new(parameters.distance_matrix.take().unwrap());
        let shared_parameters = Arc::new(parameters);
        let arena =
            shared_points
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i,point)| Node::new(i,point.view(),shared_parameters.clone()))
            .collect();
        Graph {
            arena: arena,
            points: shared_points,
            distance_matrix: distance_matrix,
            parameters: shared_parameters,
        }
    }

    pub fn len(&self) -> usize{
        self.arena.len()
    }

    pub fn subsampled_indices(&self) -> Vec<usize> {
        sample(&mut thread_rng(), self.len(), self.parameters.subsample).into_vec()
    }

    pub fn connectivity(&self) -> Array<bool,Ix2> {
        let mut connectivity = Array::default((self.arena.len(),self.arena.len()));
        connectivity.fill(false);
        for (i,node) in self.arena.iter().enumerate() {
            for target in &node.neighbors {
                connectivity[[i,*target]] = true;
            }
        }
        connectivity
    }

    pub fn connect(&mut self) {
        for i in 0..self.arena.len() {
            let indecies = self.subsampled_indices();
            let distances = indecies.iter().map(|&j| self.distance_matrix[[i,j]]).collect();
            self.arena[i].connect_subsampled(indecies,distances);
        }
    }

    pub fn wandernode(node:usize,graph:Graph) -> usize {
        let graph_arc = Arc::new(graph);
        let mut wanderer = Wanderer::new(node, graph_arc);
        wanderer.wander()
    }

    pub fn wanderlust(graph:Graph) -> Vec<usize> {
        let graph_arc = Arc::new(graph);
        (0..graph_arc.len())
            .map(|node|
                {
                    let mut wanderer = Wanderer::new(node, graph_arc.clone());
                    wanderer.wander()
                }
            )
            .collect()
    }

}

struct Wanderer {
    origin: usize,
    current_node: usize,
    graph: Arc<Graph>,
    node_history: VecDeque<usize>,
    parameters: Arc<Parameters>,
}

impl Wanderer {
    pub fn new(start:usize,graph: Arc<Graph>) -> Wanderer {
        Wanderer {
            parameters:graph.parameters.clone(),
            origin: start,
            current_node: start,
            graph: graph,
            node_history: VecDeque::with_capacity(51),
        }
    }

    pub fn reset(&mut self) {
        self.node_history.clear();
        self.current_node = self.origin;
    }

    pub fn step(&mut self,rng:&mut ThreadRng) {
        let node = &self.graph.arena[self.current_node];
        self.current_node = node.pick_a_neighbor(rng);
        self.node_history.push_back(self.current_node);
        if self.node_history.len() > 50 {
            self.node_history.pop_front();
        }
    }

    pub fn converged(&self) -> bool {
        if self.node_history.len() < 50 {
            false
        }
        else {
            let current = self.graph.points.row(self.current_node);
            let close = self.graph.points.row(self.node_history[40]);
            let far = self.graph.points.row(self.node_history[0]);
            let close_distance = self.parameters.distance.measure(current,close);
            let far_distance = self.parameters.distance.measure(current,far);
            if far_distance / close_distance < 2. {
                true
            }
            else {false}
        }

    }

    pub fn wander(&mut self) -> usize {
        let mut counter = 0;
        let mut failcounter = 0;
        let mut rng = thread_rng();
        while !self.converged() {
            self.step(&mut rng);
            counter += 1;
            if counter > 10000 {
                if failcounter > 5 {
                    eprintln!("{:?}",self.graph.distance_matrix);
                    eprintln!("{:?}",self.graph.arena[self.current_node]);
                    eprintln!("{:?}",self.node_history);
                    panic!("Not all who wander are lost, but I sure as fuck am");
                }
                failcounter += 1;
                counter = 0;
                self.reset()
            }
        }
        let counts = count_vec(self.node_history.iter().collect());
        *counts.into_iter().max_by_key(|(k,c)| *c).unwrap().0
    }
}

fn count_vec<T:Hash + Eq>(vec:Vec<T>) -> HashMap<T,usize> {
    let mut map = HashMap::new();
    for t in vec {
        *map.entry(t).or_insert(0) += 1;
    }
    map
}

fn k_nearest_neighbors(center: ArrayView<f64,Ix1>,n: usize,points: Arc<Array<f64,Ix2>>,distance:Distance) -> Vec<(usize,f64)> {

    if points.rows() < 1 {
        return vec![]
    }

    let mut neighbors: Vec<(ArrayView<f64,Ix1>,usize,f64)> = Vec::with_capacity(n+1);

    neighbors.push((points.row(0),0,distance.measure(center, points.row(0))));

    for i in 0..points.rows() {

        let point = points.row(i);
        let distance = distance.measure(center,point);

        let mut insert_index = None;

        for (i,(previous_point,previous_index,previous_distance)) in neighbors.iter().enumerate() {
            if distance < *previous_distance {
                insert_index = Some(i);
                break
            }
        }

        if let Some(insert) = insert_index {
            neighbors.insert(insert,(point,i,distance));
        }

        neighbors.truncate(n+1);

    }

    neighbors.into_iter().map(|(p,i,d)| (i,d)).collect()

}

fn k_max(k:usize,vec: &Vec<f64>) -> Vec<usize> {

    let mut queue = Vec::with_capacity(k+1);
    let mut insert_index = Some(0);
    for (i,x) in vec.iter().enumerate() {
        for (j,(_,y)) in queue.iter().enumerate().rev() {
            if x < y {
                break
            }
            else {
                insert_index = Some(j)
            }
        }
        if let Some(valid_index) = insert_index {
            queue.insert(valid_index,(i,*x));
        }
    }

    queue.iter().map(|(x,y)| *x).collect()
}

fn k_min(k:usize,vec: &Vec<f64>) -> Vec<usize> {
    let mut queue = Vec::with_capacity(k+1);
    let mut insert_index = Some(0);
    for (i,x) in vec.iter().enumerate() {
        for (j,(_,y)) in queue.iter().enumerate().rev() {
            if x > y {
                break
            }
            else {
                insert_index = Some(j)
            }
        }
        if let Some(valid_index) = insert_index {
            queue.insert(valid_index,(i,*x));
        }
    }

    queue.iter().map(|(x,y)| *x).collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    pub fn microtest() {
    }


}
