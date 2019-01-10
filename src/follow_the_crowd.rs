use io::Distance;
use io::{array_mean,vec_mean};
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::sync::Arc;
use std::collections::{HashSet,HashMap,VecDeque};
use io::Parameters;
use rand::{random,Rng,thread_rng};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
use weighted_sampling::WBST;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::Ordering;

use clustering::Cluster;

#[derive(Debug)]
struct Node {
    id: usize,
    coordinates: Array<f64,Ix1>,
    neighbors: Vec<usize>,
    sampler: WBST<i64,usize>,
    parameters: Arc<Parameters>,
}



impl Node {

    pub fn new(id: usize, view: ArrayView<f64,Ix1>,parameters:Arc<Parameters>) -> Node {
        Node {
            id: id,
            coordinates: view.to_owned(),
            neighbors: vec![],
            sampler: WBST::empty(),
            parameters: parameters,
        }
    }

    pub fn connect_subsampled(&mut self, subsampled_indecies: Vec<usize>, distances: Vec<f64>) {
        let mut nearest_indecies = k_max(self.parameters.k,&distances);
        let mut nearest_neighbors: Vec<usize> = nearest_indecies.iter().map(|i| subsampled_indecies[*i]).collect();
        let mut nearest_similarities: Vec<f64> = nearest_indecies.iter().map(|i| distances[*i]).collect();
        if !nearest_neighbors.contains(&self.id) {
            nearest_neighbors.push(self.id);
            nearest_similarities.push(self.parameters.distance.measure(self.coordinates.view(),self.coordinates.view()));
        }
        self.neighbors = nearest_neighbors;
    }

}

#[derive(Debug)]
pub struct Graph {
    arena: Vec<Node>,
    population: Array<usize,Ix1>,
    points: Arc<Array<f64,Ix2>>,
    parameters: Arc<Parameters>,
    distance_matrix: Arc<Array<f64,Ix2>>,
}

impl Graph {

    pub fn new(points: Array<f64,Ix2>,mut parameters: Parameters) -> Graph {
        let shared_points = Arc::new(points);
        let distance_matrix = Arc::new(parameters.distance_matrix.take().unwrap());
        let shared_parameters = Arc::new(parameters);
        let arena: Vec<Node> =
            shared_points
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i,point)| Node::new(i,point.view(),shared_parameters.clone()))
            .collect();
        let population = Array::ones(arena.len());
        Graph {
            arena: arena,
            population: population,
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


    pub fn subsample_neighbors(&self,origin:usize) -> (Vec<usize>, Vec<f64>) {
        let indecies = self.subsampled_indices();
        let distances = indecies.iter().map(|&j| self.distance_matrix[[origin,j]]).collect();
        (indecies,distances)
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

    pub fn populations(&self) -> Array<usize,Ix1> {
        self.population.clone()
    }

    pub fn pick_a_step(&self,start:usize,rng:&mut ThreadRng) -> Option<usize> {
        let neighbor_index = self.arena[start].sampler.draw_replace_rng(rng).map(|(k,v)| k.abs() as usize);
        neighbor_index.map(|i| self.arena[start].neighbors[i])

    }

    pub fn connect(&mut self) {
        for i in 0..self.arena.len() {
            let indecies = self.subsampled_indices();
            let distances = indecies.iter().map(|&j| self.distance_matrix[[i,j]]).collect();
            self.arena[i].connect_subsampled(indecies,distances);
        }
    }

    fn update_population(&mut self,wanderers:Vec<&Wanderer>) {
        self.population.fill(1);
        for wanderer in wanderers {
            self.population[wanderer.current_node] += 1;
        }
        for node in self.arena.iter_mut() {
            let neighbors = &node.neighbors;
            let population = &self.population;
            let populations = neighbors.iter().map(|i| population[*i] as i64).collect();
            node.sampler = WBST::<i64,usize>::index_tree(&populations)
        }
    }

    pub fn wandernode(node:usize,mut graph:Graph) -> usize {
        let mut wanderer = Wanderer::new(node, graph.parameters.clone());
        let mut counter = 0;
        let mut rng = thread_rng();
        while !wanderer.converged(&graph) {
            counter += 1;
            if counter > 1000 {
                break
            }
            let next_opt = graph.pick_a_step(wanderer.current_node,&mut rng);
            if let Some(next) = next_opt {
                wanderer.directed_step(next);
            }
            graph.update_population(vec![&wanderer]);
        }
        wanderer.current_node
    }

    pub fn wanderlust(graph:&mut Graph) -> Vec<Wanderer> {
        let mut wanderers: Vec<Wanderer> = (0..graph.arena.len()).map(|i| Wanderer::new(i,graph.parameters.clone())).collect();
        let mut converged_wanderers = 0;
        let mut rng = thread_rng();
        let mut step_counter = 0;
        while converged_wanderers < wanderers.len() {
            step_counter += 1;
            if step_counter > graph.parameters.steps {
                break
            }
            converged_wanderers = 0;
            for wanderer in wanderers.iter_mut() {
                if let Some(next) = graph.pick_a_step(wanderer.current_node,&mut rng) {
                    wanderer.directed_step(next);
                    if wanderer.converged(&graph) {
                        converged_wanderers += 1;
                    }
                }
            }

        }
        wanderers
    }

    pub fn final_positions(wanderers:Vec<Wanderer>,graph: Graph) -> (Array<f64,Ix2>,Array<f64,Ix1>) {
        let mut final_positions: Array<f64,Ix2> = (*graph.points).clone();
        let mut fuzz: Array<f64,Ix1> = Array::zeros(graph.arena.len());
        for wanderer in wanderers {
            final_positions.row_mut(wanderer.origin).assign(&graph.arena[wanderer.current_node].coordinates);
            fuzz[wanderer.origin] = wanderer.fuzz(&graph).unwrap().1;
        }
        (final_positions,fuzz)
    }

    pub fn distance(&self) -> Distance {
        self.parameters.distance
    }

}


#[derive(Debug)]
pub struct Wanderer {
    origin: usize,
    current_node: usize,
    node_history: VecDeque<usize>,
    parameters: Arc<Parameters>,
}

impl Wanderer {
    pub fn new(start:usize,parameters:Arc<Parameters>) -> Wanderer {
        Wanderer {
            parameters:parameters,
            origin: start,
            current_node: start,
            node_history: VecDeque::with_capacity(51),
        }
    }

    pub fn reset(&mut self) {
        self.node_history.clear();
        self.current_node = self.origin;
    }

    pub fn directed_step(&mut self, node:usize) {
        self.current_node = node;
        self.node_history.push_back(self.current_node);
        if self.node_history.len() > 50 {
            self.node_history.pop_front();
        }
    }

    pub fn converged(&self,graph:&Graph) -> bool {
        if let Some((far,close)) = self.fuzz(graph) {
            if far/close < 2. {
                true
            }
            else {false}
        }
        else {false}
    }

    pub fn fuzz(&self,graph:&Graph) -> Option<(f64,f64)> {
        if self.node_history.len() < 50 {
            None
        }
        else {
            let current = graph.points.row(self.current_node);
            let close = graph.points.row(self.node_history[40]);
            let far = graph.points.row(self.node_history[0]);
            let close_distance = self.parameters.distance.measure(current,close);
            let far_distance = self.parameters.distance.measure(current,far);
            Some((far_distance,close_distance))
        }
    }


}

fn count_vec<'a,U: Iterator<Item=&'a T>,T:Hash + Eq>(mut collection: U) -> HashMap<&'a T,usize> {
    let mut map = HashMap::new();
    for t in collection {
        *map.entry(t).or_insert(0) += 1;
    }
    map
}

fn mode<'a,U: Iterator<Item=&'a T>,T:Hash + Eq + Clone>(collection:U) -> &'a T {
    let map = count_vec(collection);
    map.into_iter().max_by_key(|(k,c)| *c).unwrap().0
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
            insert_index = None;
            if queue.len() > k {
                queue.pop();
            }
            else if queue.len() < k {
                    insert_index = Some(queue.len())
            }
        }
    }

    queue.iter().map(|(x,y)| *x).collect()
}

// pub fn quick_weighted_pick<'a,U>(weights: U) -> Option<(usize,f64)>
// where
//     U: IntoIterator<Item=&'a f64>,
//     U::IntoIter: Clone,
//     {
//         let iter = weights.into_iter();
//         let sum = iter.clone().sum::<f64>();
//         let pick: f64 = random::<f64>() * sum;
//         iter.scan(pick,|acc,x| {*acc = *acc - *x; Some(*acc)}).enumerate().find(|(i,y)| *y < 0.).map(|(i,z)| (i,z))
// }

pub fn quick_weighted_pick<'a,U>(weights: U) -> Option<(usize,usize)>
where
    U: IntoIterator<Item=&'a usize>,
    U::IntoIter: Clone,
    {
        let iter = weights.into_iter();
        let mut sum = iter.clone().sum();
        if sum == 0 {
            return None
        }
        let pick = thread_rng().gen_range::<usize,usize,usize>(1,sum);

    for (i,weight) in iter.enumerate() {
        sum -= weight;
        if sum <= pick {
            return Some((i,*weight))
        }
    }
    return None
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    pub fn kmax() {

        assert_eq!(vec![0,1,2],k_max(3,&vec![10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.]));
        assert_eq!(vec![4,5,6],k_max(3,&vec![7.,6.,5.,4.,10.,9.,8.,3.,2.,1.,0.]));
    }

    #[test]
    pub fn quick_weighted_pick() {

        assert_eq!(vec![0,1,2],k_max(3,&vec![10.,9.,8.,7.,6.,5.,4.,3.,2.,1.,0.]));
        assert_eq!(vec![4,5,6],k_max(3,&vec![7.,6.,5.,4.,10.,9.,8.,3.,2.,1.,0.]));
    }

}
