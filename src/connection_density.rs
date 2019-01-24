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


use rayon::prelude::*;

#[derive(Debug,Clone)]
struct Node {
    id: usize,
    // coordinates: Array<f64,Ix1>,
    neighbors: Vec<usize>,
    sampler: WBST<i64,usize>,
    parameters: Arc<Parameters>,
}



impl Node {

    pub fn new(id: usize,parameters:Arc<Parameters>) -> Node {
        Node {
            id: id,
            neighbors: vec![],
            sampler: WBST::empty(),
            parameters: parameters,
        }
    }

    pub fn connect_subsampled(&mut self, mut subsampled_indecies: Vec<usize>, mut distances: Vec<f64>) {
        let mut nearest_indecies = k_max(self.parameters.k,&distances);
        let mut nearest_neighbors: Vec<usize> = nearest_indecies.iter().map(|i| subsampled_indecies[*i]).collect();
        let mut nearest_similarities: Vec<f64> = nearest_indecies.iter().map(|i| distances[*i]).collect();
        self.neighbors = nearest_neighbors;
    }

}

#[derive(Debug,Clone)]
pub struct Graph {
    arena: Vec<Node>,
    density: Array<usize,Ix1>,
    // points: Arc<Array<f64,Ix2>>,
    pub parameters: Arc<Parameters>,
    distance_matrix: Arc<Array<f64,Ix2>>,
}

impl Graph {

    pub fn new(mut parameters: Parameters) -> Graph {
        let distance_matrix = Arc::new(parameters.distance_matrix.take().unwrap());
        let shared_parameters = Arc::new(parameters);
        let arena: Vec<Node> = (0..distance_matrix.shape()[0]).map(|i| Node::new(i,shared_parameters.clone()))
            .collect();
        let density = Array::zeros(arena.len());
        eprintln!("Graph created:{}",arena.len());
        Graph {
            arena: arena,
            density: density,
            // points: shared_points,
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

    pub fn density(&self) -> &Array<usize,Ix1> {
        &self.density
    }

    // pub fn peaks_to_valleys(&self) -> Vec<usize> {
    //     let mut available_points: HashSet<usize> = (0..self.len()).collect();
    //     let mut peak_assignments = vec![(0,0);self.len()];
    //     loop {
    //         let mut available_densities:Vec<(usize,usize)> = available_points.iter().map(|i| (*i,self.density[*i])).collect();
    //         let best_peak = available_densities.iter().max_by_key(|(i,d)| *d).map(|x| *x);
    //         if let Some((peak_index,peak_density)) = best_peak {
    //             eprintln!("Best peak:{},{}",peak_index,peak_density);
    //             available_points.remove(&peak_index);
    //             let peak_children = self.downhill(peak_index);
    //             eprintln!("REMOVING:{:?}",peak_children);
    //             for child in peak_children {
    //                 available_points.remove(&child);
    //                 if peak_assignments[child].1 < peak_density {
    //                     peak_assignments[child] = (peak_index,peak_density);
    //                 }
    //             }
    //         }
    //         if available_points.len() > 0 {
    //             eprintln!("Undescended points: {}", available_points.len());
    //         }
    //         else {
    //             break
    //         }
    //     }
    //     peak_assignments.into_iter().map(|(i,d)| i).collect()
    // }

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


    pub fn connect(&mut self) {
        for i in 0..self.arena.len() {
            let indecies = self.subsampled_indices();
            let distances = indecies.iter().map(|&j| self.distance_matrix[[i,j]]).collect();
            self.arena[i].connect_subsampled(indecies,distances);
        }
    }

    pub fn wanderlust(prototype:Arc<Graph>) -> Vec<Wanderer> {
        let mut wanderers: Vec<Wanderer> = (0..prototype.arena.len()).map(|i| Wanderer::new(i,prototype.parameters.clone())).collect();
        let mut local_prototype: Graph = (*prototype).clone();
        local_prototype.connect();
        // local_prototype.set_samplers();
        // let mut rng = thread_rng();
        let mut graph: Result<Graph,Arc<Graph>> = Ok(local_prototype);
        let mut converged = 0;
        let mut steps = 0;
        while converged < wanderers.len() {
            steps += 1;
            if steps > 1000 {
                break
            }
            // eprintln!("Step {}, {} converged", steps,converged);
            converged = 0;
            let mut local_graph = graph.ok().take().unwrap();
            local_graph.connect();
            let arc_graph = Arc::new(local_graph);
            for wanderer in wanderers.iter_mut() {
                if !wanderer.step(arc_graph.clone()).is_some() {converged += 1};
                // if !wanderer.stochastic_step(arc_graph.clone(),&mut rng).is_some() {converged += 1};
            }
            graph = Arc::try_unwrap(arc_graph)
        }
        wanderers

    }

    pub fn fuzzy_cluster(prototype:&mut Graph) -> Vec<usize> {
        let arc_graph = Arc::new(prototype.clone());
        let metavec: Vec<(usize,Vec<usize>)> = (0..20usize).into_par_iter().map(|i| {
            let wanderers = Graph::wanderlust(arc_graph.clone());
            eprintln!("Cluster repeat {:?}",i);
            (i,wanderers.iter().map(|w| w.current()).collect())
        }).collect();
        let mut fuzzy_landing: Array<usize,Ix2> = Array::zeros((20,arc_graph.len()));
        for (i,clusters) in metavec.into_iter() {
            fuzzy_landing.row_mut(i).assign(&mut Array::from_vec(clusters));
        }
        fuzzy_landing.t().axis_iter(Axis(0)).map(|row| *mode(row.iter())).collect()
    }


    pub fn pick_step(&self, start:usize,rng:&mut ThreadRng) -> Option<usize> {
        let neighbor_index = &self.arena[start].sampler.draw_replace_rng(rng);
        let pick = neighbor_index.map(|(weight,index)| self.arena[start].neighbors[index]);
        pick
    }

    pub fn highest_density_neighbor(&self,start:usize) -> Option<usize> {
        let max_neighbor_index = self.arena[start].neighbors
            .iter()
            .map(|&i| self.density[i])
            .enumerate()
            .max_by_key(|x| x.1)
            .map(|(i,d)| i);
        max_neighbor_index.map(|i| self.arena[start].neighbors[i])
    }

    pub fn stochastic_weighted_neighbor(&self,start:usize,rng:&mut ThreadRng) -> Option<usize> {
        let neighbor_index = self.arena[start].sampler.draw_replace_rng(rng);
        neighbor_index.map(|(w,i)| self.arena[start].neighbors[i])
    }

    pub fn distance(&self) -> Distance {
        self.parameters.distance
    }


    pub fn history_mode(wanderers:&Vec<Wanderer>) -> Vec<usize> {
        wanderers.iter().map(|w| mode(w.density_history.iter()).0).collect()
    }

    pub fn rapid_connection(&mut self) {
        let mut connectivity = Array::zeros(self.arena.len());
        for i in 0..self.parameters.steps {
            eprintln!("Running rapid connections: {}",i);
            self.connect();
            for node in &self.arena {
                for neighbor in &node.neighbors {
                    connectivity[*neighbor] += 1;
                }
            }
        }

        self.density = connectivity;

    }

    pub fn set_samplers(&mut self) {
        let densities = &self.density;
        for node in self.arena.iter_mut() {
            let mut neighbor_densities = node.neighbors.iter().map(|i| densities[*i] as i64).collect();
            node.sampler = WBST::<i64,usize>::index_tree(&neighbor_densities);
        }
    }

    pub fn smooth_density(&mut self) -> Array<usize,Ix1> {
        let mut new_density: Array<usize,Ix1> = Array::zeros(self.arena.len());
        for node in &self.arena {
            let neighbor_densities: Vec<usize> = node.neighbors.iter().map(|i| self.density[*i]).collect();
            new_density[node.id] = neighbor_densities.iter().sum::<usize>()/neighbor_densities.len();
        }
        new_density
    }


}

const HISTORY_SIZE: usize = 50;

#[derive(Debug,Clone)]
pub struct Wanderer {
    origin: usize,
    current_node: usize,
    density_history: VecDeque<(usize,usize)>,
    max_density: usize,
    parameters: Arc<Parameters>,
}

impl Wanderer {
    pub fn new(start:usize,parameters:Arc<Parameters>) -> Wanderer {
        Wanderer {
            parameters:parameters,
            origin: start,
            current_node: start,
            density_history: VecDeque::with_capacity(HISTORY_SIZE + 1),
            max_density: 0,
        }
    }

    pub fn reset(&mut self) {
        self.density_history.clear();
        self.current_node = self.origin;
    }

    pub fn step(&mut self, graph: Arc<Graph>) -> Option<usize> {

        if let None = self.converged(&graph) {
            let step = graph.highest_density_neighbor(self.current_node);
            if let Some(next) = step {
                self.current_node = next;
                let density = graph.density[self.current_node];
                self.density_history.push_back((self.current_node,density));
                if self.density_history.len() > HISTORY_SIZE {
                    self.density_history.pop_front();
                }
                if self.max_density < density { self.max_density = density};
            }
            step
        }
        else { None }

    }

    pub fn stochastic_step(&mut self, graph: Arc<Graph>,rng:&mut ThreadRng) -> Option<usize> {

        if let None = self.converged(&graph) {
            let step = graph.stochastic_weighted_neighbor(self.current_node, rng);
            if let Some(next) = step {
                self.current_node = next;
                let density = graph.density[self.current_node];
                self.density_history.push_back((self.current_node,graph.density[self.current_node]));
                if self.density_history.len() > HISTORY_SIZE {
                    self.density_history.pop_front();
                }
                if self.max_density < density { self.max_density = density};
            }
            step
        }
        else { None }

    }

    pub fn current(&self) -> usize {
        self.current_node
    }

    pub fn converged(&self,graph:&Graph) -> Option<usize> {

        if self.density_history.len() < HISTORY_SIZE {
            return None
        }

        if self.max_density > self.density_history[0].1 {
            None
        }
        else {
            self.density_history.get(0).map(|(i,d)| *d)
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

fn significant_mode<'a,U: Iterator<Item=&'a T>,T:Hash + Eq + Clone>(collection:U) -> Option<&'a T> {
    let map = count_vec(collection);
    let total = map.values().sum::<usize>();
    let max_key = map.into_iter().max_by_key(|(k,c)| *c).unwrap();
    if max_key.1 > (total/5) {
        Some(max_key.0)
    }
    else { None }

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

pub fn average_repeats(repeats:&Vec<Array<f64,Ix2>>) -> Array<f64,Ix2> {
    let mut dimensions = repeats.iter().map(|r| r.shape().to_vec()).collect::<Vec<Vec<usize>>>();
    dimensions.dedup();
    assert!(dimensions.len() == 1);
    let mut outer: Array<f64,Ix3> = Array::zeros((repeats.len(),dimensions[0][0],dimensions[0][1]));
    for (i,mut slice) in outer.axis_iter_mut(Axis(0)).enumerate() {
        slice.assign(&repeats[i])
    }
    let mean = outer.mean_axis(Axis(0));
    mean
}

pub fn repeat_fuzz(repeats: &Vec<Array<f64,Ix2>>, average: &Array<f64,Ix2>,distance: Distance) -> Array<f64,Ix1> {
    let mut dimensions = repeats.iter().map(|r| r.shape().to_vec()).collect::<Vec<Vec<usize>>>();
    dimensions.dedup();
    assert!(dimensions.len() == 1);
    let mut deviations = Array::zeros((dimensions[0][0],repeats.len()));
    for i in 0..repeats.len() {
        for j in 0..dimensions[0][0] {
            deviations[[j,i]] = distance.measure(repeats[i].row(j),average.row(j))
        }
    }
    deviations.mean_axis(Axis(1))
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
