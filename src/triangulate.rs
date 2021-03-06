use io::Distance;
use io::Optimization;
use io::{array_mean,vec_mean};
use ndarray::{Array,Array1,Array2,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,ArrayView2,stack};
use std::sync::Arc;
use std::collections::{HashSet,HashMap,VecDeque};
use std::collections::hash_map::Entry;
use io::Parameters;
use rand::{random,Rng,thread_rng};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
use rand::distributions::Uniform;
// use weighted_sampling::WBST;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::Ordering;

pub struct SparseDistance {
    elements: Array2<f64>,
    distance_metric: Distance,
    k: usize,
    sub: f64,
    anchor: usize,
    anchor_threshold: usize,
    distances: HashMap<(usize,usize),f64>,
    rankings: HashMap<usize,Vec<usize>>,
    pub density: Array1<usize>,
    available: HashSet<usize>,

}

impl SparseDistance {
    pub fn new(elements:Array2<f64>,k:usize,sub:f64,distance_metric:Distance) -> SparseDistance {

        let n = elements.dim().0;

        let anchor = 0;
        let anchor_threshold = k * 3.max(n / 100);
        let distances = HashMap::new();
        let rankings = HashMap::new();
        let density = Array1::zeros(n);
        let available: HashSet<usize> = (0..n).collect();

        let mut sd = SparseDistance {
            elements,
            distance_metric,
            k,
            sub,
            anchor,
            anchor_threshold,
            distances,
            rankings,
            density,
            available,
        };

        sd.reanchor(0);
        sd
    }

    fn n(&self) -> usize {
        self.elements.dim().0
    }

    fn distance(&mut self,p1:usize,p2:usize) -> f64 {
        if p1 == p2 {
            return std::f64::MAX;
        }
        if self.distances.contains_key(&(p1,p2)) {
            self.distances[&(p1,p2)]
        }
        else {
            let distance = self.distance_metric.measure(
                self.elements.row(p1),
                self.elements.row(p2)
            );
            self.distances.insert((p1,p2),distance);
            self.distances.insert((p2,p1),distance);
            distance
        }

    }

    fn reanchor(&mut self, new_anchor:usize) {
        let subsample: Vec<usize> =
            rand::thread_rng()
            .sample_iter(
                &Uniform::new(0,self.n())
            )
            .take((self.n() as f64 * self.sub) as usize)
            .collect();

        let distances: Vec<f64> = subsample.iter()
            .map(|p2|
                self.distance(new_anchor,*p2)
            )
            .collect();

        let mut local_rankings = argsort(&distances);
        let mut rankings: Vec<usize> = local_rankings.into_iter().map(|i| subsample[i]).collect();
        if rankings[0] == new_anchor {
            rankings.remove(0);
        }
        self.rankings.insert(new_anchor,rankings);
        self.anchor = new_anchor;
    }

    pub fn triangulate(&mut self) -> &HashMap<usize,Vec<usize>> {

        let mut current = 0;

        while self.available.len() > 0 {
            let distance_to_anchor = self.distance(current,self.anchor);
            let worst_anchor_distance = self.distance(self.anchor,self.rankings[&self.anchor][self.anchor_threshold]);
            let worst_distance = worst_anchor_distance - distance_to_anchor;
            let neighborhood_distances: Vec<f64> =
                (0..self.anchor_threshold)
                .map(|rank| {
                    let neighbor = self.rankings[&self.anchor][rank];
                    self.distance(current,neighbor)
                })
                .collect();

            let mut neighborhood_rankings = k_min(self.k,&neighborhood_distances);

            let worst_neighbor =
                neighborhood_distances[*neighborhood_rankings.last().unwrap()];

            if worst_neighbor > worst_distance {
                self.reanchor(current)
            }
            else {
                let mut neighborhood_indices: Vec<usize> =
                    neighborhood_rankings
                    .iter()
                    .map(|n_r|
                        self.rankings[&self.anchor][*n_r]
                    )
                    .collect();

                if let Some((i,n)) = neighborhood_indices.iter().enumerate().find(|(i,n)| **n == current) {
                    neighborhood_indices.remove(i);
                }

                self.rankings.entry(current).or_insert(neighborhood_indices);
            }

            if !self.available.remove(&current) {panic!("Douple pop")};

            for next in &self.rankings[&self.anchor] {
                if self.available.contains(&next) {
                    current = *next;
                    break
                }
            }

            // eprintln!("{:?} remaining",self.available.len());
            // eprintln!("{:?}",self.rankings.len());

        }
        // eprintln!("ranked:{:?}",self.rankings)
        &self.rankings
    }

    fn estimate_density(&mut self,subsample:f64) {

        let mut rng = rand::thread_rng();

        for _ in 0..20 {

            let subsample: Vec<usize> =
                rng
                .sample_iter(
                    &Uniform::new(0,self.n())
                )
                .take((self.n() as f64 * subsample) as usize)
                .collect();

            let subset: HashSet<usize> = subsample.iter().cloned().collect();

            // eprintln!("subset:{:?}",subset);

            for sample in &subset {

                // eprintln!("sample:{:?}",sample);

                let neighbors = &self.rankings[&sample][..self.k-1];

                for neighbor in neighbors {
                    // eprintln!("neighbor:{:?}",neighbor);
                    // self.density[*neighbor] += 1;
                    if subset.contains(neighbor) {
                        self.density[*neighbor] += 1;
                    }
                }

            }

        }

        eprintln!("Density estimate {:?}",self.density);
    }
    //
    // fn estimate_density(&mut self,subsample:f64) {
    //
    //     let mut rng = rand::thread_rng();
    //
    //     for _ in 0..self.n() {
    //
    //         let mut current = rng.gen_range(0,self.n());
    //
    //         for _ in 0..1000 {
    //             self.density[[current]] += 1;
    //             let neighbors = &self.rankings[&current];
    //             current = neighbors[rng.gen_range(0,neighbors.len())];
    //         }
    //     }
    //
    //     eprintln!("Density estimate {:?}",self.density);
    // }

    fn ascend_density(&self) -> Array1<i32> {
        let mut final_index: Array1<i32> = Array1::zeros(self.n()) - 1;

        let mut current = 0;
        for i in 0..self.n() {
            current = i;
            'a: for _ in 0..self.n() {
                let neighbors = &self.rankings[&current][..self.k-1];
                let neighbor_densities = neighbors.into_iter().map(|n_i| self.density[[*n_i]]);
                let densest_neighbor = neighbors[argmax(neighbor_densities).unwrap()];
                if final_index[densest_neighbor] != -1 {
                    final_index[i] = final_index[densest_neighbor];
                    break 'a
                }
                else if self.density[densest_neighbor] <= self.density[current]{
                    final_index[i] = current as i32;
                    break 'a
                }
                else {current = densest_neighbor}
            }
            if final_index[i] == -1 {
                panic!(format!["{:?} Failed to converge!",i])
                // final_index[i] = current as i32;
            }

        }

        final_index
    }

    pub fn cluster(&mut self) -> Array1<i32> {
        eprintln!("Computing neighbors");
        self.triangulate();
        eprintln!("Estimating density");
        self.estimate_density(self.sub);
        eprintln!("Ascending");
        self.ascend_density()
    }

}


fn k_max(k:usize,vec: &[f64]) -> Vec<usize> {

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

fn k_min(k:usize,vec: &[f64]) -> Vec<usize> {

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

fn argsort(input: &[f64]) -> Vec<usize> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| x.0).collect();
    out
}

pub fn argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Less => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Greater => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        maximum = check;

    };
    maximum.map(|(i,_)| i)
}
