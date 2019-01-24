// use std::sync::Arc;
// use std::collections::HashSet;
// use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
// use std::f64;
// use std::cmp::{max,Ordering};
// use rayon::prelude::*;
//
// use io::Parameters;
//
// use io::Distance;
//
// #[derive(Clone)]
// pub struct Cluster {
//     pub id: usize,
//     pub weight: usize,
//     pub radius: f64,
//     pub center: Array<f64,Ix1>,
//     pub members: Vec<usize>,
//     pub array: Arc<Array<f64,Ix2>>,
//     distance: Distance
// }
//
// impl Cluster {
//     pub fn init(id: usize, points: Arc<Array<f64,Ix2>>, point_id: usize, distance: Distance) -> Cluster {
//         let point = points.row(point_id);
//         let mut array = Array::zeros((1,point.shape()[0]));
//         array.row_mut(0).assign(&point);
//         Cluster {
//             id: id,
//             weight: 1,
//             radius: 0.0,
//             center: point.to_owned(),
//             members: vec![point_id],
//             array: points.clone(),
//             distance: distance
//         }
//     }
//
//     pub fn merge_cluster(&self,cluster: &Cluster) -> Cluster {
//         let new_weight = self.weight + cluster.weight;
//         // let new_center = ((&self.center * self.weight as f64) + (&cluster.center * cluster.weight as f64)) / (new_weight) as f64;
//         let new_members = [self.members.iter(),cluster.members.iter()].iter().flat_map(|x| x.clone()).cloned().collect();
//         let mut new_cluster = Cluster {
//             id : self.id,
//             weight: new_weight,
//             radius: 0.0,
//             center: self.center.clone(),
//             members: new_members,
//             array: self.array.clone(),
//             distance: self.distance
//         };
//         new_cluster.center = new_cluster.center();
//         new_cluster.radius = new_cluster.radius();
//         new_cluster
//     }
//
//     pub fn merge_point(&mut self ,point: ArrayView<f64,Ix1>, point_id:usize) -> usize {
//
//         self.center *= self.weight as f64/(self.weight as f64 + 1.);
//         self.center.scaled_add(1./(self.weight as f64 + 1.), &point);
//         self.weight += 1;
//         self.members.push(point_id);
//         self.radius = self.radius();
//         self.id
//     }
//
//     pub fn radius(&self) -> f64 {
//         let radius = self.members.iter().map(|x| self.distance.measure(self.array.row(*x).view(), self.center.view())).sum::<f64>() / self.weight as f64;
//         // eprintln!("R:{:?}",radius);
//         radius
//     }
//
//     pub fn center(&self) -> Array<f64,Ix1> {
//         let mut center = Array::zeros(self.array.shape()[1]);
//         for i in self.members.iter() {
//             center += &(&self.array.row(*i) / self.weight as f64);
//         }
//         center
//     }
//
//     pub fn empty(&mut self) {
//         self.weight = 0;
//         self.radius = 0.;
//         self.members = vec![];
//     }
//
//     pub fn clusters_to_labels(clusters:Vec<Cluster>) -> Vec<usize> {
//         let total_points = clusters.iter().fold(0,|acc,c| acc + c.members.len());
//         let mut labels = vec![0;total_points];
//         for cluster in clusters {
//             for point in cluster.members {
//                 labels[point] = cluster.id;
//             }
//         }
//         labels
//     }
//
//     pub fn quick_cluster(final_positions_raw: Array<f64,Ix2>, fuzz_raw:Array<f64,Ix1>, distance:Distance) -> Vec<usize> {
//         let final_positions = Arc::new(final_positions_raw);
//         let fuzz = Arc::new(fuzz_raw);
//         let mut clusters = Cluster::cluster_points(final_positions.clone(), fuzz.clone(), distance, vec![]);
//         Cluster::clusters_to_labels(clusters)
//     }
//
//     pub fn cluster_points(final_positions: Arc<Array<f64,Ix2>>, fuzz:Arc<Array<f64,Ix1>>, distance:Distance, mut clusters: Vec<Cluster>) -> Vec<Cluster> {
//
//         // eprintln!("{:?}",final_positions.axis_iter(Axis(0)).enumerate().collect::<Vec<(usize,ArrayView<f64,Ix1>)>>());
//         // eprintln!("{:?}",fuzz.iter().enumerate().collect::<Vec<(usize,&f64)>>());
//
//         // let final_positions = Arc::new(final_positions_raw);
//
//         let mut available_points: HashSet<usize> = (0..final_positions.shape()[0]).into_iter().collect();
//
//         while available_points.len() > 0 {
//
//             let mut moved_points = vec![];
//
//             for point_index in available_points.iter().cloned() {
//
//                 let point = final_positions.row(point_index);
//
//                 let mut distances_to_clusters = vec![];
//
//                 for (i,cluster) in clusters.iter().enumerate() {
//
//                     distances_to_clusters.push((i,distance.measure(point, cluster.center.view())));
//
//                     // distances_to_clusters.push((i,length((&point - &cluster.center).view())));
//
//                     // if self.parameters.distance.unwrap_or(Distance::Cosine).measure(point,cluster.center.view()) < (cluster.radius + self.fuzz[point_index]) {
//                     // // if distance(point,cluster.center.view()) < self.parameters.scaling_factor.unwrap_or(0.1) * self.parameters.convergence_factor.unwrap_or(5.){
//                     //     cluster.merge_point(point,point_index);
//                     //     moved_points.push(point_index);
//                     //     break
//                     // }
//                 }
//
//                 let best_cluster_option = distances_to_clusters.iter().min_by(|x,y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Greater));
//
//                 // eprintln!("PP:{:?}",point);
//                 // eprintln!("FF:{:?}",self.fuzz[point_index]);
//                 // eprintln!("BCO:{:?}", best_cluster_option);
//                 // eprintln!("BCR:{:?}",self.clusters[best_cluster_option.unwrap().0].radius);
//
//                 if let Some((best_cluster_index,distance_to_cluster)) = best_cluster_option {
//                     let best_cluster: &mut Cluster = &mut clusters[*best_cluster_index];
//                     // eprintln!("Try");
//                     // eprintln!("ID:{:?}",best_cluster.id);
//                     // eprintln!("FF:{:?}",self.fuzz[point_index]);
//                     // eprintln!("CM:{:?}",distance_to_cluster);
//                     // eprintln!("DS:{:?}",displacement);
//                     // eprintln!("CC:{:?}",best_cluster.center);
//                     // eprintln!("PC:{:?}",point);
//
//                     // if *distance_to_cluster < (best_cluster.radius + self.fuzz[point_index]) * 2. {
//                     if *distance_to_cluster < (best_cluster.radius + (fuzz[point_index] * 3.)) {
//                         moved_points.push(point_index);
//                         best_cluster.merge_point(point,point_index);
//                         // eprintln!("ID:{:?}",best_cluster.id);
//                         // eprintln!("FF:{:?}",self.fuzz[point_index]);
//                         // eprintln!("CR:{:?}",best_cluster.radius);
//                         // eprintln!("CM:{:?}",distance_to_cluster);
//                         // eprintln!("DS:{:?}",displacement);
//                         // eprintln!("CC:{:?}",best_cluster.center);
//                         // eprintln!("PC:{:?}",point);
//                     }
//                 }
//
//             }
//
//
//             for point in &moved_points {
//                 available_points.remove(point);
//             }
//
//             if moved_points.len() < 1 {
//                 let best_cluster_candidate = available_points.iter().min_by(|&&x,&&y| fuzz[[x]].partial_cmp(&fuzz[y]).unwrap_or(Ordering::Greater)).map(|x| *x);
//                 if let Some(new_cluster_point) = best_cluster_candidate {
//                     available_points.remove(&new_cluster_point);
//                     let new_cluster = Cluster::init(clusters.len()+1, final_positions.clone(), new_cluster_point,distance);
//                     clusters.push(new_cluster);
//                 }
//                 else {
//                     break
//                 }
//             }
//
//             // eprintln!("Unclustered:{:?}",available_points.len());
//             // eprintln!("Clusters:{:?}",clusters.len());
//
//         }
//
//
//         eprintln!("Coarse clusters: {:?}", clusters.len());
//
//         clusters
//     }
//
//     pub fn recluster(mut clusters: Vec<Cluster>, final_positions: Arc<Array<f64,Ix2>>, fuzz:Arc<Array<f64,Ix1>>, distance:Distance) -> Vec<Cluster> {
//         for cluster in clusters.iter_mut() {
//             cluster.empty();
//         }
//
//         Cluster::cluster_points(final_positions, fuzz, distance, clusters)
//     }
//
// //
// //     pub fn merge_clusters(mut clusters: Vec<Cluster>, final_positions: Arc<Array<f64,Ix2>>,distance:Distance) -> Vec<Cluster> {
// //
// //     loop {
// //
// //         let mut merge_candidates: Option<(usize,usize)> = None;
// //
// //         'i_loop: for i in 0..clusters.len() {
// //
// //
// //             let c1 = &clusters[i];
// //
// //             'j_loop: for j in 0..clusters.len() {
// //                 if i != j {
// //
// //                     let c2 = &clusters[j];
// //
// //                     // if self.parameters.distance.unwrap_or(Distance::Cosine).measure(c1.center.view(),c2.center.view()) < (c1.radius + c2.radius)*2. {
// //                     //     eprintln!("Failed");
// //                     //     eprintln!("C1:{:?}",c1.center);
// //                     //     eprintln!("C2:{:?}",c2.center);
// //                     //     eprintln!("R1:{:?}",c1.radius);
// //                     //     eprintln!("R2:{:?}",c2.radius);
// //                     //     eprintln!("Merging:{:?}",merge_candidates);
// //                     //     eprintln!("Distance:{:?}",self.parameters.distance.unwrap_or(Distance::Cosine).measure(c1.center.view(),c2.center.view()));
// //                     // }
// //
// //                     // if distance.measure(c1.center.view(),c2.center.view()) < (c1.radius.sqrt() + c2.radius.sqrt()).powi(2) {
// //                     if distance.measure(c1.center.view(),c2.center.view()) < (c1.radius + c2.radius) {
// //                         merge_candidates = Some((i,j));
// //                         // eprintln!("C1:{:?}",c1.center);
// //                         // eprintln!("C2:{:?}",c2.center);
// //                         // eprintln!("R1:{:?}",c1.radius);
// //                         // eprintln!("R2:{:?}",c2.radius);
// //                         // eprintln!("W1:{:?}",c1.weight);
// //                         // eprintln!("W2:{:?}",c2.weight);
// //                         // eprintln!("Merging:{:?}",merge_candidates);
// //                         break 'i_loop;
// //                     }
// //                 }
// //             }
// //
// //         }
// //
// //         if let Some((c1i,c2i)) = merge_candidates {
// //             let new_cluster = clusters[c1i].merge_cluster(&clusters[c2i]);
// //             // eprintln!("N:{:?}",new_cluster.center);
// //             // eprintln!("N:{:?}",new_cluster.radius);
// //             clusters[c1i] = new_cluster;
// //             clusters.remove(c2i);
// //         }
// //         else {
// //             break
// //         }
// //
// //     }
// //
// //     eprintln!("Merged Clusters: {:?}",clusters.len());
// //
// //     clusters
// // }
//
//
//
// }
