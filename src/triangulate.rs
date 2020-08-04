use io::Distance;
use io::Optimization;
use io::{array_mean,vec_mean};
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::sync::Arc;
use std::collections::{HashSet,HashMap,VecDeque};
use io::Parameters;
use rand::{random,Rng,thread_rng};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
// use weighted_sampling::WBST;
use std::hash::Hash;
use std::cmp::Eq;
use std::cmp::Ordering;

fn triangulate(&ArrayView2<f64>) -> Array2<usize> {
    Array
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
