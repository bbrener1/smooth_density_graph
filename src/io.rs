use std::fs::File;
use std::fs::OpenOptions;
use std::io::Error;

use std::io;
use std::io::prelude::*;
use std::collections::HashMap;
use num_cpus;
use std::f64;
use std::fmt::Debug;
use rayon::iter::IntoParallelIterator;
use std::cmp::Ordering;

use ndarray::{Array,ArrayView,Ix1,Ix2,Axis};
// use ndarray_linalg::*;

#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    pub command: Command,
    pub counts: Option<Array<f64,Ix2>>,
    pub distance_matrix: Option<Array<f64,Ix2>>,
    pub report_address: Option<String>,
    pub dump_error: Option<String>,

    pub steps: usize,
    pub k: usize,
    pub subsample: usize,
    pub distance: Distance,

    count_array_file: String,
    distance_matrix_file: String,

    processor_limit: Option<usize>,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            command: Command::FitPredict,
            count_array_file: "".to_string(),
            distance_matrix_file: "".to_string(),
            counts: None,
            distance_matrix: None,
            report_address: None,
            dump_error: None,
            distance: Distance::Cosine,

            steps: 0,
            k: 0,
            subsample:0,

            processor_limit: None,

        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        eprintln!("Reading parameters");

        let mut arg_struct = Parameters::empty();

        let _raw_command = args.next();

        arg_struct.command = Command::parse(&args.next().expect("Please enter a command"));

        let mut _supress_warnings = false;

        while let Some((i,arg)) = args.enumerate().next() {

                match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        eprintln!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                _supress_warnings = true;
                },
                "-auto" | "-a"=> {
                    arg_struct.auto = true;
                    arg_struct.auto()
                },
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.counts = Some(read_counts(&arg_struct.count_array_file))
                },
                "-stdin" => {
                    arg_struct.counts = Some(read_standard_in());
                }
                "-stdout" => {
                    arg_struct.report_address = None;
                }
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = Some(args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit"));
                },
                "-o" | "-output" => {
                    arg_struct.report_address = Some(args.next().expect("Error processing output destination"))
                },
                "-error" => {
                    arg_struct.dump_error = Some(args.next().expect("Error processing error destination"))
                },
                "-k" | "-locality" => {
                    arg_struct.k = args.next().map(|x| x.parse::<usize>()).expect("k parse error. Not a number?").expect("Iteration error")
                },
                "-steps" => {
                    arg_struct.steps = args.next().map(|x| x.parse::<usize>()).expect("step parse error. Not a number?").expect("Iteration error")
                },
                "-d" | "-distance" => {
                    arg_struct.distance = args.next().map(|x| Distance::parse(&x)).expect("Distance parse error")
                },
                "-dm" | "-distance_matrix" => {
                    arg_struct.distance_matrix_file = args.next().expect("Error parsing count location!");
                    arg_struct.distance_matrix = Some(read_counts(&arg_struct.distance_matrix_file));
                },
                "-ss" | "-subsample" => {
                    arg_struct.subsample = args.next().expect("Error iterating subsample").parse::<usize>().expect("Error parsing subsample, not a number?")
                },

                &_ => {
                    panic!("Not a valid argument: {}", arg);
                }

            }
        }

        arg_struct

    }


    fn auto(&mut self) {

        let mtx_o = self.counts.iter().chain(self.distance_matrix.iter()).next();
        let mtx: &Array<f64,Ix2> = mtx_o.expect("Please specify counts file before the \"-auto\" argument.");

        let processors = num_cpus::get();

        let subsample = mtx.shape()[0] / 4;

        let k = (((mtx.shape()[0] as f64).log10() * 2.) + 1.) as usize;

        self.auto = true;

        self.processor_limit.get_or_insert( processors );
        self.subsample = subsample;
        self.k = k;
        self.steps = 10;

    }

    pub fn summary(&self) {
        eprintln!("Parameter summary:");
        eprintln!("k:{:?}",self.k);
        eprintln!("sub:{:?}",self.subsample);
        eprintln!("steps:{:?}",self.steps);    
    }

    pub fn distance(&self, p1:ArrayView<f64,Ix1>,p2:ArrayView<f64,Ix1>) -> f64 {
        self.distance.measure(p1,p2)
    }

    pub fn from_vec(shape:(usize,usize),vec:Vec<f64>,k:usize,subsample:usize) -> Parameters {
        let mut p = Parameters::empty();
        p.counts = Some(Array::from_shape_vec(shape,vec).unwrap());
        p.k = k;
        p.subsample = subsample;
        p
    }

}


fn read_header(location: &str) -> Vec<String> {

    eprintln!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let feature = line.unwrap_or("error".to_string());
        let mut renamed = feature.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [feature.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    eprintln!("Read {} lines", header_vector.len());

    header_vector
}

fn read_sample_names(location: &str) -> Vec<String> {

    let mut header_vector = Vec::new();

    let sample_name_file = File::open(location).expect("Sample name file error!");
    let mut sample_name_lines = io::BufReader::new(&sample_name_file).lines();

    for line in sample_name_lines.by_ref() {
        header_vector.push(line.expect("Error reading header line!").trim().to_string())
    }

    header_vector
}



fn read_counts(location:&str) -> Array<f64,Ix2> {


    let count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (i,line) in count_array_lines.by_ref().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        let gene_line = line.expect("Readline error");

        for (j,gene) in gene_line.split_whitespace().enumerate() {

            if j == 0 && i%200==0{
                eprint!("\n");
            }

            if i%200==0 && j%200 == 0 {
                eprint!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
            }

            // if !((gene.0 == 1686) || (gene.0 == 4660)) {
            //     continue
            // }

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

        if i % 100 == 0 {
            eprintln!("{}", i);
        }


    };

    let array = Array::from_shape_vec((samples,counts.len()/samples),counts).unwrap_or(Array::zeros((0,0)));

    eprintln!("===========");
    eprintln!("{},{}", array.shape()[0], array.shape()[1]);

    array
}

fn read_standard_in() -> Array<f64,Ix2> {

    let stdin = io::stdin();
    let count_array_pipe_guard = stdin.lock();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (_i,line) in count_array_pipe_guard.lines().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        for (_j,gene) in line.as_ref().expect("readline error").split_whitespace().enumerate() {

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    if exp_val.is_nan() {
                        eprintln!("Read a nan: {:?},{:?}", gene,exp_val);
                        panic!("Read a nan!")
                    }
                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        eprintln!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        eprintln!("Cell content: {:?}", gene);
                        panic!("Parsing error");
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

    };

    // eprintln!("Counts read:");
    // eprintln!("{:?}", counts);

    let array = Array::from_shape_vec((samples,counts.len()/samples),counts).unwrap_or(Array::zeros((0,0)));

    assert!(!array.iter().any(|x| x.is_nan()));

    array
}

pub fn standardize(input: &Array<f64,Ix2>) -> Array<f64,Ix2> {
    let mut means = input.mean_axis(Axis(0));
    let mut variances = input.var_axis(Axis(0),0.);

    let mut standardized = input.clone();

    for i in 0..standardized.shape()[0] {
        let mut row = standardized.slice_mut(s![i,..]);
        // eprintln!("{:?}",row);
        let centered_row = &row - &means;
        // eprintln!("{:?}",centered);
        let mut standardized_row = centered_row * &variances;
        for v in standardized_row.iter_mut() {
            if !v.is_finite() {
                *v = 0.;
            }
        }
        // eprintln!("{:?}",standardized);
        row.assign(&standardized_row);
    }

    standardized

}

pub fn sanitize(mut input: Array<f64,Ix2>) -> Array<f64,Ix2> {
    let sums = input.sum_axis(Axis(0));
    let non_zero = sums.iter().map(|x| if *x > 0. {1} else {0}).sum::<usize>();
    if non_zero < input.shape()[1] {
        eprintln!("WARNING: This input isn't sanitized, some features are all 0. We recommend using sanitized data");
        eprintln!("Sums:{:?}",sums.shape());
        eprintln!("Non-zero:{:?}",non_zero);
        let mut sanitized = Array::zeros((input.shape()[0],non_zero));
        eprintln!("Sanitized:{:?}",sanitized.shape());
        let mut feature_iter = input.axis_iter(Axis(1));
        let mut counter = 0;
        for (f,&s) in feature_iter.zip(sums.into_iter()) {
            // eprintln!("Feature:{:?}",f.shape());
            if s > 0. {
                sanitized.column_mut(counter).assign(&f);
                counter += 1;
            }
        };
        sanitized
    }
    else { input }
}


pub fn cosine_similarity_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let sanitized = sanitize(slice.to_owned());
    let mut products = slice.dot(&slice.t());
    // eprintln!("Products");
    let mut geo = (&slice * &slice).sum_axis(Axis(1));
    // eprintln!("geo");
    geo.mapv_inplace(f64::sqrt);
    for i in 0..slice.rows() {
        for j in 0..slice.rows() {
            products[[i,j]] /= (&geo[i] * &geo[j])
        }
    }
    for i in 0..slice.rows() {
        products[[i,i]] = 1.;
    }
    products
}


pub fn euclidean_similarity_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut products = slice.dot(&slice.t());
    eprintln!("Products");
    let mut geo = (&slice * &slice).sum_axis(Axis(1));
    eprintln!("geo");

    for i in 0..slice.rows() {
        for j in 0..slice.rows() {
            products[[i,j]] = 1.0 / (&geo[i] + &geo[j] - 2.0 * products[[i,j]]).sqrt();
            if !products[[i,j]].is_finite() {
                products[[i,j]] = 1.0;
            };
            products[[j,i]] = products[[i,j]];
        }
    }

    for i in 0..slice.rows() {
        products[[i,i]] = 1.0;
    }

    products
}

pub fn correlation_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut output = Array::zeros((slice.rows(),slice.cols()));
    for i in 0..slice.rows() {
        for j in i..slice.cols() {
            let c = correlation(slice.row(i),slice.row(j));
            output[[i,j]] = c;
            output[[j,i]] = c;
        }
    }
    output
}


fn argsort(input: &Vec<f64>) -> Vec<usize> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| x.0).collect();
    out
}

#[derive(Debug,Clone,Copy)]
pub enum Distance {
    Manhattan,
    Euclidean,
    Cosine,
    Correlation,
}

impl Distance {
    pub fn parse(argument: &str) -> Distance {
        match &argument[..] {
            "manhattan" | "m" | "cityblock" => Distance::Manhattan,
            "euclidean" | "e" => Distance::Euclidean,
            "cosine" | "c" | "cos" => Distance::Cosine,
            "correlation" => Distance::Correlation,
            _ => {
                eprintln!("Not a valid distance option, defaulting to cosine");
                Distance::Cosine
            }
        }
    }

    pub fn measure(&self,p1:ArrayView<f64,Ix1>,p2:ArrayView<f64,Ix1>) -> f64 {
        match self {
            Distance::Manhattan => {
                (&p1 - &p2).scalar_sum()
            },
            Distance::Euclidean => {
                (&p1 - &p2).map(|x| x.powi(2)).sum().sqrt()
            },
            Distance::Cosine => {
                let dot_product = p1.dot(&p2);
                let p1ss = p1.map(|x| x.powi(2)).sum().sqrt();
                let p2ss = p2.map(|x| x.powi(2)).sum().sqrt();
                1.0 - (dot_product / (p1ss * p2ss))
            }
            Distance::Correlation => {
                correlation(p1,p2)
            }
        }
    }

    pub fn matrix(&self,p1:ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
        match self {
            Distance::Manhattan => {

                let mut mtx = Array::zeros((p1.rows(),p1.rows()));

                for i in 0..p1.rows() {
                    for j in i..p1.rows() {
                        let d = (&p1.row(i) - &p1.row(j)).sum();
                        mtx[[i,j]] = d;
                        mtx[[j,i]] = d;
                    }
                }
                mtx
            },
            Distance::Euclidean => {
                euclidean_similarity_matrix(p1)
            },
            Distance::Cosine => {
                cosine_similarity_matrix(p1)
            }
            Distance::Correlation => {
                correlation_matrix(p1)
            }
        }
    }
}


#[derive(Debug,Clone)]
pub enum Command {
    FitPredict,
    Density,
}

impl Command {

    pub fn parse(command: &str) -> Command {

        match &command[..] {
            "fitpredict" | "fit_predict" | "combined" => Command::FitPredict,
            "density" => Command::Density,
            _ =>{
                eprintln!("Not a valid top-level command, please choose from \"fit\",\"predict\", or \"fitpredict\". Exiting");
                panic!()
            }
        }
    }
}

pub fn array_mean(input: &ArrayView<f64,Ix1>) -> f64 {
    input.iter().sum::<f64>() / (input.len() as f64)
}

pub fn vec_mean(input: &Vec<f64>) -> f64 {
    input.iter().sum::<f64>() / input.len() as f64
}

pub fn correlation(p1: ArrayView<f64,Ix1>,p2: ArrayView<f64,Ix1>) -> f64 {

    if p1.len() != p2.len() {
        panic!("Tried to compute correlation for unequal length vectors: {}, {}",p1.len(),p2.len());
    }

    let mean1: f64 = array_mean(&p1);
    let mean2: f64 = array_mean(&p2);

    let dev1: Vec<f64> = p1.iter().map(|x| (x - mean1)).collect();
    let dev2: Vec<f64> = p2.iter().map(|x| (x - mean2)).collect();

    let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (p1.len() as f64 - 1.);

    let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (p1.len() as f64 - 1.).max(1.)).sqrt();
    let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (p2.len() as f64 - 1.).max(1.)).sqrt();

    // println!("{},{}", std_dev1,std_dev2);

    let r = covariance / (std_dev1*std_dev2);

    if r.is_nan() {0.} else {r}

}

pub fn write_array<T: Debug>(input: Array<T,Ix2>,target:&Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .outer_iter()
        .map(|x| x.iter()
            .map(|y| format!("{:?}",y))
            .collect::<Vec<String>>()
            .join("\t")
        )
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}

pub fn write_vector<T: Debug>(input: Array<T,Ix1>,target: &Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .iter()
        .map(|x| format!("{:?}",x))
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}

pub fn write_vec<T: Debug>(input: Vec<T>,target: &Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .iter()
        .map(|x| format!("{:?}",x))
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}

pub fn renumber(invec:&Vec<usize>) -> Vec<usize> {
    let mut map: HashMap<usize,usize> = HashMap::new();
    let mut outvec = Vec::with_capacity(invec.len());
    for i in invec {
        let l = map.len();
        outvec.push(*map.entry(*i).or_insert(l));
    }
    outvec
}


//
// fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {
//
//     input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")
//
// }










//
