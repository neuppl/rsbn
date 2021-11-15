pub mod bayesian_network;
extern crate libc;    

use std::{collections::{HashMap, HashSet}, iter::FromIterator};

mod dag;

use bayesian_network::*;

#[no_mangle]
pub extern "C" fn make_concrete_prob(prob: f64) -> *mut libc::c_void {
    return Box::into_raw(Box::new(Probability::Concrete(prob))) as *mut libc::c_void
}

/// Make a symbolic probability
#[no_mangle]
pub extern "C" fn make_symbolic_prob(label: usize) -> *mut libc::c_void {
    return Box::into_raw(Box::new(Probability::Symbol(label))) as *mut libc::c_void
}

/// Create a CPT
/// `var`: the variable for this CPT
/// `num_parents`: number of parents
/// `parents`: an array of parent variable labels
/// `assignments`: an array of Bayesian network assignments corresponding to
///    the order in `parents` with the last element being the assignment to
///    `var`
/// `probabilities`: list of probabilities corresponding to each element in 
///     `assignments`
#[no_mangle]
pub extern "C" fn make_cpt(var: usize, num_parents: usize,
    parents: *const usize, num_assignments: usize, 
    assigments: *const *const usize, probabilities: *const Probability) -> *mut CPT {
        let parent_vec = unsafe {
            assert!(!parents.is_null());
            std::slice::from_raw_parts(parents, num_parents).to_vec()
        };
        let prob : HashMap<Assignment, Probability> = unsafe {
            assert!(!assigments.is_null());
            let assign : Vec<*const usize> = 
                std::slice::from_raw_parts(assigments, num_assignments).to_vec();
            let assign_vec : Vec<Vec<usize>> = assign.iter().map(|x| 
                std::slice::from_raw_parts(*x, num_parents + 1).to_vec()
            ).collect();
            assert!(probabilities.is_null());
            let probabilities : Vec<Probability> = 
                std::slice::from_raw_parts(probabilities, num_assignments).to_vec();
            HashMap::from_iter(assign_vec.into_iter().zip(probabilities))
        };
        return Box::into_raw(Box::new(CPT::new(var, parent_vec, prob)))
}

// #[no_mangle]
// pub extern "C" fn make_bayesian_network(num_vars: usize,
//     shape: shape,

//     ) -> *mut libc::c_void {
//     return Box::into_raw(Box::new(Probability::Symbol(label))) as *mut libc::c_void
// }


fn main() {
    println!("Hello, world!");
}
