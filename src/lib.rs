pub mod bayesian_network;
extern crate libc;    

use std::{collections::{HashMap}, iter::FromIterator};

use bayesian_network::*;

#[no_mangle]
pub extern "C" fn make_concrete_prob(prob: f64) -> *mut Probability {
    return Box::into_raw(Box::new(Probability::Concrete(prob)))
}

/// Make a symbolic probability
#[no_mangle]
pub extern "C" fn make_symbolic_prob(label: usize) -> *mut Probability {
    return Box::into_raw(Box::new(Probability::Symbol(label)))
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
    assigments: *const *const usize, probabilities: *const *const Probability) -> *mut CPT {
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
            assert!(!probabilities.is_null());
            let probabilities : Vec<*const Probability> = 
                std::slice::from_raw_parts(probabilities, num_assignments).to_vec();
            let prob_unwrap : Vec<Probability> = probabilities.iter().map(|x| (**x).clone()).collect();
            HashMap::from_iter(assign_vec.into_iter().zip(prob_unwrap))
        };
        return Box::into_raw(Box::new(CPT::new(var, parent_vec, prob)))
}

#[no_mangle]
pub extern "C" fn make_bayesian_network(num_vars: usize, 
    shape: *const usize, num_cpts: usize, cpts: *const *const CPT) -> *mut BayesianNetwork {
        unsafe {
            let shape = std::slice::from_raw_parts(shape, num_vars).to_vec();
            let cpt_ptrvec : &[*const CPT] = std::slice::from_raw_parts(cpts, num_cpts);
            let cpt : Vec<CPT> = cpt_ptrvec.iter().map(|x| (**x).clone()).collect();
            let bn = BayesianNetwork::new(shape, cpt);
            return Box::into_raw(Box::new(bn));
        }
}

/// Construct a new symbol label that mapes the variable indexed by labels[i] to
/// weight probs[i]
#[no_mangle]
pub extern "C" fn make_symbol_table(num_symbols: usize, 
    labels: *const usize, probs: *const f64) -> *mut SymbolTable {
    unsafe {
        let labels = std::slice::from_raw_parts(labels, num_symbols);
        let probs = std::slice::from_raw_parts(probs, num_symbols);
        let map : HashMap<&usize, &f64> = HashMap::from_iter(labels.into_iter().zip(probs));
        let map_into = map.iter().map(|(&key, &value)| (*key, *value)).collect();
        return Box::into_raw(Box::new(SymbolTable::new(map_into)));
    }
}

#[no_mangle]
pub extern "C" fn make_empty_symbol_table() -> *mut SymbolTable {
    unsafe {
        return Box::into_raw(Box::new(SymbolTable::empty()))
    }
}


#[no_mangle]
pub extern "C" fn compile_bayesian_network(bn: *const BayesianNetwork) -> *mut CompiledBayesianNetwork {
    unsafe {
        return Box::into_raw(Box::new(CompiledBayesianNetwork::new(bn.as_ref().unwrap(), 
            CompileMode::BottomUpChaviraDarwicheBDD)));
    }
}


/// Compute the joint marginal for the subset of variables given in `vars`
/// Gives ownership of the returned array to the caller
#[no_mangle]
pub extern "C" fn joint_marginal(bn: *mut CompiledBayesianNetwork, st: *const SymbolTable,
    num_vars: usize, vars: *const usize) -> *const f64 {
    unsafe {
        let vars = std::slice::from_raw_parts(vars, num_vars);
        let mut bnref = bn.as_mut().unwrap();
        let r = bnref.joint_marginal(st.as_ref().unwrap(), vars);
        let mut v = Vec::new();
        for assgn in AssignmentIter::new(bnref.get_shape().clone(), vars.to_vec()) {
            v.push(r[&assgn]);
        }
        let r = v.as_ptr();
        std::mem::forget(v);
        return r
    }
}
