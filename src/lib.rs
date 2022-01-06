pub mod bayesian_network;
extern crate libc;

use std::{collections::{HashMap, HashSet}, iter::FromIterator};

use rsdd::manager::rsbdd_manager::{BddManager, BddWmc};
use rsdd::repr::bdd::BddPtr;
use rsdd::repr::cnf::Cnf;
use rsdd::repr::var_label::{Literal, VarLabel};

/// A probability
/// Can be symbolic (to be substituted in later for a known probability)
/// or concrete
#[derive(Debug, Clone)]
pub enum Probability {
    /// a symbolic probability to be substituted in later
    Symbol(usize),
    /// a concrete probability whose value is fixed at compile-time
    Concrete(f64)
}

/// A Bayesian network variable
type BnVar = usize;
/// A Bayesian network value
type BnVal = usize;
pub type Assignment = Vec<BnVal>;

/// A conditional probability table
#[derive(Debug, Clone)]
pub struct CPT {
    var: BnVar,
    parents: Vec<BnVar>,
    /// assignments occurr relative to the order in `parents`, with
    /// `var` *last*
    probabilities: HashMap<Assignment, Probability>
}

impl CPT {
    pub fn new(var: BnVar, parents: Vec<BnVar>, probabilities: HashMap<Assignment, Probability>) -> CPT {
        // TODO: verify that this is a valid CPT
        CPT { var, parents, probabilities }
    }
}

/// Iterates over assignments in the shape
/// Example: Shape = [2, 3, 4]
/// Vars = [0, 1]
/// Generates a sequence of assignments:
/// [[0,0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
pub struct AssignmentIter {
    shape: Vec<usize>,
    vars: Vec<usize>,
    cur: Option<Assignment>
}

impl AssignmentIter {
    pub fn new(shape: Vec<usize>, vars: Vec<usize>) -> AssignmentIter {
        AssignmentIter {
            shape: shape, vars: vars, cur: None
        }
    }
}

impl Iterator for AssignmentIter {
    type Item = Assignment;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur.is_none() {
            self.cur = Some(self.vars.iter().map(|_| 0).collect());
            return self.cur.clone()
        } else {
            // attempt to do a binary increment of the current state
            let cur_shape : Vec<usize> = self.vars.iter().map(|x| self.shape[*x]).collect();
            for (idx, assgn) in self.cur.as_mut().unwrap().iter_mut().enumerate() {
                if *assgn + 1 < cur_shape[idx] {
                    *assgn = *assgn + 1;
                    return self.cur.clone();
                } else {
                    // add and carry
                    *assgn = 0;
                }
            }
            // we failed to add without carrying
            return None;

        }
    }
}

pub struct BayesianNetwork {
    shape: Vec<usize>,
    cpts: Vec<CPT>, // assumed (internally) to be in topological order
}

impl BayesianNetwork {
    /// Generate a new Bayesian network
    /// The `shape` dictates how many variables there are and their domain.
    /// shape[i] says how many values variable i can take.
    pub fn new(shape: Vec<usize>, cpts: Vec<CPT>) -> BayesianNetwork {
        // sort the CPTs in topological order
        let mut sorted : Vec<CPT> = Vec::new();
        let mut added : HashSet<usize> = HashSet::new();
        fn add(sorted: &mut Vec<CPT>, added: &mut HashSet<BnVar>, cpts: &Vec<CPT>, cpt: &CPT) -> () {
            if added.contains(&cpt.var) { return () };
            // add the parents
            for parent in cpt.parents.iter() {
                let par = cpts.iter().find(|x| x.var == *parent).unwrap();
                add(sorted, added, cpts, par);
            }
            assert!(!added.contains(&cpt.var), "Cycle detected in Bayesian network CPT structure");
            added.insert(cpt.var);
            sorted.push(cpt.clone())
        }
        for cpt in cpts.iter() {
            add(&mut sorted, &mut added, &cpts, cpt);
        }
        return BayesianNetwork { shape: shape, cpts: sorted }
    }

    fn cpts_topological(&self) -> std::slice::Iter<'_, CPT> {
        return self.cpts.iter()
    }

    fn num_vars(&self) -> usize {
        return self.shape.len()
    }

    fn get_shape(&self) -> &Vec<usize> {
        return &self.shape
    }
}

pub struct SymbolTable {
    /// maps a Bayesian network parameter to its probability (of being true)
    symbol_map: HashMap<usize, f64>
}

impl SymbolTable {
    pub fn empty() -> SymbolTable {
        SymbolTable { symbol_map: HashMap::new() }
    }

    pub fn new(symbol_map: HashMap<usize, f64>) -> SymbolTable {
        SymbolTable {
            symbol_map
        }
    }
}

pub enum CompileMode {
    BottomUpChaviraDarwicheBDD
}

pub struct CompiledBayesianNetwork {
    manager: BddManager,
    mode: CompileMode,
    shape: Vec<usize>,
    bdd: BddPtr,
    /// a map from symbolic variables to their corresponding BDD variables
    symbolic_map: HashMap<usize, VarLabel>,
    weights: HashMap<VarLabel, (f64, f64)>,
    /// map from (var, value) -> indicator
    /// where `var` is an index into `shape`, and 0 <= usize <= shape[value]
    indicators: HashMap<(usize, usize), VarLabel>
}


impl CompiledBayesianNetwork {
    pub fn new(bn: &BayesianNetwork, mode: CompileMode) -> CompiledBayesianNetwork {
        // the key is (var, state)
        let mut varcount = 0;
        // (var, value) -> indicator
        let mut indicator_table : HashMap<(usize, usize), VarLabel> = HashMap::new();
        let mut weight_table : HashMap<VarLabel, (f64, f64)> = HashMap::new();
        let mut symbolic_map : HashMap<usize, VarLabel> = HashMap::new();
        let mut clauses : Vec<Vec<Literal>> = Vec::new();

        // make indicator clauses
        for (var_label, sz) in bn.get_shape().iter().enumerate() {
            let mut vars = Vec::new();
            // build the vars vec
            for cur_value in 0..(*sz) {
                let v = VarLabel::new(varcount); varcount += 1;
                indicator_table.insert((var_label, cur_value), v);
                weight_table.insert(v.clone(), (1.0, 1.0));
                vars.push(v);
            }
            // build exactly-one constraint
            clauses.push(vars.iter().map(|var| Literal::new(var.clone(), true)).collect());
            for i in 0..vars.len() {
                for j in i..vars.len() {
                    if i == j {
                        continue;
                    }
                    clauses.push(vec![Literal::new(vars[i], false), Literal::new(vars[j], false)]);
                }
            }
        }

        // make parameter clauses
        for cpt in bn.cpts_topological() {
            // make parameters vec
            let mut params : HashMap<Assignment, VarLabel> = HashMap::new();
            for (assignment, prob) in cpt.probabilities.iter() {
                let v = VarLabel::new(varcount); varcount += 1;
                params.insert(assignment.clone(), v);
                match prob {
                    &Probability::Concrete(p) =>  { weight_table.insert(v.clone(), (1.0, p)); () },
                    &Probability::Symbol(vlbl) => { symbolic_map.insert(vlbl, v); () }
                };
            }

            let mut var_vec = cpt.parents.clone();
            var_vec.push(cpt.var);

            // make parameters clause
            for (assignment, param) in params.iter() {
                let mut cur_indic: Vec<VarLabel> = Vec::new();
                for (indic_var, indic_value) in assignment.iter().enumerate() {
                    cur_indic.push(indicator_table[&(var_vec[indic_var], *indic_value)]);
                }
                // construct clause of cur_indic <=> param
                // first, cur_indic => param
                let mut indicparam : Vec<Literal> = cur_indic.iter().map(|x| Literal::new(*x, false)).collect();
                indicparam.push(Literal::new(*param, true));
                clauses.push(indicparam);
                // now do param => cur_indic
                for indic in cur_indic.iter() {
                    clauses.push(vec![Literal::new(*indic, true), Literal::new(*param, false)]);
                }
            }
        }

        // finally, ready to compile the CNF
        let cnf = Cnf::new(clauses);
        let mut mgr = BddManager::new_default_order(varcount as usize);
        let bdd = mgr.from_cnf(&cnf);
        return CompiledBayesianNetwork {
            manager: mgr,
            bdd: bdd,
            indicators: indicator_table,
            symbolic_map: symbolic_map,
            shape: bn.shape.clone(),
            weights: weight_table,
            mode: CompileMode::BottomUpChaviraDarwicheBDD
        }
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        return &self.shape
    }

    /// Computes the joint marginal probability of the subset of variables `vars`
    pub fn joint_marginal(&mut self, st: &SymbolTable, vars: &[BnVar]) -> HashMap<Assignment, f64> {
        // iterate over assignments, create indicator for that assignment,
        // conjoin it to the bdd and do a weighted model count
        let mut r : HashMap<Assignment, f64> = HashMap::new();
        let mut weights = self.weights.clone();
        // insert all the weights from the symbol table
        for (bnlbl, highw) in st.symbol_map.iter() {
            // translate bnlbl into
            let bddlbl = self.symbolic_map[bnlbl];
            weights.insert(bddlbl, (1.0 - highw, *highw));
        }

        // do the weighted model counts
        for assgn in AssignmentIter::new(self.shape.clone(), vars.to_vec()) {
            let mut cur_bdd = self.bdd;
            assgn.iter().enumerate().for_each(|(var_idx, value)| {
                let indic = self.indicators[&(var_idx, *value)];
                let v = self.manager.var(indic, true);
                cur_bdd = self.manager.and(cur_bdd, v);
            });
            let wmc_param = BddWmc::new_with_default(0.0, 1.0, weights.clone());

            let wmc = self.manager.wmc(cur_bdd, &wmc_param);
            r.insert(assgn, wmc);
        }
        return r;
    }

}

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
