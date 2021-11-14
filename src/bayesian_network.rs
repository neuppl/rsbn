use std::collections::{HashMap, HashSet};

use rsdd::manager::rsbdd_manager::{BddManager, BddWmc};
use rsdd::repr::bdd::BddPtr;
use rsdd::repr::cnf::Cnf;
use rsdd::repr::var_label::{Literal, VarLabel};

/// A probability
/// Can be symbolic (to be substituted in later for a known probability)
/// or concrete
#[derive(Debug, Clone)]
enum Probability {
    /// a symbolic probability to be substituted in later
    Symbol(usize),
    /// a concrete probability whose value is fixed at compile-time
    Concrete(f64)
}

/// A Bayesian network variable
type BnVar = usize;
/// A Bayesian network value
type BnVal = usize;
type Assignment = Vec<BnVal>;

/// A conditional probability table
#[derive(Debug, Clone)]
struct CPT {
    vars: Vec<BnVar>,
    // assignments occurr relative to the order in `vars`
    probabilities: HashMap<Assignment, Probability>
}

impl CPT {
    fn new(vars: Vec<BnVar>, probabilities: HashMap<Assignment, Probability>) -> CPT {
        CPT { vars, probabilities }
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
    cur: Assignment
}

impl AssignmentIter {
    pub fn new(shape: Vec<usize>, vars: Vec<usize>) -> AssignmentIter {
        let cur = vars.iter().map(|_| 0).collect();
        AssignmentIter {
            shape: shape, vars: vars, cur: cur
        }
    } 
}

impl Iterator for AssignmentIter {
    type Item = Assignment;
    fn next(&mut self) -> Option<Self::Item> {
        // do a binary increment of the current state
        let cur_shape = self.vars.iter().map(|x| self.shape[*x]);
        let (acc, n) = self.cur.iter().zip(cur_shape).fold((1, Vec::new()), |(cur_acc, mut cur_l), (cur_v, cur_shape)| {
            if cur_v + cur_acc >= cur_shape {
                cur_l.push(0);
                (1, cur_l)
            } else {
                cur_l.push(cur_v + cur_acc);
                (0, cur_l)
            }
        });
        if acc == 1 {
            return None
        } else {
            return Some(n)
        }
    }
}

// /// Computes a vector of all possible assignments to variables in shape indexed by vars
// /// `vars` is a vector of variables (indexes into `shape`)
// pub fn all_assignments(shape: &Vec<usize>, vars: &[usize]) -> Vec<Assignment> {
    
// }


struct BayesianNetwork {
    shape: Vec<usize>,
    cpts: Vec<CPT>, // assumed (internally) to be in topological order
}

impl BayesianNetwork {
    /// Generate a new Bayesian network
    /// The `shape` dictates how many variables there are and their domain.
    /// shape[i] says how many values variable i can take.
    fn new(shape: Vec<usize>, cpts: Vec<CPT>) -> BayesianNetwork {
        // sort the CPTs in topological order
        let mut r : Vec<CPT> = Vec::new();
        let mut added : HashSet<BnVar> = HashSet::new(); // added to return set
        while r.len() != cpts.len() {
            for (i, cur) in cpts.iter().enumerate() {
                if added.contains(&i) {
                    continue;
                }
                let unseen_pars = cur.vars.iter().filter(|i| !added.contains(i));
                if unseen_pars.count() == 0 {
                    r.push(cur.clone());
                }
            }
        }
        // now r is a list of topologically sorted CPTs
        return BayesianNetwork { shape: shape, cpts: r }
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

struct SymbolTable {
    /// maps a Bayesian network parameter to its probability (of being true)
    symbol_map: HashMap<usize, f64>
}

enum CompileMode {
    BottomUpChaviraDarwicheBDD
}

struct CompiledBayesianNetwork {
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
    pub fn new(bn: BayesianNetwork, mode: CompileMode) -> CompiledBayesianNetwork {
        // the key is (var, state)
        let mut varcount = 0;
        // (var, value) -> indicator
        let mut indicator_table : HashMap<(usize, usize), VarLabel> = HashMap::new();
        let mut weight_table : HashMap<VarLabel, (f64, f64)> = HashMap::new();
        let mut symbolic_map : HashMap<usize, VarLabel> = HashMap::new();
        let mut clauses : Vec<Vec<Literal>> = Vec::new();
        // (var, value) -> parameter
        let mut parameter_table : HashMap<(usize, usize), VarLabel> = HashMap::new();

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
                    &Probability::Concrete(p) =>  { weight_table.insert(v.clone(), (1.0 - p, p)); () },
                    &Probability::Symbol(vlbl) => { symbolic_map.insert(vlbl, v); () }
                };
            }

            // make parameters clause
            for (assignment, param) in params.iter() {
                let mut cur_indic: Vec<VarLabel> = Vec::new();
                for (indic_var, indic_value) in assignment.iter().enumerate() {
                    cur_indic.push(indicator_table[&(indic_var, *indic_value)]);
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
        let mut mgr = BddManager::new_default_order(5);
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

    /// Computes the joint marginal probability of the subset of variables `vars`
    pub fn joint_marginal(&mut self, st: SymbolTable, vars: Vec<BnVar>) -> HashMap<Assignment, f64> {
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
        for assgn in AssignmentIter::new(self.shape.clone(), vars) {
            let mut cur_bdd = self.bdd;
            let indicators = assgn.iter().enumerate().map(|(var_idx, value)| {
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

#[test]
fn test_marginal_1() {
    // BN : (a) -> (b)
    let shape = vec![vec![2, 3]];
    // let cpts = vec![CPT::new]
    // let bn = BayesianNetwork::new()
}