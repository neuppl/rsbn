use std::collections::{HashMap, HashSet};

use rsdd::manager::rsbdd_manager::BddManager;
use rsdd::repr::bdd::BddPtr;
use rsdd::repr::cnf::Cnf;
use rsdd::repr::var_label::{Literal, VarLabel};

/// A probability
/// Can be symbolic (to be substituted in later for a known probability)
/// or concrete
#[derive(Debug, Clone)]
enum Probability {
    Symbol(usize),
    Concrete(f64)
}

type Var = usize;
type Val = usize;
type Assignment = Vec<Val>;

/// A conditional probability table
#[derive(Debug, Clone)]
struct CPT {
    vars: Vec<Var>,
    // assignments occurr relative to the order in `vars`
    probabilities: HashMap<Assignment, Probability>
}


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
        let mut added : HashSet<Var> = HashSet::new(); // added to return set
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
    symbol_map: HashMap<u64, f64>
}

enum CompileMode {
    BottomUpChaviraDarwicheBDD
}

struct CompiledBayesianNetwork {
    manager: BddManager,
    mode: CompileMode,
    bdd: BddPtr
}


impl CompiledBayesianNetwork {
    pub fn new(bn: BayesianNetwork, mode: CompileMode) -> CompiledBayesianNetwork {
        // the key is (var, state)
        let mut varcount = 0;
        // (var, value) -> indicator
        let mut indicator_table : HashMap<(usize, usize), VarLabel> = HashMap::new();
        let mut weight_table : HashMap<VarLabel, (f64, f64)> = HashMap::new();
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
                    &Probability::Concrete(p) => weight_table.insert(v.clone(), (1.0 - p, p)),
                    _ => None
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
            mode: CompileMode::BottomUpChaviraDarwicheBDD
        }
    }
}
