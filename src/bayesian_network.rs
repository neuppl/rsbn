use std::collections::HashMap;

use assert_approx_eq::assert_approx_eq;

use crate::*;

#[test]
fn test_marginal_0() {
    // BN : (a)
    let shape = vec![2];
    let cpts = vec![
        CPT::new(0, vec![], HashMap::from([
            (vec![0], Probability::Concrete(0.1)),
            (vec![1], Probability::Concrete(0.9))
            ])),
    ];
    let bn = BayesianNetwork::new(shape, cpts);
    let mut compiled = CompiledBayesianNetwork::new(&bn, CompileMode::BottomUpChaviraDarwicheBDD);
    let r = compiled.joint_marginal(&SymbolTable::empty(), &vec![0]);
    assert_eq!(r[&vec![1]], 0.9);
    assert_eq!(r[&vec![0]], 0.1);
}

#[test]
fn test_marginal_1() {
    // BN : (a) -> (b)
    let shape = vec![2, 2];
    let cpts = vec![
        CPT::new(0, vec![], HashMap::from([
            (vec![0], Probability::Concrete(0.1)),
            (vec![1], Probability::Concrete(0.9))
            ])),
         CPT::new(1, vec![0], HashMap::from([
            (vec![0, 0], Probability::Concrete(0.3)),
            (vec![0, 1], Probability::Concrete(0.7)),
            (vec![1, 0], Probability::Concrete(0.4)),
            (vec![1, 1], Probability::Concrete(0.6)),
            ])),
    ];
    let bn = BayesianNetwork::new(shape, cpts);
    let mut compiled = CompiledBayesianNetwork::new(&bn, CompileMode::BottomUpChaviraDarwicheBDD);
    let r = compiled.joint_marginal(&SymbolTable::empty(), &vec![0, 1]);

    assert_eq!(r[&vec![0, 0]], 0.3 * 0.1);
    assert_eq!(r[&vec![0, 1]], 0.7 * 0.1);
    assert_eq!(r[&vec![1, 0]], 0.4 * 0.9);
    assert_eq!(r[&vec![1, 1]], 0.6 * 0.9);
}

#[test]
fn test_marginal_2() {
    // BN : (a) -> (b)
    let shape = vec![2, 3];
    let cpts = vec![
        CPT::new(0, vec![], HashMap::from([
            (vec![0], Probability::Concrete(0.1)),
            (vec![1], Probability::Concrete(0.9))
            ])),
         CPT::new(1, vec![0], HashMap::from([
            (vec![0, 0], Probability::Concrete(0.1)),
            (vec![0, 1], Probability::Concrete(0.4)),
            (vec![0, 2], Probability::Concrete(0.5)),
            (vec![1, 0], Probability::Concrete(0.3)),
            (vec![1, 1], Probability::Concrete(0.5)),
            (vec![1, 2], Probability::Concrete(0.2)),
            ])),
    ];
    let bn = BayesianNetwork::new(shape, cpts);
    let mut compiled = CompiledBayesianNetwork::new(&bn, CompileMode::BottomUpChaviraDarwicheBDD);
    let r = compiled.joint_marginal(&SymbolTable::empty(), &vec![0, 1]);

    assert_eq!(r[&vec![0, 2]], 0.1 * 0.5);
    assert_eq!(r[&vec![1, 2]], 0.9 * 0.2);
    assert_eq!(r[&vec![1, 1]], 0.9 * 0.5);
}

#[test]
fn test_marginal_3() {
    // BN : (a) -> (b) <- (c)
    let shape = vec![2, 2, 2];
    let cpts = vec![
        CPT::new(0, vec![], HashMap::from([
            (vec![0], Probability::Concrete(0.1)),
            (vec![1], Probability::Concrete(0.9))
            ])),
        CPT::new(2, vec![], HashMap::from([
            (vec![0], Probability::Concrete(0.2)),
            (vec![1], Probability::Concrete(0.8))
            ])),
         CPT::new(1, vec![0, 2], HashMap::from([
            (vec![0, 0, 0], Probability::Concrete(0.1)),
            (vec![0, 0, 1], Probability::Concrete(0.4)),
            (vec![1, 0, 0], Probability::Concrete(0.3)),
            (vec![1, 0, 1], Probability::Concrete(0.5)),
            (vec![1, 1, 0], Probability::Concrete(0.15)),
            (vec![1, 1, 1], Probability::Concrete(0.85)),
            ])),
    ];
    let bn = BayesianNetwork::new(shape, cpts);
    let mut compiled = CompiledBayesianNetwork::new(&bn, CompileMode::BottomUpChaviraDarwicheBDD);
    let r = compiled.joint_marginal(&SymbolTable::empty(), &vec![0, 1, 2]);

    assert_approx_eq!(r[&vec![0, 0, 0]], 0.1 * 0.2 * 0.1, 1e-3);
    assert_approx_eq!(r[&vec![1, 0, 1]], 0.9 * 0.8 * 0.15, 1e-3);
}
