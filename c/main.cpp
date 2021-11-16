#include "rsbn.h"
#include <stdio.h>

int main() {
    Probability* p1 = make_concrete_prob(0.1);
    Probability* p2 = make_concrete_prob(0.9);
    uintptr_t assgn1[1] = {0};
    uintptr_t assgn2[1] = {1};
    uintptr_t* assignments[2] = {assgn1, assgn2};
    const Probability* probs[2] = { p1, p2 };
    uintptr_t parents[0] = {};

    // simple CPT with Pr(A) = 0.1, Pr(!A) = 0.9
    CPT* v0 = make_cpt(0, 0, parents, 2, assignments, probs);

    // a single variable with 2 values
    const uintptr_t shape[1] = {2}; 

    // a single CPT
    const CPT* cpts[1] = {v0};

    BayesianNetwork* bn = make_bayesian_network(1, shape, 1, cpts);

    // compile it
    CompiledBayesianNetwork* cbn = compile_bayesian_network(bn);

    // compute the marginals
    SymbolTable* empty = make_empty_symbol_table();
    uintptr_t query_vars[1] = {0};
    const double* marginals = joint_marginal(cbn, empty, 1, query_vars);
    
    printf("Prob 1: %f, Prob 2: %f", marginals[0], marginals[1]);
    return 0;
}