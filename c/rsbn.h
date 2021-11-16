#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct BayesianNetwork;

/// A conditional probability table
struct CPT;

struct CompiledBayesianNetwork;

/// A probability
/// Can be symbolic (to be substituted in later for a known probability)
/// or concrete
struct Probability;

struct SymbolTable;

extern "C" {

Probability *make_concrete_prob(double prob);

/// Make a symbolic probability
Probability *make_symbolic_prob(uintptr_t label);

/// Create a CPT
/// `var`: the variable for this CPT
/// `num_parents`: number of parents
/// `parents`: an array of parent variable labels
/// `assignments`: an array of Bayesian network assignments corresponding to
///    the order in `parents` with the last element being the assignment to
///    `var`
/// `probabilities`: list of probabilities corresponding to each element in
///     `assignments`
CPT *make_cpt(uintptr_t var,
              uintptr_t num_parents,
              const uintptr_t *parents,
              uintptr_t num_assignments,
              const uintptr_t *const *assigments,
              const Probability *const *probabilities);

BayesianNetwork *make_bayesian_network(uintptr_t num_vars,
                                       const uintptr_t *shape,
                                       uintptr_t num_cpts,
                                       const CPT *const *cpts);

/// Construct a new symbol label that mapes the variable indexed by labels[i] to
/// weight probs[i]
SymbolTable *make_symbol_table(uintptr_t num_symbols, const uintptr_t *labels, const double *probs);

SymbolTable *make_empty_symbol_table();

CompiledBayesianNetwork *compile_bayesian_network(const BayesianNetwork *bn);

/// Compute the joint marginal for the subset of variables given in `vars`
/// Gives ownership of the returned array to the caller
const double *joint_marginal(CompiledBayesianNetwork *bn,
                             const SymbolTable *st,
                             uintptr_t num_vars,
                             const uintptr_t *vars);

} // extern "C"
