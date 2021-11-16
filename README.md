# Introduction

* To build: download `rust` and run `cargo build`. This builds a dynamically linkable library in `target/debug`; to build a release build (much faster) run `cargo build --release`
* To test: run `cargo test`

# Rust API

* Can be found inside of `src/bayesian_network.rs`
* Includes interfaces for building Bayesian networks, compiling them, and querying them

# C/C++ API

* Can be found in `src/lib.rs`
* An example of building is found in the `c/` subdirectory, which has a makefile
  for building and linking against an `rsbn` build
* See `c/rsbn.h` for the API. This is automatically generated based on
  `src/lib.rs` using `cbindgen`; to regenerate it (if the rust API changes), 
  call `cbindgen --crate rsbn --output c/rsbn.h`