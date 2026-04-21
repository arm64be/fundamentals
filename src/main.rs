#![feature(
    get_disjoint_mut_helpers,
    optimize_attribute,
    string_from_utf8_lossy_owned,
    portable_simd,
    core_intrinsics,
    iter_collect_into
)]

use clap::Parser;

use crate::{action::ModelAction, tokenizer::Tokenizer};

mod action;
mod cli;
mod containers;
mod dataset;
mod math;
mod mem;
mod model;
mod tokenizer;

fn main() {
    let args = cli::Args::parse();

    let tokenizer = Tokenizer::create();

    if !tokenizer.verify() {
        eprintln!("Tokenizer failed to verify.");
        return;
    }

    let test_string = "Hello, Machine Learning!";
    let tokenized = tokenizer.tokenize(test_string.to_owned());
    let decoded = tokenizer.decode(tokenized, false);

    if test_string != decoded {
        eprintln!("Tokenizer failed end-to-end.");
        return;
    }

    println!("Tokenizer is good.");

    ModelAction::from_args(args).perform(tokenizer);
}
