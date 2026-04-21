use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    version,
    about = None,
    long_about = None
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
#[command()]
pub enum Command {
    #[command(
        name = "train",
        about = "Train a model.",
        long_about = None
    )]
    Train {
        #[arg(name = "epochs")]
        epochs: usize,
        #[arg(name = "dataset")]
        dataset: PathBuf,
    },

    #[command(
        name = "infer",
        arg_required_else_help(true),
        about = "Run inference on a weights file using a given test prompt.",
        long_about = None
    )]
    Infer {},
}
