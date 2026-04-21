use std::{path::PathBuf, time::SystemTime};

use colored::Colorize;
use humanize_duration::prelude::DurationExt;

use crate::{
    action::train::training_loop,
    cli::{
        Args,
        Command::{Infer, Train},
    },
    dataset::DatasetType,
    model::{CFG_ACCUM, CFG_SEQ_LEN},
    tokenizer::Tokenizer,
};

mod infer;
mod train;

#[derive(Debug)]
pub enum ModelAction {
    Train { epochs: usize, dataset: PathBuf },
}

impl ModelAction {
    pub fn from_args(args: Args) -> ModelAction {
        match args.command {
            Train { epochs, dataset } => ModelAction::Train { epochs, dataset },
            Infer {} => todo!(),
        }
    }

    pub fn perform(self, tokenizer: Tokenizer) {
        match self {
            ModelAction::Train { epochs, dataset } => {
                println!("{} configuration", "stage |".bright_blue());
                println!("{} {} epochs", "config |".bright_yellow(), epochs);
                println!(
                    "{} {} tokens/sequence",
                    "config |".bright_yellow(),
                    CFG_SEQ_LEN
                );
                println!(
                    "{} {} sequences/step",
                    "config |".bright_yellow(),
                    CFG_ACCUM
                );
                println!("{} dataset", "stage |".bright_blue());
                let dataset_start_time = SystemTime::now();
                let dataset = DatasetType::from_file(dataset);
                let dataset = dataset.tokenize(&tokenizer);
                let dataset_timing = SystemTime::now()
                    .duration_since(dataset_start_time)
                    .expect("system clock broken");
                println!(
                    "{} {} to load dataset",
                    "timings |".bright_magenta(),
                    dataset_timing.human(humanize_duration::Truncate::Micro)
                );
                println!("{} {} sequences", "config |".bright_yellow(), dataset.len());
                println!(
                    "{} {} steps/epoch",
                    "config |".bright_yellow(),
                    dataset.len().div_ceil(CFG_ACCUM)
                );
                training_loop(tokenizer, epochs, dataset);
            }
        }
    }
}
