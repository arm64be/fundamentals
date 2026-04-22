use std::{
    array,
    ops::{BitXor, Index, IndexMut, Mul},
    time::SystemTime,
};

use colored::Colorize;
use humanize_duration::prelude::DurationExt;
use rand::seq::SliceRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};

use crate::{
    mem::slide_window,
    model::{
        CFG_ACCUM, CFG_SEQ_LEN, CTX_FULL, LanguageModel, N_EMBD, ND_PROCESS_EMBD,
        backward_pass::backward_pass, forward_pass::forward_pass, loss::calculate_loss,
    },
    tokenizer::Tokenizer,
};

const INFLATED_SEQ_LEN: usize = CTX_FULL as usize + CFG_SEQ_LEN;

#[optimize(speed)]
#[inline(never)]
pub fn training_loop(_tokenizer: Tokenizer, epochs: usize, mut dataset: Vec<[u16; CFG_SEQ_LEN]>) {
    let mut start_time: SystemTime;

    let mut rng = ChaCha8Rng::from_seed([42; 32]);

    dataset.shuffle(&mut rng);

    // turn dataset into 2d for vectorization
    let dataset = {
        println!("{} dataset repack", "stage |".bright_blue());
        let start_time = SystemTime::now();

        let mut filled_dataset: Vec<[[u16; CFG_SEQ_LEN]; CFG_ACCUM]> =
            Vec::with_capacity(dataset.len().div_ceil(CFG_ACCUM));
        let mut finished = false;

        loop {
            let mut current_vec: [[u16; CFG_SEQ_LEN]; CFG_ACCUM] = [[0; CFG_SEQ_LEN]; CFG_ACCUM];

            if dataset.len() >= CFG_ACCUM {
                let drained = dataset.drain(..CFG_ACCUM);
                current_vec.copy_from_slice(drained.as_slice());
            } else {
                let remaining = CFG_ACCUM - dataset.len();
                current_vec[..dataset.len()].copy_from_slice(&dataset);

                println!(
                    "{} saturated {} to reach {}",
                    "trace |".bright_green(),
                    remaining,
                    CFG_ACCUM
                );

                // NOTE: this assumes that there are more than CFG_ACCUM sequences in the dataset
                // all things considered, if you make this panic, you deserve it.

                let first_entry = filled_dataset.index(0);
                current_vec[CFG_ACCUM - remaining..CFG_ACCUM]
                    .copy_from_slice(&first_entry[0..remaining]);

                finished = true;
            }

            filled_dataset.push(current_vec);

            if finished {
                break;
            }
        }

        let repacking_timing = SystemTime::now()
            .duration_since(start_time)
            .expect("system clock broken");

        println!(
            "{} repacked as {}x{}x{}",
            "trace |".bright_green(),
            filled_dataset.len(),
            CFG_ACCUM,
            CFG_SEQ_LEN
        );

        println!(
            "{} {} to repack dataset",
            "timings |".bright_magenta(),
            repacking_timing.human(humanize_duration::Truncate::Micro)
        );

        println!("{} dataset inflatee", "stage |".bright_blue());
        let start_time = SystemTime::now();

        let mut inflated: Vec<[[u16; INFLATED_SEQ_LEN]; CFG_ACCUM]> =
            Vec::with_capacity(filled_dataset.len());
        filled_dataset
            .iter()
            .map(|batch| {
                array::from_fn(|batch_idx| {
                    array::from_fn(|token_idx| {
                        if token_idx < CTX_FULL as usize {
                            0
                        } else {
                            batch[batch_idx][token_idx.saturating_sub(CTX_FULL as usize)]
                        }
                    })
                })
            })
            .collect_into(&mut inflated);
        drop(filled_dataset);

        let inflate_time = SystemTime::now()
            .duration_since(start_time)
            .expect("system clock broken");

        println!(
            "{} inflated {} to {}",
            "trace |".bright_green(),
            CFG_SEQ_LEN,
            INFLATED_SEQ_LEN
        );

        println!(
            "{} {} to inflate dataset",
            "timings |".bright_magenta(),
            inflate_time.human(humanize_duration::Truncate::Micro)
        );

        inflated
    };

    println!("{} model initialization", "stage |".bright_blue());

    let mut model = LanguageModel::initialize(rng.clone());

    println!(
        "{} initialized {} parameters",
        "trace |".bright_green(),
        model.param_count()
    );

    let steps_per_epoch = dataset.len();

    // NOTE: arena pools for the forward and backward passes to reduce moving around memory
    // TODO: use a proper allocator and align on page size
    let mut forward_embedding_arena = [[[0u16; N_EMBD]; CTX_FULL as usize]; CFG_ACCUM];
    let mut sequence_loss_arena = [[0u8; CFG_SEQ_LEN]; CFG_ACCUM];
    let mut accum_loss_arena = [0u32; CFG_ACCUM];
    let mut transform_inflate_arena = [[0u16; ND_PROCESS_EMBD]; CFG_ACCUM];

    let mut lowest_loss = 522240u32.strict_mul(CFG_ACCUM as u32);
    let mut model_working = model.clone_allocating();

    for epoch in 0..epochs {
        println!("{} epoch {}/{}", "stage |".bright_blue(), epoch + 1, epochs);
        start_time = SystemTime::now();

        for (idx, batch) in dataset.iter().enumerate() {
            train_step(
                batch,
                &mut model,
                &mut forward_embedding_arena,
                &mut sequence_loss_arena,
                &mut accum_loss_arena,
                &mut transform_inflate_arena,
                &mut lowest_loss,
                &mut model_working,
                &mut rng,
            );

            #[cfg(debug_assertions)]
            println!(
                "{} {}/{} {}",
                "step |".bright_red(),
                idx + 1,
                steps_per_epoch,
                format!("{:0.4} loss", (lowest_loss as f64).log10()).red(),
            );
        }

        println!(
            "{} epoch {} finished {} {}",
            "stage |".bright_blue(),
            epoch + 1,
            format!("{:0.4} loss", (lowest_loss as f64).log10()).red(),
            format!(
                "{} elapsed",
                SystemTime::now()
                    .duration_since(start_time)
                    .expect("broken system clock")
                    .human(humanize_duration::Truncate::Micro)
            )
            .bright_black()
        )
    }
}

#[optimize(speed)]
#[inline(never)]
#[allow(clippy::complexity)]
fn train_step(
    sequence_batch: &[[u16; INFLATED_SEQ_LEN]; CFG_ACCUM],
    model: &mut LanguageModel,
    forward_embedding_arena: &mut [[[u16; N_EMBD]; CTX_FULL as usize]; CFG_ACCUM],
    sequence_loss_arena: &mut [[u8; CFG_SEQ_LEN]; CFG_ACCUM],
    accum_loss_arena: &mut [u32; CFG_ACCUM],
    transform_inflate_arena: &mut [[u16; ND_PROCESS_EMBD]; CFG_ACCUM],
    lowest_loss: &mut u32,
    model_working: &mut LanguageModel,
    rng: &mut ChaCha8Rng,
) {
    backward_pass(
        model_working,
        rng,
        (*lowest_loss as f64).mul(4.0).log10().floor() as usize,
    );

    // NOTE: this should be auto-unrolled by llvm
    // if it isn't, make sure you're using enough optimization passes
    for accum_idx in 0..CFG_ACCUM {
        let sequence = sequence_batch.index(accum_idx);
        let fwd_embeddings = forward_embedding_arena.index_mut(accum_idx);
        let loss = accum_loss_arena.index_mut(accum_idx);
        let sequence_loss = sequence_loss_arena.index_mut(accum_idx);
        let transform_inflate = transform_inflate_arena.index_mut(accum_idx);
        handle_sequence(
            sequence,
            model_working,
            fwd_embeddings,
            sequence_loss,
            transform_inflate,
        );
        *loss = calculate_loss(sequence_loss);
    }

    let sum_loss: u32 = accum_loss_arena.iter().sum();

    if *lowest_loss > sum_loss {
        *lowest_loss = sum_loss;
        model_working.clone_weights(model);
    } else {
        model.clone_weights(model_working);
    }
}

#[optimize(speed)]
// #[inline(never)]
fn handle_sequence(
    sequence: &[u16; INFLATED_SEQ_LEN],
    model: &LanguageModel,
    forward_embedding_arena: &mut [[u16; N_EMBD]; CTX_FULL as usize],
    sequence_loss_arena: &mut [u8; CTX_FULL as usize],
    transform_inflate_arena: &mut [u16; ND_PROCESS_EMBD],
) {
    // FIXME: this fucking sucks i hope the compiler can save it
    for token_idx in CTX_FULL.strict_add(17) as usize..INFLATED_SEQ_LEN {
        let context_slice: &[u16; CTX_FULL as usize] =
            slide_window(sequence, token_idx.strict_sub(1));
        let real_token = sequence[token_idx];
        let generated_token = forward_pass(
            model,
            context_slice,
            (CTX_FULL as usize)
                .saturating_sub(token_idx.strict_sub(CTX_FULL as usize).strict_sub(1)),
            forward_embedding_arena,
            transform_inflate_arena,
        );
        let token_delta = real_token.bitxor(generated_token);
        sequence_loss_arena[token_idx.strict_sub(CTX_FULL as usize)] =
            token_delta.count_ones() as u8;
    }
}
