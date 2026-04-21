use std::ops::{BitXor, IndexMut};

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::{
    model::{CFG_BIT_FLIPS, LanguageModel, N_EMBD, N_LAYER, N_PROCESS, ND_PROCESS_EMBD},
    tokenizer::{BLOCK_COUNT, SINGLE_BLOCK_SIZE},
};

const PROJ_WIDTH: u32 = 2 + N_LAYER as u32;

#[optimize(speed)]
pub fn backward_pass(model: &mut LanguageModel, rng: &mut ChaCha8Rng) {
    for _ in 0..CFG_BIT_FLIPS {
        match rng.next_u32().rem_euclid(PROJ_WIDTH) {
            0 => {
                let idx = (rng.next_u32() as usize).rem_euclid(SINGLE_BLOCK_SIZE);
                let embd_idx = (rng.next_u32() as usize).rem_euclid(N_EMBD);

                for block_idx in 0..BLOCK_COUNT {
                    let embedding = model
                        .embeddings
                        .index_mut(idx.strict_add(block_idx.strict_mul(SINGLE_BLOCK_SIZE)));
                    let flip_bit = rng.next_u32().rem_euclid(16);
                    let embd = embedding.0.index_mut(embd_idx);
                    *embd = embd.bitxor(1u16.strict_shl(flip_bit));
                }
            }
            1 => {
                let idx = (rng.next_u32() as usize).rem_euclid(SINGLE_BLOCK_SIZE);
                let embd_idx = (rng.next_u32() as usize).rem_euclid(N_EMBD);

                for block_idx in 0..BLOCK_COUNT {
                    let embedding = model
                        .embedding_norm
                        .index_mut(idx.strict_add(block_idx.strict_mul(SINGLE_BLOCK_SIZE)));
                    let flip_bit = rng.next_u32().rem_euclid(8);
                    let embd = embedding.0.index_mut(embd_idx);
                    *embd = embd.bitxor(1u8.strict_shl(flip_bit));
                }
            }
            index => {
                let layer_idx = index.strict_sub(2) as usize;
                let layer = model.layers.index_mut(layer_idx);
                let block_idx = (rng.next_u32() as usize).rem_euclid(N_PROCESS.strict_add(1));
                let embd_idx = (rng.next_u32() as usize).rem_euclid(N_EMBD);

                if block_idx == N_PROCESS {
                    let flip_bit = rng.next_u32().rem_euclid(8);
                    let embd = layer.process_norms.index_mut(embd_idx);
                    *embd = embd.bitxor(1u8.strict_shl(flip_bit));
                } else {
                    let flip_bit = rng.next_u32().rem_euclid(16);
                    let embd = layer
                        .forward
                        .index_mut(embd_idx.strict_add(block_idx.strict_mul(N_EMBD)));
                    *embd = embd.bitxor(1u16.strict_shl(flip_bit));
                }
            }
        }
    }
}
