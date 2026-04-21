use std::array;

use rand_chacha::ChaCha8Rng;

use crate::{
    math::sparse_random,
    model::{
        embedding::{Embedding, EmbeddingNorm},
        layer::Layer,
    },
    tokenizer::FULL_BLOCK_SIZE,
};

pub mod backward_pass;
mod embedding;
pub mod forward_pass;
mod layer;
pub mod loss;

pub const N_LAYER: usize = 8;
pub const N_EMBD: usize = 256;
pub const N_PROCESS: usize = 4;
pub const N_PROCESS_NORM: usize = 2;
pub const N_VOCAB: usize = FULL_BLOCK_SIZE;

pub const ND_PROCESS_EMBD: usize = N_EMBD.strict_mul(N_PROCESS);
pub const ND_PROCESS_NORM: usize = N_EMBD.strict_mul(N_PROCESS_NORM);

pub const CTX_SLIDING: u16 = 16;
pub const CTX_FULL: u16 = 256;

pub const CFG_SEQ_LEN: usize = 256;
pub const CFG_BIT_FLIPS: usize = 256;

// NOTE: avx512 has 512 bit lanes unlike everything else released in the past century, so they can
// accum higher

#[cfg(target_feature = "avx512f")]
pub const CFG_ACCUM: usize = 512usize.strict_div(16usize);
#[cfg(not(target_feature = "avx512f"))]
pub const CFG_ACCUM: usize = 256usize.strict_div(16usize);

pub struct LanguageModel {
    pub embeddings: [Embedding; N_VOCAB],
    pub layers: [Layer; N_LAYER],
    pub embedding_norm: [EmbeddingNorm; N_VOCAB],
}

impl LanguageModel {
    pub fn initialize(mut rng: ChaCha8Rng) -> Self {
        Self {
            embeddings: array::from_fn(|_| {
                Embedding(array::from_fn(|_| sparse_random(&mut rng) as u16))
            }),
            layers: array::from_fn(|idx| Layer {
                process_norms: array::from_fn(|_| sparse_random(&mut rng) as u8),
                attn_window: if idx.rem_euclid(4) == 3 {
                    CTX_FULL
                } else {
                    CTX_SLIDING
                },
                forward: array::from_fn(|_| sparse_random(&mut rng) as u16),
                forward_norms: array::from_fn(|_| sparse_random(&mut rng) as u8),
            }),
            embedding_norm: array::from_fn(|_| {
                EmbeddingNorm(array::from_fn(|_| sparse_random(&mut rng) as u8))
            }),
        }
    }

    pub fn param_count(&self) -> usize {
        self.embeddings
            .len()
            .saturating_mul(N_EMBD)
            .saturating_add(
                self.layers.len().saturating_mul(
                    1usize
                        .saturating_add(N_EMBD)
                        .saturating_add(ND_PROCESS_NORM)
                        .saturating_add(ND_PROCESS_EMBD),
                ),
            )
            .saturating_add(self.embedding_norm.len().saturating_mul(N_EMBD))
    }

    pub fn clone_weights(&self, target: &mut Self) {
        target.embeddings.clone_from(&self.embeddings);
        target.layers.clone_from(&self.layers);
        target.embedding_norm.clone_from(&self.embedding_norm);
    }

    pub fn clone_allocating(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            layers: self.layers.clone(),
            embedding_norm: self.embedding_norm.clone(),
        }
    }
}
