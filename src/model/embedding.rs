use crate::model::N_EMBD;

#[derive(Clone)]
pub struct Embedding(pub [u16; N_EMBD]);

#[derive(Clone)]
pub struct EmbeddingNorm(pub [u8; N_EMBD]);
