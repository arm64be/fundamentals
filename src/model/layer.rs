use crate::model::{N_EMBD, ND_PROCESS_EMBD, ND_PROCESS_NORM};

#[derive(Clone)]
pub struct Layer {
    pub process_norms: [u8; N_EMBD],
    pub attn_window: u16,
    pub forward: [u16; ND_PROCESS_EMBD],
    pub forward_norms: [u8; ND_PROCESS_NORM],
}
