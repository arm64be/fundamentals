use colored::Colorize;

use crate::{model::CFG_SEQ_LEN, tokenizer::Tokenizer};

#[optimize(speed)]
#[inline(never)]
pub fn inference_loop(tokenizer: Tokenizer, prompt: String, max_tokens: usize) {}
