// ASCII tokenizer with some cool tricks
// rust is stupid, char type is 32bit unicode

use core::slice::GetDisjointMutIndex;
use std::ops::AddAssign;

use crate::{containers::Int16Array, mem::endian_restrict};

const SPECIAL_TOKEN_SET: [u8; 2] = [0, 10];
pub const BLOCK_COUNT: usize = 1;
const SINGLE_BLOCK_NO_SPECIAL_SIZE: usize = 128usize.strict_sub(32usize);
pub const SINGLE_BLOCK_SIZE: usize =
    SINGLE_BLOCK_NO_SPECIAL_SIZE.strict_add(SPECIAL_TOKEN_SET.len()); // extended ASCII - control characters + special set
pub const FULL_BLOCK_SIZE: usize = SINGLE_BLOCK_SIZE.strict_mul(BLOCK_COUNT);

pub struct Tokenizer {
    token_map: Int16Array<u8, FULL_BLOCK_SIZE>,
}

impl Tokenizer {
    pub fn create() -> Self {
        let mut map = [30u8; FULL_BLOCK_SIZE];

        for idx in 0..BLOCK_COUNT {
            let offset = idx.strict_mul(SINGLE_BLOCK_SIZE);
            let mut idx = 0;

            for special in SPECIAL_TOKEN_SET {
                map[offset.strict_add(idx)] = special;
                idx.add_assign(1);
            }

            for sub_idx in 0..SINGLE_BLOCK_NO_SPECIAL_SIZE {
                let char_idx = sub_idx.strict_add(32usize);
                assert!(char_idx.is_in_bounds(128));
                map[offset.strict_add(idx).strict_add(sub_idx)] = endian_restrict(char_idx);
            }
        }

        Self {
            token_map: Int16Array(map),
        }
    }

    pub fn verify(&self) -> bool {
        for c in &self.token_map.0 {
            if *c == 30u8 {
                return false;
            }
        }

        true
    }

    pub fn tokenize(&self, input: String) -> Vec<u16> {
        let mut tokenized = Vec::with_capacity(input.len());

        for byte in input.bytes() {
            if byte.is_ascii() && byte > 31 {
                let first_block_index = SPECIAL_TOKEN_SET
                    .len()
                    .strict_add(byte as usize)
                    .strict_sub(32usize);

                if self.token_map.0[first_block_index] != byte {
                    eprintln!(
                        "Character byte {} doesn't map properly? Found {}.",
                        byte, self.token_map.0[first_block_index]
                    );
                } else {
                    assert!(first_block_index.is_in_bounds(SINGLE_BLOCK_SIZE));
                    let token = first_block_index as u16;
                    tokenized.push(token);
                }
            } else {
                let mut saved = false;

                for (idx, special) in SPECIAL_TOKEN_SET.iter().enumerate() {
                    if byte == *special {
                        tokenized.push(endian_restrict(idx));
                        saved = true;
                        break;
                    }
                }

                if !saved {
                    eprintln!("Skipping unknown character byte: {}", byte);
                }
            }
        }

        tokenized
    }

    pub fn decode(&self, tokens: Vec<u16>, stop_on_end: bool) -> String {
        let mut characters = Vec::with_capacity(tokens.len());

        let vocab_count = self.token_map.0.len();
        for token in tokens {
            let token = token as usize;

            if token.is_in_bounds(vocab_count) {
                let character = self.token_map.0[token];

                if character == 0 {
                    if stop_on_end {
                        break;
                    } else {
                        characters.extend_from_slice(b"<|end_of_text|>");
                    }
                } else if character == 10 {
                    characters.extend_from_slice(b"<|new_line|>");
                } else {
                    characters.push(character);
                }
            } else {
                eprintln!("Skipping unknown token: {}", token);
            }
        }

        String::from_utf8_lossy_owned(characters)
    }
}
