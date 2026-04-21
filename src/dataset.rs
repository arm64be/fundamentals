use std::{fs, path::PathBuf};

use crate::{model::CFG_SEQ_LEN, tokenizer::Tokenizer};

pub enum DatasetType {
    RawText { text: String },
}

impl DatasetType {
    // TODO: this should be done smarter
    // FIXME: this abuses panics, fix once done prototyping
    pub fn from_file(path: PathBuf) -> Self {
        match path.extension() {
            Some(string) if string.eq_ignore_ascii_case("txt") => Self::RawText {
                text: fs::read_to_string(path).expect("failed to read dataset"),
            },
            _ => unimplemented!(),
        }
    }

    // TODO: this fucking sucks i hope the auto-vectorizer can save it or it might actually be
    // slower than the hf tokenizers one holy shit
    // FIXME: this has multiple spots where it can just shit itself and die
    pub fn tokenize(self, tokenizer: &Tokenizer) -> Vec<[u16; CFG_SEQ_LEN]> {
        let mut sequences = Vec::new();

        match self {
            DatasetType::RawText { text } => {
                let mut cursor = 0usize;
                let mut finished = false;
                loop {
                    let mut target = cursor + CFG_SEQ_LEN;

                    if target > text.len() {
                        target = text.len();
                        // NOTE: this can panic if the dataset is less than 1 sequence
                        cursor = target.strict_sub(CFG_SEQ_LEN);
                        finished = true;
                    }

                    let sequence = &text[cursor..target];
                    let tokenized = tokenizer.tokenize(sequence.to_owned());
                    assert_eq!(tokenized.len(), CFG_SEQ_LEN);
                    let tokenized: [u16; CFG_SEQ_LEN] =
                        (&tokenized as &[u16]).try_into().expect("bad assert");
                    sequences.push(tokenized);

                    if finished {
                        break;
                    }

                    cursor += CFG_SEQ_LEN;
                }
            }
        }

        sequences
    }
}
