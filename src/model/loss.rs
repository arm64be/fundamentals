use crate::{mem::endian_restrict, model::CFG_SEQ_LEN};

#[optimize(speed)]
pub fn calculate_loss(sequence_loss: &mut [u8; CFG_SEQ_LEN]) -> u32 {
    let mut loss: u32 = 0;

    for (idx, value) in sequence_loss.iter().enumerate() {
        let scale: u32 = endian_restrict(CFG_SEQ_LEN - idx);
        loss += scale.strict_mul(value.count_ones());
    }

    loss
}
