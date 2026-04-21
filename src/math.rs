use std::ops::{BitAnd, BitXorAssign};

use rand_chacha::{ChaCha8Rng, rand_core::Rng};

#[inline(always)]
#[optimize(speed)]
pub fn sparse_random(rng: &mut ChaCha8Rng) -> u64 {
    rng.next_u64().bitand(rng.next_u64())
}

#[inline(always)]
#[optimize(speed)]
pub fn u16_bitand_bitxor_assign(mask: &mut u16, rhs: u16) -> u16 {
    let masked = mask.bitand(rhs);
    mask.bitxor_assign(rhs);
    masked
}
