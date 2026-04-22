use std::{
    intrinsics::assume,
    ops::{BitAnd, BitOr, BitXor, Index, IndexMut, Shl, Shr},
};

use crate::{
    math::u16_bitand_bitxor_assign,
    mem::endian_restrict,
    model::{CTX_FULL, LanguageModel, N_EMBD, N_PROCESS, N_PROCESS_NORM, N_VOCAB, ND_PROCESS_EMBD},
};

pub fn forward_pass(
    model: &LanguageModel,
    ctx: &[u16; CTX_FULL as usize],
    ctx_start: usize,
    embedding_arena: &mut [[u16; N_EMBD]; CTX_FULL as usize],
    transform_inflate_arena: &mut [u16; ND_PROCESS_EMBD],
) -> u16 {
    unsafe {
        // NOTE: if this happens you are fucking stupid
        // the model cannot be the very first turn in context with nothing else
        // what could it possibly predict off of
        assume(ctx.len() > ctx_start);
    }

    let ctx_len = (CTX_FULL as usize).strict_sub(ctx_start);

    // TODO: this could maybe use an embedding cache once not only in training
    #[allow(clippy::needless_range_loop)] // needless anything else??
    for idx in 0..ctx_len {
        let token_idx = ctx_start.strict_add(idx);
        let embedding = embedding_arena.index_mut(idx);
        embedding.clone_from(&model.embeddings.index(ctx[token_idx] as usize).0);
    }

    // TODO: wire in another arena, this is slow, i just want to get it working first
    let mut next_token = *embedding_arena.index_mut(ctx_len.strict_sub(1));

    // complexity on this is ropefuel before you realize it's basically free bitshift ops and
    // cpus have actual real clock speed lmao
    for layer in &model.layers {
        let attn_window = (layer.attn_window as usize).min(ctx_len);
        let mut causal_mask = [u16::MAX; N_EMBD];

        #[allow(clippy::needless_range_loop)]
        for dim_idx in 0..N_EMBD {
            let dim = next_token[dim_idx];
            let process = layer.process_norms[dim_idx];
            let inflate_idx = dim_idx.strict_mul(4);
            transform_inflate_arena[inflate_idx] =
                dim.strict_shl(dim.leading_zeros().saturating_sub(1));
            transform_inflate_arena[inflate_idx.strict_add(1)] = dim
                .strict_shr(dim.trailing_zeros().saturating_sub(1))
                .strict_shl(process.count_ones());
            transform_inflate_arena[inflate_idx.strict_add(2)] = dim
                .strict_shl(dim.leading_zeros().saturating_sub(1))
                .strict_shr(process.count_zeros());
            transform_inflate_arena[inflate_idx.strict_add(3)] =
                dim.strict_shr(dim.trailing_zeros().saturating_sub(1));
        }

        // TODO: this should be manually vectorized with SIMD
        for attn_idx in (1..attn_window).rev() {
            let token = embedding_arena.index(attn_idx);
            #[allow(clippy::needless_range_loop)]
            for dim_idx in 0..N_EMBD {
                let dim = token[dim_idx];
                let process = layer.process_norms[dim_idx];
                let inflate_idx = dim_idx.strict_mul(4);
                let causal_mask = causal_mask.index_mut(dim_idx);
                transform_inflate_arena[inflate_idx] = u16_bitand_bitxor_assign(
                    causal_mask,
                    dim.strict_shl(dim.leading_zeros().saturating_sub(1)),
                );
                transform_inflate_arena[inflate_idx.strict_add(1)] = u16_bitand_bitxor_assign(
                    causal_mask,
                    dim.strict_shr(dim.trailing_zeros().saturating_sub(1))
                        .strict_shl(process.count_ones()),
                );
                transform_inflate_arena[inflate_idx.strict_add(2)] = u16_bitand_bitxor_assign(
                    causal_mask,
                    dim.strict_shl(dim.leading_zeros().saturating_sub(1))
                        .strict_shr(process.count_zeros()),
                );
                transform_inflate_arena[inflate_idx.strict_add(3)] = u16_bitand_bitxor_assign(
                    causal_mask,
                    dim.strict_shr(dim.trailing_zeros().saturating_sub(1)),
                );
            }
        }

        #[allow(clippy::needless_range_loop)]
        for idx in 0..ND_PROCESS_EMBD {
            let dim = transform_inflate_arena[idx];
            let forward_norm = layer.forward[idx];
            let dim = dim.bitxor(forward_norm);
            transform_inflate_arena[idx] = dim;
        }

        // FIXME: may god and llvm be on my side
        #[allow(clippy::needless_range_loop)]
        for idx in 0..N_EMBD {
            let inflate_idx = idx.strict_mul(N_PROCESS);
            let norm_idx = idx.strict_mul(N_PROCESS_NORM);
            let proj_1 = transform_inflate_arena[inflate_idx];
            let proj_1 = proj_1.strict_shl(proj_1.leading_zeros().saturating_sub(1));
            let proj_2 = transform_inflate_arena[inflate_idx.strict_add(1)];
            let proj_2 = proj_2.strict_shr(proj_2.trailing_zeros().saturating_sub(1));
            let proj_2 = proj_2
                .bitand(layer.forward_norms[norm_idx] as u16)
                .shl(8u16);
            let proj_3 = transform_inflate_arena[inflate_idx.strict_add(2)];
            let proj_3 = proj_3.strict_shl(proj_3.leading_zeros().saturating_sub(1));
            let proj_3 = proj_3
                .shr(8u16)
                .bitand(layer.forward_norms[norm_idx.strict_add(1)] as u16);
            let proj_4 = transform_inflate_arena[inflate_idx.strict_add(3)];
            let proj_4 = proj_4.strict_shr(proj_4.trailing_zeros().saturating_sub(1));
            let proj_outer = proj_1.bitxor(proj_4);
            let proj_inner = proj_2.bitor(proj_3);
            let proj = proj_outer.bitxor(proj_inner);
            next_token[idx] = proj;
        }
    }

    let mut lowest_dist = u32::MAX;
    let mut lowest_dist_node = 1u16; // <|end_of_text|>

    'dist: for vocab_idx in 0..N_VOCAB {
        let embedding = model.embeddings.index(vocab_idx);
        let mut dist_sum = 0u32;

        #[allow(clippy::needless_range_loop)]
        for dim_idx in 0..N_EMBD {
            dist_sum = dist_sum.saturating_add(
                next_token[dim_idx]
                    .bitxor(embedding.0[dim_idx])
                    .count_ones(),
            );

            if dist_sum > lowest_dist {
                continue 'dist;
            }
        }

        if dist_sum == 0 {
            return endian_restrict(vocab_idx);
        }

        if dist_sum < lowest_dist {
            lowest_dist_node = vocab_idx as u16;
            lowest_dist = dist_sum;
        }
    }

    lowest_dist_node
}
