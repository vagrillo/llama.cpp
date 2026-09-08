#pragma once

// MoE expert expansion (docs/moe-expansion.md, port of the ds4 "qwen35moe"
// layer-scoped expert-budget expansion).
//
// This is a runtime-only routing change for sparse MoE models: the routed
// expert budget N is raised above the model's native top-K (or cut with a
// dynamic threshold), the ranks above the native K get a linearly decaying
// influence factor (0.99 down to D) applied to the raw probability before
// renormalization, and the kept weights are renormalized to sum 1. Router
// weights, softmax, shared experts and output scales are untouched; with the
// feature off the engine must behave bit-identically to stock.
//
// Implementation strategy is the fixed-N mask ("Strategy B"): the graph always
// carries N selection slots per token and dropped ranks simply get zero
// weight, so the downstream expert kernels stay stock on every backend.

#include "ggml.h"

#include <algorithm>
#include <cmath>

// MoE expert expansion post-pass.
//
//   weights   [1, n_used, n_tokens] non-increasing per token (router top-N,
//             rank order, gathered from the router probabilities)
//   n_used    N: expanded routed-expert budget (N >= 2)
//   k_native  K: the model's native top-k (decay ramp reference)
//   threshold T: keep ranks while weight >= T * weight(rank N/2); 0 = keep N
//   decay_end D: influence of the last extra rank; linear 0.99..D, off if no_decay
//   renormalize: true -> kept weights renormalized to sum 1 (softmax-style routers,
//                e.g. Qwen; semantically a no-op vs the stock renormalization)
//                false -> kept weights keep their raw decayed score scale and the
//                dropped mass is discarded: the correct semantics for routers whose
//                stock scores do not sum to 1 (sqrt-softplus + bias with
//                expert_weights_norm=false, e.g. DeepSeek-V4 conversions)
//   sel_count out: [1] sum of per-token kept-expert counts (optional)
//
// returns the post-processed weights, [1, n_used, n_tokens]; sums to 1 per token
// when renormalize is true.
inline ggml_tensor * build_moe_expansion_weights(
        ggml_context * ctx0,
        ggml_tensor  * weights,
        const int64_t  n_used,
        const int64_t  k_native,
        const float    threshold,
        const float    decay_end,
        const bool     no_decay,
        const bool     renormalize,
        ggml_tensor  ** sel_count) {
    const int64_t n_tokens = weights->ne[2];

    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(n_used >= 2);

    // rank indices j = 0..n_used-1 as [1, N, 1] (0-based, non-increasing weights)
    ggml_tensor * ranks = ggml_reshape_3d(ctx0,
            ggml_arange(ctx0, 0.0f, (float) n_used, 1.0f), 1, n_used, 1);
    ggml_tensor * ones = ggml_fill(ctx0, ranks, 1.0f); // [1, N, 1] constant 1

    // selection: keep = step(F - j) | step(w - T * p_ref), with
    //   F     = max(1, N/4)                 (floor: core ranks always kept)
    //   p_ref = weight of rank R = N/2      (1-indexed reference probability)
    // the kept set is a contiguous prefix: c = min(N, max(F, A)), where A is
    // the number of ranks above the threshold (weights are non-increasing)
    ggml_tensor * keep = nullptr;
    if (threshold > 0.0f) {
        const int64_t n_floor = std::max<int64_t>(1, n_used / 4);

        ggml_tensor * keep_floor = ggml_step(ctx0, ggml_sub(ctx0,
                    ggml_fill(ctx0, ranks, (float) n_floor),
                    ranks));                                        // [1, N, 1] step(F - j)

        // p_ref per token: weight of rank R - 1 (0-based), [1, 1, n_tokens]
        ggml_tensor * p_ref = ggml_view_3d(ctx0, weights,
                1, 1, n_tokens,
                weights->nb[1], weights->nb[2],
                (n_used / 2 - 1) * weights->nb[1]);
        p_ref = ggml_repeat(ctx0, p_ref, weights);                  // [1, N, n_tokens]

        ggml_tensor * above = ggml_step(ctx0, ggml_sub(ctx0, weights,
                    ggml_scale(ctx0, p_ref, threshold)));           // [1, N, n_tokens]

        keep = ggml_clamp(ctx0,
                ggml_add(ctx0, ggml_repeat(ctx0, keep_floor, weights), above),
                0.0f, 1.0f);
    } else {
        keep = ggml_repeat(ctx0, ones, weights);                    // [1, N, n_tokens]
    }

    if (sel_count) {
        // sum of kept expert counts over all tokens: [N, T] -> [1, T] -> [1]
        ggml_tensor * cnt = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, keep, n_used, n_tokens));
        *sel_count = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, cnt, n_tokens, 1));
    }

    ggml_tensor * w = ggml_mul(ctx0, weights, keep);

    // expansion decay: when N exceeds the model's native top-K, every kept rank
    // above K gets a linearly decaying influence factor, 0.99 for the first
    // extra rank down to D for rank N (midpoint with a single extra). computed
    // against the maximum N, so a rank's factor does not jump when the cut moves
    if (n_used > k_native && !no_decay) {
        ggml_tensor * sel_k = ggml_step(ctx0, ggml_sub(ctx0,
                    ggml_add(ctx0, ranks, ggml_fill(ctx0, ranks, 0.5f)),
                    ggml_fill(ctx0, ranks, (float) k_native)));     // [1, N, 1] step(j - K + 0.5)

        ggml_tensor * t = ggml_clamp(ctx0, ggml_scale(ctx0,
                    ggml_sub(ctx0, ranks, ggml_fill(ctx0, ranks, (float) k_native)),
                    n_used - k_native > 1 ? 1.0f / (float) (n_used - k_native - 1) : 0.0f),
                0.0f, 1.0f);                                        // ramp position 0..1
        if (n_used - k_native == 1) {
            t = ggml_add(ctx0, t, ggml_scale(ctx0, sel_k, 0.5f));   // single extra -> midpoint
        }

        ggml_tensor * fac = ggml_add(ctx0,
                ggml_mul(ctx0, sel_k, ggml_add(ctx0,
                        ggml_scale(ctx0, t, decay_end - 0.99f),
                        ggml_fill(ctx0, ranks, 0.99f))),
                ggml_sub(ctx0, ones, sel_k));                       // sel_k*lerp + (1 - sel_k)

        w = ggml_mul(ctx0, w, ggml_repeat(ctx0, fac, weights));
    }

    // renormalize the kept weights to sum 1 per token (renormalize=true, softmax-style
    // routers). with renormalize=false the kept weights keep their raw decayed score
    // scale - renormalizing would change the routed-branch output magnitude vs the
    // stock behavior of the same model with the feature off. (sum > 0 guaranteed when
    // renormalizing: at least F >= 1 ranks are kept and the scores are positive)
    if (renormalize) {
        ggml_tensor * wsum = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, w, n_used, n_tokens)); // [1, n_tokens]
        wsum = ggml_clamp(ctx0, wsum, 6.103515625e-5f, INFINITY);
        w = ggml_div(ctx0, w, ggml_repeat(ctx0, ggml_reshape_3d(ctx0, wsum, 1, 1, n_tokens), w));
    }

    return w;
}
