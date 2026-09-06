// Unit tests for the MoE expert-expansion routing post-pass
// (src/llama-moe-expansion.h), checking it against a direct implementation of
// the normative selection spec (docs/moe-expansion.md, section 2).
//
// note: expert-id tie-breaking (lowest id first) is a property of the router
// argsort, not of this post-pass, which receives rank-ordered weights.

#include "../src/llama-moe-expansion.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct case_result {
    std::vector<float> w;  // [N * n_tokens] post-processed weights
    std::vector<float> counts; // per-token kept-expert count
};

static case_result run_expansion(const std::vector<float> & probs, int64_t n_used, int64_t k_native,
                                 float threshold, float decay_end, bool no_decay, int64_t n_tokens) {
    case_result res;

    ggml_init_params ip = {
        /*.mem_size   =*/ 256ull*1024ull*1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, n_tokens);
    memcpy(weights->data, probs.data(), ggml_nbytes(weights));

    ggml_tensor * sel_count = nullptr;
    ggml_tensor * out = build_moe_expansion_weights(ctx, weights, n_used, k_native,
            threshold, decay_end, no_decay, &sel_count);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (sel_count) {
        ggml_build_forward_expand(gf, sel_count);
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(cpu_dev, nullptr);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_graph_compute(backend, gf);

    // ctx-pool tensors are plain host memory for the CPU backend: read directly
    res.w.resize(ggml_nelements(out));
    memcpy(res.w.data(), out->data, ggml_nbytes(out));

    if (n_tokens == 1) {
        // sel_count sums the kept counts over all tokens; with one token the
        // sum is the per-token kept-expert count
        res.counts.resize(1);
        res.counts[0] = *((const float *) sel_count->data);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);

    return res;
}

// one token per call, so the summed sel_count is the per-token count
static case_result run_expansion_1t(const std::vector<float> & probs, int64_t n_used, int64_t k_native,
                                    float threshold, float decay_end, bool no_decay) {
    case_result r = run_expansion(probs, n_used, k_native, threshold, decay_end, no_decay, 1);
    return r;
}

// reference: normative spec, section 2 (rank-ordered top-N probabilities in, kept count + weights out)
static std::pair<int, std::vector<float>> ref_expansion(const float * p, int N, int K, float T, float D, bool decay_on) {
    const int R = N / 2;
    const int F = std::max(1, N / 4);

    int c;
    if (T == 0.0f) {
        c = N;
    } else {
        const float p_ref = p[R - 1];
        int A = 0;
        for (int j = 0; j < N; j++) {
            if (p[j] >= T * p_ref) {
                A++;
            }
        }
        c = std::min(N, std::max(F, A));
    }

    std::vector<float> w(p, p + N);
    for (int j = c; j < N; j++) {
        w[j] = 0.0f;
    }

    if (N > K && decay_on) {
        const int extra = N - K;
        for (int j = K; j < c; j++) {
            const float t = extra > 1 ? (float)(j - K) / (float)(extra - 1) : 0.5f;
            w[j] *= 0.99f + (D - 0.99f) * t;
        }
    }

    float s = 0.0f;
    for (int j = 0; j < c; j++) {
        s += w[j];
    }
    for (int j = 0; j < N; j++) {
        w[j] = (j < c) ? w[j] / s : 0.0f;
    }

    return { c, w };
}

// a realistic power-law-ish router distribution (256 experts collapse fast)
static std::vector<float> power_law(int N, float head) {
    std::vector<float> p(N);
    float s = 0.0f;
    for (int j = 0; j < N; j++) {
        p[j] = head / (float)(j + 1);
        s += p[j];
    }
    for (int j = 0; j < N; j++) {
        p[j] /= s;
    }
    return p;
}

static int n_fail = 0;

static void expect(bool ok, const std::string & what) {
    printf("%s : %s\n", ok ? "  PASS" : "  FAIL", what.c_str());
    if (!ok) {
        n_fail++;
    }
}

static bool vec_close(const std::vector<float> & a, const std::vector<float> & b, float tol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("    mismatch at %zu: got %g, want %g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool sum1(const std::vector<float> & w, int N, int c, float tol) {
    float s = 0.0f;
    for (int j = 0; j < c; j++) {
        s += w[j];
    }
    if (fabsf(s - 1.0f) > tol) {
        return false;
    }
    for (int j = c; j < N; j++) {
        if (w[j] != 0.0f) {
            return false;
        }
    }
    return true;
}

int main() {
    const float D05 = 0.50f;

    // 1. T = 0, N == K: plain renormalization, all N kept, no decay
    {
        auto p = power_law(8, 0.30f);
        auto r = run_expansion_1t(p, 8, 8, 0.0f, D05, false);
        auto ref = ref_expansion(p.data(), 8, 8, 0.0f, D05, true);
        expect(r.counts[0] == 8.0f, "T=0, N=K=8: c == N");
        expect(vec_close(r.w, ref.second, 1e-6f), "T=0, N=K=8: weights == reference");
        expect(sum1(r.w, 8, 8, 1e-6f), "T=0, N=K=8: weights sum to 1");
    }

    // 2. T = 0.8, N = 20, K = 8: c in [R, N], matches the reference
    {
        auto p = power_law(20, 0.25f);
        auto r = run_expansion_1t(p, 20, 8, 0.8f, D05, false);
        auto ref = ref_expansion(p.data(), 20, 8, 0.8f, D05, true);
        expect(r.counts[0] == (float) ref.first, "T=0.8: kept count matches reference");
        expect(ref.first >= 10 && ref.first <= 20, "T=0.8: c in [R, N]");
        expect(vec_close(r.w, ref.second, 1e-5f), "T=0.8: weights == reference (decay + renorm)");
        expect(sum1(r.w, 20, ref.first, 1e-5f), "T=0.8: kept weights sum to 1");
    }

    // 2b. decay factors are rank-based, independent of the cut point:
    //     two different distributions, same rank order -> same relative factors
    {
        auto p1 = power_law(20, 0.25f);
        auto p2 = power_law(20, 0.05f);
        auto r1 = run_expansion_1t(p1, 20, 8, 0.8f, D05, false);
        auto r2 = run_expansion_1t(p2, 20, 8, 0.8f, D05, false);
        auto ref1 = ref_expansion(p1.data(), 20, 8, 0.8f, D05, true);
        auto ref2 = ref_expansion(p2.data(), 20, 8, 0.8f, D05, true);
        const int c_min = std::min(ref1.first, ref2.first); // factors apply to kept ranks only
        bool ok = c_min > 8;
        for (int j = 8; j < c_min && ok; j++) { // 0-based ranks above native K
            const float f1 = (r1.w[j] / p1[j]) / (r1.w[0] / p1[0]);
            const float f2 = (r2.w[j] / p2[j]) / (r2.w[0] / p2[0]);
            const float t = (float)(j - 8) / (float)(20 - 8 - 1);
            const float want = 0.99f + (D05 - 0.99f) * t;
            ok = fabsf(f1 - want) < 1e-4f && fabsf(f2 - want) < 1e-4f;
        }
        expect(ok, "decay factors depend on rank, not on the cut point");
    }

    // 3. T = 10: deep prune, c == floor
    {
        auto p = power_law(20, 0.25f);
        auto r = run_expansion_1t(p, 20, 8, 10.0f, D05, false);
        expect(r.counts[0] == 5.0f, "T=10: c == F (floor = N/4)");
        expect(sum1(r.w, 20, 5, 1e-6f), "T=10: kept weights sum to 1");
    }

    // 4. flat probabilities, T = 0.9: everything stays above the cut -> c == N
    {
        std::vector<float> p(20, 0.05f);
        auto r = run_expansion_1t(p, 20, 8, 0.9f, D05, false);
        expect(r.counts[0] == 20.0f, "T=0.9 flat probs: c == N");
        expect(sum1(r.w, 20, 20, 1e-5f), "T=0.9 flat probs: weights sum to 1");
    }

    // 5. single extra expert (N = K + 1): factor exactly 0.745 (t = 0.5)
    {
        auto p = power_law(9, 0.30f);
        auto r = run_expansion_1t(p, 9, 8, 0.0f, D05, false);
        const float f8 = (r.w[8] / p[8]) / (r.w[0] / p[0]);
        expect(fabsf(f8 - 0.745f) < 1e-5f, "N=K+1: single extra gets midpoint factor 0.745");
        expect(sum1(r.w, 9, 9, 1e-5f), "N=K+1: weights sum to 1");
    }

    // 6. D = 0.30: first extra factor 0.99, last extra factor 0.30
    {
        auto p = power_law(20, 0.25f);
        auto r = run_expansion_1t(p, 20, 8, 0.0f, 0.30f, false);
        const float f8  = (r.w[8]  / p[8])  / (r.w[0] / p[0]);
        const float f19 = (r.w[19] / p[19]) / (r.w[0] / p[0]);
        expect(fabsf(f8 - 0.99f) < 1e-5f, "D=0.30: first extra factor 0.99");
        expect(fabsf(f19 - 0.30f) < 1e-5f, "D=0.30: last extra factor 0.30");
    }

    // 6b. --moe-no-expert-decay: extra ranks at full influence
    {
        auto p = power_law(20, 0.25f);
        auto r = run_expansion_1t(p, 20, 8, 0.0f, D05, true);
        auto ref = ref_expansion(p.data(), 20, 8, 0.0f, D05, false);
        expect(vec_close(r.w, ref.second, 1e-6f), "no_decay: weights == reference");
    }

    // 7. per-token independence: three different rows cut independently
    {
        auto p1 = power_law(20, 0.25f);
        auto p2 = power_law(20, 0.05f);
        std::vector<float> p3(20);
        for (int j = 0; j < 20; j++) {
            p3[j] = (j < 12) ? 0.2f / 12.0f : 1e-4f; // sharp cut after 12
        }
        std::vector<float> batch;
        batch.insert(batch.end(), p1.begin(), p1.end());
        batch.insert(batch.end(), p2.begin(), p2.end());
        batch.insert(batch.end(), p3.begin(), p3.end());

        auto r3 = run_expansion(batch, 20, 8, 0.8f, D05, false, 3);
        auto s1 = run_expansion_1t(p1, 20, 8, 0.8f, D05, false);
        auto s2 = run_expansion_1t(p2, 20, 8, 0.8f, D05, false);
        auto s3 = run_expansion_1t(p3, 20, 8, 0.8f, D05, false);

        bool ok = vec_close({r3.w.begin(),          r3.w.begin() + 20},          s1.w, 1e-6f) &&
                  vec_close({r3.w.begin() + 20,     r3.w.begin() + 40},          s2.w, 1e-6f) &&
                  vec_close({r3.w.begin() + 40,     r3.w.begin() + 60},          s3.w, 1e-6f);
        expect(ok, "per-token cuts are independent (batched = per-row)");
    }

    // 8. pruning below native K (T > 1 with N == K): c in [F, N]
    {
        auto p = power_law(8, 0.30f);
        auto r = run_expansion_1t(p, 8, 8, 2.0f, D05, false);
        auto ref = ref_expansion(p.data(), 8, 8, 2.0f, D05, true);
        expect(r.counts[0] == (float) ref.first, "T=2, N=K=8: count matches reference");
        expect(ref.first >= 2 && ref.first <= 8, "T=2, N=K=8: c in [F, N]");
        expect(sum1(r.w, 8, ref.first, 1e-5f), "T=2, N=K=8: kept weights sum to 1");
    }

    // 9. random-ish sweep: post-pass == reference across many configurations
    {
        bool ok = true;
        float seed = 0.1234f;
        auto rnd = [&seed]() { seed = fmodf(seed*37.13f + 0.717f, 1.0f); return seed; };
        for (int trial = 0; trial < 32 && ok; trial++) {
            const int N = 4 + (trial % 13) * 2;       // even sizes 4..28
            const int K = 2 + (trial % 3) * 2;        // 2, 4 or 6
            std::vector<float> p(N);
            float s = 0.0f;
            for (int j = 0; j < N; j++) {
                p[j] = 0.5f + rnd();
                s += p[j];
            }
            for (int j = 0; j < N; j++) {
                p[j] /= s;
            }
            // the post-pass receives rank-ordered weights (argsort + get_rows
            // upstream in the engine), so feed it a non-increasing vector
            std::sort(p.begin(), p.end(), std::greater<float>());
            const float T = (trial % 4 == 0) ? 0.0f : 0.5f + rnd() * 1.5f;
            const float D = 0.30f + rnd() * 0.6f;
            if (D >= 0.99f) {
                continue;
            }
            const bool decay_on = trial % 8 != 7;
            auto r = run_expansion_1t(p, N, K, T, D, !decay_on); // run_expansion_1t takes no_decay
            auto ref = ref_expansion(p.data(), N, K, T, D, decay_on);
            const bool same = r.counts[0] == (float) ref.first && vec_close(r.w, ref.second, 1e-4f) &&
                sum1(r.w, N, ref.first, 1e-4f);
            if (!same) {
                printf("    sweep trial %d failed (N=%d K=%d T=%g D=%g c=%d)\n",
                        trial, N, K, T, D, ref.first);
                ok = false;
            }
        }
        expect(ok, "32-config sweep matches reference");
    }

    if (n_fail == 0) {
        printf("test-moe-expansion: ALL PASS\n");
        return 0;
    }
    printf("test-moe-expansion: %d FAILURES\n", n_fail);
    return 1;
}
