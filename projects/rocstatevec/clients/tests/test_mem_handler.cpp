/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Tracks every internal device-memory allocation rocSTATEVEC performs
 * by installing a `rocstatevec_device_mem_handler_t` and asserting the
 * counters move during apply / sampler operations.
 * ************************************************************************ */

#include "test_helpers.hpp"

#include <gtest/gtest.h>
#include <rocstatevec.h>

#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>

using namespace rocstatevec::test;
using cd = std::complex<double>;

namespace
{
struct counters
{
    std::atomic<size_t> n_allocs{0};
    std::atomic<size_t> n_frees{0};
    std::atomic<size_t> bytes_inflight{0};
    std::atomic<size_t> bytes_peak{0};
};

int track_alloc(void* ctx, void** ptr, size_t size, hipStream_t stream)
{
    auto* c = reinterpret_cast<counters*>(ctx);
    if(hipMallocAsync(ptr, size, stream) != hipSuccess) return -1;
    c->n_allocs.fetch_add(1, std::memory_order_relaxed);
    size_t cur = c->bytes_inflight.fetch_add(size, std::memory_order_relaxed) + size;
    size_t prev = c->bytes_peak.load(std::memory_order_relaxed);
    while(cur > prev
          && !c->bytes_peak.compare_exchange_weak(prev, cur, std::memory_order_relaxed))
    {
    }
    return 0;
}

int track_free(void* ctx, void* ptr, size_t size, hipStream_t stream)
{
    auto* c = reinterpret_cast<counters*>(ctx);
    if(hipFreeAsync(ptr, stream) != hipSuccess) return -1;
    c->n_frees.fetch_add(1, std::memory_order_relaxed);
    c->bytes_inflight.fetch_sub(size, std::memory_order_relaxed);
    return 0;
}
} // namespace

TEST(MemHandler, ApplyMatrixCallsCustomAllocator)
{
    if(skip_if_no_gpu()) GTEST_SKIP();

    counters                            c{};
    rocstatevec_device_mem_handler_t mem{};
    mem.ctx          = &c;
    mem.device_alloc = track_alloc;
    mem.device_free  = track_free;
    std::strcpy(mem.name, "tracker");

    handle_guard hg; ASSERT_NE(hg.h, nullptr);
    ASSERT_EQ(rocstatevec_set_device_mem_handler(hg.h, &mem),
              ROCSTATEVEC_STATUS_SUCCESS);

    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;

    device_buffer<cd> dsv(N);
    ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
              ROCSTATEVEC_STATUS_SUCCESS);

    const double s = 1.0 / std::sqrt(2.0);
    cd      H_mat[4] = {{s, 0}, {s, 0}, {s, 0}, {-s, 0}};
    int32_t q0       = 0;
    ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                       H_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q0, 1, nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, nullptr, 0),
              ROCSTATEVEC_STATUS_SUCCESS);

    EXPECT_GT(c.n_allocs.load(), 0u);
    EXPECT_EQ(c.n_allocs.load(), c.n_frees.load());
    EXPECT_EQ(c.bytes_inflight.load(), 0u);
}

TEST(MemHandler, NotCalledWhenUserWorkspaceCoversEverything)
{
    if(skip_if_no_gpu()) GTEST_SKIP();

    counters                            c{};
    rocstatevec_device_mem_handler_t mem{};
    mem.ctx          = &c;
    mem.device_alloc = track_alloc;
    mem.device_free  = track_free;
    std::strcpy(mem.name, "tracker");

    handle_guard hg; ASSERT_NE(hg.h, nullptr);
    ASSERT_EQ(rocstatevec_set_device_mem_handler(hg.h, &mem),
              ROCSTATEVEC_STATUS_SUCCESS);

    constexpr uint32_t n = 3;
    constexpr size_t   N = size_t{1} << n;

    size_t bytes = 0;
    ASSERT_EQ(rocstatevec_apply_matrix_get_workspace_size(hg.h, ROCSTATEVEC_C_64F, n,
                                                          nullptr, ROCSTATEVEC_C_64F,
                                                          ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                                          1, 0,
                                                          ROCSTATEVEC_COMPUTE_64F, &bytes),
              ROCSTATEVEC_STATUS_SUCCESS);
    ASSERT_GT(bytes, 0u);

    void* d_ws = nullptr;
    ASSERT_EQ(hipMalloc(&d_ws, bytes), hipSuccess);

    device_buffer<cd> dsv(N);
    ASSERT_EQ(rocstatevec_initialize_state_vector(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                                  ROCSTATEVEC_STATE_VECTOR_TYPE_ZERO),
              ROCSTATEVEC_STATUS_SUCCESS);

    const double s = 1.0 / std::sqrt(2.0);
    cd      H_mat[4] = {{s, 0}, {s, 0}, {s, 0}, {-s, 0}};
    int32_t q0       = 0;
    ASSERT_EQ(rocstatevec_apply_matrix(hg.h, dsv.ptr, ROCSTATEVEC_C_64F, n,
                                       H_mat, ROCSTATEVEC_C_64F,
                                       ROCSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                       &q0, 1, nullptr, nullptr, 0,
                                       ROCSTATEVEC_COMPUTE_64F, d_ws, bytes),
              ROCSTATEVEC_STATUS_SUCCESS);

    // The user workspace fully covers the call, so no fall-back
    // allocations should hit the tracker.
    EXPECT_EQ(c.n_allocs.load(), 0u);
    EXPECT_EQ(c.n_frees.load(), 0u);

    // Ensure the kernel and DtoD copy that consumed d_ws have finished
    // before we hand the buffer back to the runtime allocator.
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    hipFree(d_ws);
}
