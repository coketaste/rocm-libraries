/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal-only definitions for rocDENSITYMAT implementations.
 * ************************************************************************ */

#ifndef ROCDENSITYMAT_INTERNAL_HPP
#define ROCDENSITYMAT_INTERNAL_HPP

#include "rocdensitymat.h"

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

#include <atomic>
#include <complex>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocdensitymat
{

constexpr int    default_threads_per_block = 256;
constexpr int    max_factors_per_term      = 32;
constexpr size_t default_workspace_align   = 256;

/*! \brief Logger state shared across the process; protected by a coarse mutex.
 *  Logging is rare so contention is irrelevant. */
struct logger_state
{
    std::mutex                              mu;
    rocdensitymat_logger_callback_t         cb       = nullptr;
    rocdensitymat_logger_callback_data_t    cb_data  = nullptr;
    void*                                   user     = nullptr;
    void*                                   file     = nullptr;
    int32_t                                 level    = 0;
    int32_t                                 mask     = 0;
    bool                                    disabled = false;
};

logger_state& global_logger();

/*! \brief Library context. Stored on the heap, exposed only by opaque pointer. */
struct handle_impl
{
    hipStream_t                            stream    = nullptr;
    rocdensitymat_device_mem_handler_t     mem       = {};
    bool                                   has_mem   = false;
    int                                    device_id = 0;
    uint32_t                               seed      = 0;
};

inline size_t element_size_bytes(rocdensitymat_data_type t)
{
    switch(t)
    {
    case ROCDENSITYMAT_C_64F: return 16;
    case ROCDENSITYMAT_C_32F: return 8;
    case ROCDENSITYMAT_R_64F: return 8;
    case ROCDENSITYMAT_R_32F: return 4;
    }
    return 0;
}

template <typename T>
inline T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

#define ROCDENSITYMAT_HIP_CHECK(expr)                                         \
    do {                                                                      \
        hipError_t _err = (expr);                                             \
        if(_err != hipSuccess)                                                \
        {                                                                     \
            return ROCDENSITYMAT_STATUS_HIP_ERROR;                            \
        }                                                                     \
    } while(0)

#define ROCDENSITYMAT_CHECK_HANDLE(h)                                         \
    do {                                                                      \
        if((h) == nullptr) return ROCDENSITYMAT_STATUS_NOT_INITIALIZED;       \
    } while(0)

#define ROCDENSITYMAT_CHECK_PTR(p)                                            \
    do {                                                                      \
        if((p) == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;         \
    } while(0)

#define ROCDENSITYMAT_RC_CHECK(expr)                                          \
    do {                                                                      \
        rocdensitymat_status _rc = (expr);                                    \
        if(_rc != ROCDENSITYMAT_STATUS_SUCCESS) return _rc;                   \
    } while(0)

inline size_t align_up(size_t v, size_t align)
{
    return (v + align - 1) & ~(align - 1);
}

inline rocdensitymat_status
dev_alloc(handle_impl* h, void** out, size_t bytes, hipStream_t stream)
{
    if(out == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    if(bytes == 0) { *out = nullptr; return ROCDENSITYMAT_STATUS_SUCCESS; }
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_alloc(h->mem.ctx, out, bytes, stream);
        return (rc == 0) ? ROCDENSITYMAT_STATUS_SUCCESS
                         : ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipMallocAsync(out, bytes, stream);
    return (rc == hipSuccess) ? ROCDENSITYMAT_STATUS_SUCCESS
                              : ROCDENSITYMAT_STATUS_ALLOC_FAILED;
}

inline rocdensitymat_status
dev_free(handle_impl* h, void* ptr, size_t bytes, hipStream_t stream)
{
    if(ptr == nullptr) return ROCDENSITYMAT_STATUS_SUCCESS;
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_free(h->mem.ctx, ptr, bytes, stream);
        return (rc == 0) ? ROCDENSITYMAT_STATUS_SUCCESS
                         : ROCDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipFreeAsync(ptr, stream);
    return (rc == hipSuccess) ? ROCDENSITYMAT_STATUS_SUCCESS
                              : ROCDENSITYMAT_STATUS_EXECUTION_FAILED;
}

/*! \brief Bump-allocator that prefers a user-supplied workspace buffer and
 *  falls back to `dev_alloc` on overflow. */
class dev_arena
{
public:
    dev_arena(handle_impl* h,
              hipStream_t  stream,
              void*        user_ws       = nullptr,
              size_t       user_ws_bytes = 0)
        : h_(h)
        , stream_(stream)
        , user_ws_(static_cast<char*>(user_ws))
        , user_ws_bytes_(user_ws ? user_ws_bytes : 0)
        , user_offset_(0)
    {
    }

    ~dev_arena() { release_all(); }

    dev_arena(const dev_arena&)            = delete;
    dev_arena& operator=(const dev_arena&) = delete;

    rocdensitymat_status alloc(void** out, size_t bytes)
    {
        if(out == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
        if(bytes == 0) { *out = nullptr; return ROCDENSITYMAT_STATUS_SUCCESS; }

        size_t off = align_up(user_offset_, default_workspace_align);
        size_t end = off + bytes;
        if(user_ws_ != nullptr && end <= user_ws_bytes_)
        {
            *out         = user_ws_ + off;
            user_offset_ = end;
            return ROCDENSITYMAT_STATUS_SUCCESS;
        }
        void* p = nullptr;
        auto  rc = dev_alloc(h_, &p, bytes, stream_);
        if(rc != ROCDENSITYMAT_STATUS_SUCCESS) return rc;
        heap_.emplace_back(p, bytes);
        *out = p;
        return ROCDENSITYMAT_STATUS_SUCCESS;
    }

    void release_all()
    {
        for(auto& kv : heap_)
        {
            (void)dev_free(h_, kv.first, kv.second, stream_);
        }
        heap_.clear();
    }

    hipStream_t  stream() const { return stream_; }
    handle_impl* h() const      { return h_; }

private:
    handle_impl*                            h_;
    hipStream_t                             stream_;
    char*                                   user_ws_;
    size_t                                  user_ws_bytes_;
    size_t                                  user_offset_;
    std::vector<std::pair<void*, size_t>>   heap_;
};

} // namespace rocdensitymat

#endif /* ROCDENSITYMAT_INTERNAL_HPP */
