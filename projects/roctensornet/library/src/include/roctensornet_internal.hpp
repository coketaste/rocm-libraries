/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal-only definitions for rocTENSORNET implementations.
 *
 * Mirrors the discipline of rocSTATEVEC: a single `handle` struct, a
 * `dev_arena` bump-allocator that prefers user workspace and falls back
 * to the user-registered device memory handler, host/device pointer
 * detection, and a small set of error-translation macros.
 * ************************************************************************ */

#ifndef ROCTENSORNET_INTERNAL_HPP
#define ROCTENSORNET_INTERNAL_HPP

#include "roctensornet.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace roctensornet
{

constexpr int    default_threads_per_block = 256;
constexpr size_t default_workspace_align   = 256;
constexpr int    max_supported_modes       = 64;

/*! \brief Logger state shared across the process. */
struct logger_state
{
    std::mutex                            mu;
    roctensornet_logger_callback_t        cb        = nullptr;
    roctensornet_logger_callback_data_t   cb_data   = nullptr;
    void*                                 user      = nullptr;
    void*                                 file      = nullptr;
    int32_t                               level     = 0;
    int32_t                               mask      = 0;
    bool                                  disabled  = false;
};

logger_state& global_logger();

/*! \brief Library context. Stored on the heap and exposed by opaque pointer.
 *
 *  v0.1.0 implements contractions and tensor decompositions with direct
 *  HIP kernels and tightly scoped host helpers; hipTENSOR / rocSOLVER
 *  back-ends are wired by future revisions and only require recompile.
 *  The handle struct already carries opaque slots so the ABI is stable.
 */
struct handle
{
    hipStream_t                          stream    = nullptr;
    roctensornet_device_mem_handler_t    mem       = {};
    bool                                 has_mem   = false;
    int                                  device_id = 0;

    /* Reserved for future hipTENSOR / rocSOLVER sub-handles. */
    void*  hiptensor_handle  = nullptr;
    void*  rocblas_handle    = nullptr;

    handle() noexcept = default;
};

/*! \brief Map a `roctensornet_data_type` enum value to its element size in bytes.
 *  Returns 0 for unsupported types. */
inline size_t element_size_bytes(roctensornet_data_type t)
{
    switch(t)
    {
    case ROCTENSORNET_R_32F:  return 4;
    case ROCTENSORNET_R_64F:  return 8;
    case ROCTENSORNET_C_32F:  return 8;
    case ROCTENSORNET_C_64F:  return 16;
    case ROCTENSORNET_R_16F:  return 2;
    case ROCTENSORNET_R_16BF: return 2;
    }
    return 0;
}

/*! \brief True for complex element types. */
inline bool is_complex_dtype(roctensornet_data_type t)
{
    return t == ROCTENSORNET_C_32F || t == ROCTENSORNET_C_64F;
}

/*! \brief Round-up integer division. */
template <typename T>
inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

/*! \brief Standard error-translation helper for HIP runtime calls. */
#define ROCTENSORNET_HIP_CHECK(expr)                                          \
    do {                                                                      \
        hipError_t _err = (expr);                                             \
        if(_err != hipSuccess) return ROCTENSORNET_STATUS_EXECUTION_FAILED;   \
    } while(0)

/*! \brief Validate that a handle pointer is non-null. */
#define ROCTENSORNET_CHECK_HANDLE(h)                                          \
    do {                                                                      \
        if((h) == nullptr) return ROCTENSORNET_STATUS_NOT_INITIALIZED;        \
    } while(0)

/*! \brief Validate a generic non-null pointer argument. */
#define ROCTENSORNET_CHECK_PTR(p)                                             \
    do {                                                                      \
        if((p) == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;          \
    } while(0)

/*! \brief Forward roctensornet_status returns. */
#define ROCTENSORNET_RC_CHECK(expr)                                           \
    do {                                                                      \
        roctensornet_status _rc = (expr);                                     \
        if(_rc != ROCTENSORNET_STATUS_SUCCESS) return _rc;                    \
    } while(0)

/*! \brief Best-effort host vs device pointer classifier.
 *
 *  Returns true when the pointer was allocated with `hipMalloc`,
 *  `hipMallocManaged`, or a user-supplied device allocator backed by
 *  device memory; false otherwise (host pointers, invalid pointers,
 *  unknown). Identical contract to rocSTATEVEC's helper.
 */
inline bool is_device_pointer(const void* p)
{
    if(p == nullptr) return false;
    hipPointerAttribute_t attr{};
    hipError_t            rc = hipPointerGetAttributes(&attr, p);
    if(rc != hipSuccess)
    {
        (void)hipGetLastError();
        return false;
    }
    return attr.type == hipMemoryTypeDevice
        || attr.type == hipMemoryTypeManaged;
}

/*! \brief Allocate `bytes` of device memory through the handle's
 *  user-supplied device-mem handler when one is bound, otherwise via
 *  `hipMallocAsync`. Always async on `stream`. */
inline roctensornet_status
dev_alloc(handle* h, void** out, size_t bytes, hipStream_t stream)
{
    if(out == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
    if(bytes == 0)     { *out = nullptr; return ROCTENSORNET_STATUS_SUCCESS; }
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_alloc(h->mem.ctx, out, bytes, stream);
        return (rc == 0) ? ROCTENSORNET_STATUS_SUCCESS
                         : ROCTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipMallocAsync(out, bytes, stream);
    return (rc == hipSuccess) ? ROCTENSORNET_STATUS_SUCCESS
                              : ROCTENSORNET_STATUS_ALLOC_FAILED;
}

/*! \brief Free a device buffer allocated via `dev_alloc`. */
inline roctensornet_status
dev_free(handle* h, void* ptr, size_t bytes, hipStream_t stream)
{
    if(ptr == nullptr) return ROCTENSORNET_STATUS_SUCCESS;
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_free(h->mem.ctx, ptr, bytes, stream);
        return (rc == 0) ? ROCTENSORNET_STATUS_SUCCESS
                         : ROCTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipFreeAsync(ptr, stream);
    return (rc == hipSuccess) ? ROCTENSORNET_STATUS_SUCCESS
                              : ROCTENSORNET_STATUS_EXECUTION_FAILED;
}

inline size_t align_up(size_t v, size_t align)
{
    return (v + align - 1) & ~(align - 1);
}

/*! \brief Bump allocator that prefers a caller-supplied workspace
 *  buffer and falls back to `dev_alloc` on overflow.
 *
 *  Same contract as the rocSTATEVEC `dev_arena`: pointer-stable slices
 *  during the call; tracked overflow allocations are async-freed on
 *  the bound stream when the arena goes out of scope; the user buffer
 *  is never freed by the arena.
 */
class dev_arena
{
public:
    dev_arena(handle*     h,
              hipStream_t stream,
              void*       user_ws       = nullptr,
              size_t      user_ws_bytes = 0)
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

    roctensornet_status alloc(void** out, size_t bytes)
    {
        if(out == nullptr) return ROCTENSORNET_STATUS_INVALID_VALUE;
        if(bytes == 0) { *out = nullptr; return ROCTENSORNET_STATUS_SUCCESS; }

        size_t off = align_up(user_offset_, default_workspace_align);
        size_t end = off + bytes;
        if(user_ws_ != nullptr && end <= user_ws_bytes_)
        {
            *out         = user_ws_ + off;
            user_offset_ = end;
            return ROCTENSORNET_STATUS_SUCCESS;
        }
        void* p  = nullptr;
        auto  rc = dev_alloc(h_, &p, bytes, stream_);
        if(rc != ROCTENSORNET_STATUS_SUCCESS) return rc;
        heap_.emplace_back(p, bytes);
        *out = p;
        return ROCTENSORNET_STATUS_SUCCESS;
    }

    roctensornet_status alloc_and_copy(void** out, const void* src, size_t bytes)
    {
        auto rc = alloc(out, bytes);
        if(rc != ROCTENSORNET_STATUS_SUCCESS) return rc;
        if(bytes == 0) return ROCTENSORNET_STATUS_SUCCESS;
        if(hipMemcpyAsync(*out, src, bytes, hipMemcpyHostToDevice, stream_)
           != hipSuccess)
            return ROCTENSORNET_STATUS_EXECUTION_FAILED;
        return ROCTENSORNET_STATUS_SUCCESS;
    }

    void release_all()
    {
        for(auto& kv : heap_)
            (void)dev_free(h_, kv.first, kv.second, stream_);
        heap_.clear();
    }

    hipStream_t stream() const { return stream_; }
    handle*     h() const      { return h_; }
    size_t      used_user_bytes() const { return user_offset_; }
    size_t      num_overflow() const    { return heap_.size(); }

private:
    handle*                                 h_;
    hipStream_t                             stream_;
    char*                                   user_ws_;
    size_t                                  user_ws_bytes_;
    size_t                                  user_offset_;
    std::vector<std::pair<void*, size_t>>   heap_;
};

} // namespace roctensornet

#endif /* ROCTENSORNET_INTERNAL_HPP */
