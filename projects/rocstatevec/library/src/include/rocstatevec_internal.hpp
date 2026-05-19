/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Internal-only definitions for rocSTATEVEC implementations.
 * ************************************************************************ */

#ifndef ROCSTATEVEC_INTERNAL_HPP
#define ROCSTATEVEC_INTERNAL_HPP

#include "rocstatevec.h"

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocstatevec
{

constexpr int    default_threads_per_block = 256;
constexpr int    max_targets_per_apply     = 32;
constexpr size_t default_workspace_align   = 256;

/*! \brief Logger state shared across the process.
 *
 *  Logger configuration is global in cuStateVec; we mirror that. The
 *  members are protected by a coarse mutex; logging is rare so contention
 *  is irrelevant. */
struct logger_state
{
    std::mutex                            mu;
    rocstatevec_logger_callback_t         cb       = nullptr;
    rocstatevec_logger_callback_data_t    cb_data  = nullptr;
    void*                                 user     = nullptr;
    void*                                 file     = nullptr;
    int32_t                               level    = 0;
    int32_t                               mask     = 0;
    bool                                  disabled = false;
};

logger_state& global_logger();

/*! \brief Library context. Stored on the heap, exposed only by opaque pointer. */
struct handle
{
    hipStream_t                            stream    = nullptr;
    rocstatevec_device_mem_handler_t       mem       = {};
    bool                                   has_mem   = false;
    int                                    device_id = 0;

    handle() noexcept = default;
};

/*! \brief Sampler descriptor (defined in src/sampler.cpp). */
struct sampler_descriptor;

/*! \brief Accessor descriptor (defined in src/accessor.cpp). */
struct accessor_descriptor;

/*! \brief Map a `rocstatevec_data_type` enum value to its element size in bytes.
 *  Returns 0 for unsupported / non-complex types. */
inline size_t element_size_bytes(rocstatevec_data_type t)
{
    switch(t)
    {
    case ROCSTATEVEC_C_64F: return 16;
    case ROCSTATEVEC_C_32F: return 8;
    case ROCSTATEVEC_R_64F: return 8;
    case ROCSTATEVEC_R_32F: return 4;
    }
    return 0;
}

/*! \brief Convert a count of qubits to a state-vector length (2^n). */
inline rocstatevec_index_t state_vector_length(uint32_t n_index_bits)
{
    return rocstatevec_index_t{1} << n_index_bits;
}

/*! \brief Round-up integer division for kernel grid sizing. */
template <typename T>
inline T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

/*! \brief Standard error-translation helper for HIP runtime calls. */
#define ROCSTATEVEC_HIP_CHECK(expr)                                          \
    do {                                                                     \
        hipError_t _err = (expr);                                            \
        if(_err != hipSuccess)                                               \
        {                                                                    \
            return ROCSTATEVEC_STATUS_EXECUTION_FAILED;                      \
        }                                                                    \
    } while(0)

/*! \brief Validate that a handle pointer is non-null. */
#define ROCSTATEVEC_CHECK_HANDLE(h)                                          \
    do {                                                                     \
        if((h) == nullptr) return ROCSTATEVEC_STATUS_NOT_INITIALIZED;        \
    } while(0)

/*! \brief Validate a generic non-null pointer argument. */
#define ROCSTATEVEC_CHECK_PTR(p)                                             \
    do {                                                                     \
        if((p) == nullptr) return ROCSTATEVEC_STATUS_INVALID_VALUE;          \
    } while(0)

/*! \brief Best-effort host vs device pointer classification.
 *
 *  Returns true when the pointer was allocated with `hipMalloc`,
 *  `hipMallocManaged`, or a user-supplied device allocator backed by
 *  device memory. Returns false for host pointers, invalid pointers, or
 *  any case in which the runtime cannot answer the query (we default to
 *  "host" in the unknown case so the caller falls back to an explicit
 *  H2D copy, which is always safe).
 */
inline bool is_device_pointer(const void* p)
{
    if(p == nullptr) return false;
    hipPointerAttribute_t attr{};
    hipError_t            rc = hipPointerGetAttributes(&attr, p);
    if(rc != hipSuccess)
    {
        // Clear the sticky last-error so it doesn't poison the next HIP call.
        (void)hipGetLastError();
        return false;
    }
    // Modern HIP (ROCm 5.0+) uses `attr.type`. Older CUDA-style HIP
    // exposed `attr.memoryType`; rocm-libraries targets ROCm 6.x+ so we
    // unconditionally use the modern field.
    return attr.type == hipMemoryTypeDevice
        || attr.type == hipMemoryTypeManaged;
}

/*! \brief Allocate `bytes` of device memory through the handle's
 *  user-supplied device-mem handler when one is bound, otherwise via
 *  `hipMallocAsync`. Always async on `stream`. */
inline rocstatevec_status
dev_alloc(handle* h, void** out, size_t bytes, hipStream_t stream)
{
    if(out == nullptr) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    if(bytes == 0)     { *out = nullptr; return ROCSTATEVEC_STATUS_SUCCESS; }
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_alloc(h->mem.ctx, out, bytes, stream);
        return (rc == 0) ? ROCSTATEVEC_STATUS_SUCCESS
                         : ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipMallocAsync(out, bytes, stream);
    return (rc == hipSuccess) ? ROCSTATEVEC_STATUS_SUCCESS
                              : ROCSTATEVEC_STATUS_ALLOC_FAILED;
}

/*! \brief Free a device buffer allocated by `dev_alloc`. Best-effort,
 *  honors the handle's device-mem handler when bound. */
inline rocstatevec_status
dev_free(handle* h, void* ptr, size_t bytes, hipStream_t stream)
{
    if(ptr == nullptr) return ROCSTATEVEC_STATUS_SUCCESS;
    if(h != nullptr && h->has_mem)
    {
        int rc = h->mem.device_free(h->mem.ctx, ptr, bytes, stream);
        return (rc == 0) ? ROCSTATEVEC_STATUS_SUCCESS
                         : ROCSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR;
    }
    hipError_t rc = hipFreeAsync(ptr, stream);
    return (rc == hipSuccess) ? ROCSTATEVEC_STATUS_SUCCESS
                              : ROCSTATEVEC_STATUS_EXECUTION_FAILED;
}

inline size_t align_up(size_t v, size_t align)
{
    return (v + align - 1) & ~(align - 1);
}

/*! \brief Bump-allocator that prefers a user-supplied workspace buffer
 *  and falls back to `dev_alloc` on overflow.
 *
 *  Lifecycle:
 *    1. Construct with `(handle*, stream, user_ws, user_ws_bytes)`.
 *    2. Call `alloc(&p, n)` zero or more times. Each `alloc` either
 *       carves out a slice of the user workspace (no real allocation)
 *       or routes through `dev_alloc` and tracks the pointer.
 *    3. The destructor `dev_free`s every tracked pointer on the bound
 *       stream. Slices into the user workspace are not freed (the user
 *       owns the buffer).
 *
 *  The arena does not move ownership of the user workspace pointer.
 *  Callers may pass `(nullptr, 0)` to force every alloc through
 *  `dev_alloc`. Allocations are aligned to `default_workspace_align`. */
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

    /*! \brief Allocate `bytes` bytes from the arena. */
    rocstatevec_status alloc(void** out, size_t bytes)
    {
        if(out == nullptr) return ROCSTATEVEC_STATUS_INVALID_VALUE;
        if(bytes == 0) { *out = nullptr; return ROCSTATEVEC_STATUS_SUCCESS; }

        size_t off  = align_up(user_offset_, default_workspace_align);
        size_t end  = off + bytes;
        if(user_ws_ != nullptr && end <= user_ws_bytes_)
        {
            *out         = user_ws_ + off;
            user_offset_ = end;
            return ROCSTATEVEC_STATUS_SUCCESS;
        }
        void* p = nullptr;
        auto  rc = dev_alloc(h_, &p, bytes, stream_);
        if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;
        heap_.emplace_back(p, bytes);
        *out = p;
        return ROCSTATEVEC_STATUS_SUCCESS;
    }

    /*! \brief Convenience: allocate + H2D copy. The src buffer is treated
     *  as host memory; if you have a device pointer, just keep using it. */
    rocstatevec_status alloc_and_copy(void** out, const void* src, size_t bytes)
    {
        auto rc = alloc(out, bytes);
        if(rc != ROCSTATEVEC_STATUS_SUCCESS) return rc;
        if(bytes == 0) return ROCSTATEVEC_STATUS_SUCCESS;
        if(hipMemcpyAsync(*out, src, bytes, hipMemcpyHostToDevice, stream_)
           != hipSuccess)
        {
            return ROCSTATEVEC_STATUS_EXECUTION_FAILED;
        }
        return ROCSTATEVEC_STATUS_SUCCESS;
    }

    /*! \brief Free every allocation that came from `dev_alloc`. */
    void release_all()
    {
        for(auto& kv : heap_)
        {
            (void)dev_free(h_, kv.first, kv.second, stream_);
        }
        heap_.clear();
    }

    hipStream_t stream() const { return stream_; }
    handle*     h() const      { return h_; }

private:
    handle*                                 h_;
    hipStream_t                             stream_;
    char*                                   user_ws_;
    size_t                                  user_ws_bytes_;
    size_t                                  user_offset_;
    std::vector<std::pair<void*, size_t>>   heap_;
};

/*! \brief Same shape as `ROCSTATEVEC_HIP_CHECK` but for arena/dev_alloc
 *  return values that are already `rocstatevec_status`. */
#define ROCSTATEVEC_RC_CHECK(expr)                                           \
    do {                                                                     \
        rocstatevec_status _rc = (expr);                                     \
        if(_rc != ROCSTATEVEC_STATUS_SUCCESS) return _rc;                    \
    } while(0)

} // namespace rocstatevec

#endif /* ROCSTATEVEC_INTERNAL_HPP */
