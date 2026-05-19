/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Process-global logger plumbing.
 * ************************************************************************ */

#include "rocstatevec_internal.hpp"

namespace rocstatevec
{
logger_state& global_logger()
{
    static logger_state s;
    return s;
}
} // namespace rocstatevec

using namespace rocstatevec;

extern "C" rocstatevec_status rocstatevec_logger_set_callback(rocstatevec_logger_callback_t cb)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = cb;
    g.cb_data = nullptr;
    g.user    = nullptr;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_set_callback_data(
    rocstatevec_logger_callback_data_t cb, void* user)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = nullptr;
    g.cb_data = cb;
    g.user    = user;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_set_file(void* file)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.file = file;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_open_file(const char* path)
{
    if(path == nullptr) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    FILE* fp = std::fopen(path, "w");
    if(fp == nullptr) return ROCSTATEVEC_STATUS_INTERNAL_ERROR;
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.file = fp;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_set_level(int32_t level)
{
    if(level < 0 || level > 5) return ROCSTATEVEC_STATUS_INVALID_VALUE;
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.level = level;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_set_mask(int32_t mask)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.mask = mask;
    return ROCSTATEVEC_STATUS_SUCCESS;
}

extern "C" rocstatevec_status rocstatevec_logger_force_disable(void)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.disabled = true;
    return ROCSTATEVEC_STATUS_SUCCESS;
}
