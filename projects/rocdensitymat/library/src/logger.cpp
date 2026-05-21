/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "rocdensitymat_internal.hpp"

#include <cstdio>

namespace rocdensitymat
{

logger_state& global_logger()
{
    static logger_state instance;
    return instance;
}

} // namespace rocdensitymat

extern "C" {

rocdensitymat_status rocdensitymat_logger_set_callback(
    rocdensitymat_logger_callback_t cb)
{
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = cb;
    g.cb_data = nullptr;
    g.user    = nullptr;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_set_callback_data(
    rocdensitymat_logger_callback_data_t cb, void* user_data)
{
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = nullptr;
    g.cb_data = cb;
    g.user    = user_data;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_set_file(void* file)
{
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.file = file;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_open_file(const char* path)
{
    if(path == nullptr) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    FILE* f = std::fopen(path, "a");
    if(f == nullptr) return ROCDENSITYMAT_STATUS_IO_ERROR;
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    if(g.file != nullptr && g.file != stdout && g.file != stderr)
    {
        std::fclose(static_cast<FILE*>(g.file));
    }
    g.file = f;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_set_level(int32_t level)
{
    if(level < 0 || level > 5) return ROCDENSITYMAT_STATUS_INVALID_VALUE;
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.level = level;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_set_mask(int32_t mask)
{
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.mask = mask;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

rocdensitymat_status rocdensitymat_logger_force_disable(void)
{
    auto& g = rocdensitymat::global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.disabled = true;
    g.cb       = nullptr;
    g.cb_data  = nullptr;
    g.user     = nullptr;
    return ROCDENSITYMAT_STATUS_SUCCESS;
}

} // extern "C"
