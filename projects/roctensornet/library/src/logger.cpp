/* ************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc. All rights Reserved.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "roctensornet_internal.hpp"

namespace roctensornet
{
logger_state& global_logger()
{
    static logger_state g;
    return g;
}
} // namespace roctensornet

using namespace roctensornet;

extern "C" {

roctensornet_status roctensornet_logger_set_callback(roctensornet_logger_callback_t cb)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = cb;
    g.cb_data = nullptr;
    g.user    = nullptr;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status
roctensornet_logger_set_callback_data(roctensornet_logger_callback_data_t cb, void* user)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.cb      = nullptr;
    g.cb_data = cb;
    g.user    = user;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_logger_set_file(void* file)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.file = file;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_logger_open_file(const char* file_name)
{
    ROCTENSORNET_CHECK_PTR(file_name);
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    FILE* f = std::fopen(file_name, "a");
    if(f == nullptr) return ROCTENSORNET_STATUS_IO_ERROR;
    g.file = f;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_logger_set_level(int32_t level)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.level = level;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_logger_set_mask(int32_t mask)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.mask = mask;
    return ROCTENSORNET_STATUS_SUCCESS;
}

roctensornet_status roctensornet_logger_force_disable(void)
{
    auto& g = global_logger();
    std::lock_guard<std::mutex> lk(g.mu);
    g.disabled = true;
    return ROCTENSORNET_STATUS_SUCCESS;
}

} // extern "C"
