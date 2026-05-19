/* ************************************************************************
 * ghz_sample.cpp
 *
 * Sketch: prepare a 3-qubit GHZ state via NetworkState gate application.
 * This sample exercises the gate-recording API; the full state
 * synthesizer (`roctensornet_state_compute`) returns NOT_SUPPORTED
 * in v0.1.0, so the sample just verifies the build pipeline.
 * ************************************************************************ */

#include "roctensornet.h"
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

int main()
{
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess || n <= 0) return 0;
    roctensornet_handle h = nullptr;
    roctensornet_create(&h);

    roctensornet_index_t qd[3] = {2, 2, 2};
    roctensornet_state s = nullptr;
    roctensornet_create_state(h, ROCTENSORNET_STATE_PURITY_PURE, 3, qd, ROCTENSORNET_C_64F, &s);

    int32_t q0[1] = {0}, q01[2] = {0, 1}, q12[2] = {1, 2};
    int64_t id = -1;
    /* H on qubit 0 */
    roctensornet_state_apply_tensor_operator(h, s, 1, q0, nullptr, nullptr, 1, 0, 1, &id);
    /* CX(0->1) */
    roctensornet_state_apply_tensor_operator(h, s, 2, q01, nullptr, nullptr, 1, 0, 1, &id);
    /* CX(1->2) */
    roctensornet_state_apply_tensor_operator(h, s, 2, q12, nullptr, nullptr, 1, 0, 1, &id);

    std::printf("GHZ-state gate sequence recorded (3 gates).\n");

    roctensornet_destroy_state(s);
    roctensornet_destroy(h);
    return 0;
}
