#include "test_helpers.hpp"

#include <cstring>

TEST(HipTNStatus, ErrorStrings)
{
    EXPECT_STRNE(hiptensornet_get_error_string(HIPTENSORNET_STATUS_SUCCESS), "");
    EXPECT_STRNE(hiptensornet_get_error_string(HIPTENSORNET_STATUS_NOT_SUPPORTED), "");
    EXPECT_STRNE(hiptensornet_get_error_string(HIPTENSORNET_STATUS_INVALID_VALUE), "");
}
