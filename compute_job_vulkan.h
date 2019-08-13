#pragma once

#include "vulkan_utils.h"

#include <vulkan/vulkan.h>


namespace pk
{

class IComputeJobVulkan {
public:
    // Mandatory Vulkan members; assigned by the Vulkan implementation of computeSubmitJob()
    VulkanContext vulkan;
};

} // namespace pk
