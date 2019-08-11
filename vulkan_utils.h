#pragma once

#include "result.h"
#include <string>
#include <vulkan/vulkan.h>

namespace pk
{

class VulkanUtils {
public:
    static result createComputeShader( const VkDevice& device, const std::string& shaderPath, VkShaderModule* pShader );
    static bool   createBuffer( const VkDevice& device, const VkPhysicalDevice& physicalDevice, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
};

} // namespace pk
