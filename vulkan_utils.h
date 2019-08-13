#pragma once

#include "compute.h"
#include "compute_buffer.h"
#include "result.h"

#include <string>
#include <vulkan/vulkan.h>

namespace pk
{

struct VulkanContext {
    SpinLock*        pSpinlock;
    VkDevice         device;
    VkPhysicalDevice physicalDevice;
    VkDescriptorPool descriptorPool;
    VkCommandPool    commandPool; // TODO: a commandPool should be created and bound per-thread
    VkQueue          queue;
};

class VulkanUtils {
public:
    // All instances of a shader share the same program / pipeline
    struct ComputeShaderProgram {
        const char*           shaderPath;
        VkShaderModule        shaderModule;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipeline            pipeline;
        VkPipelineLayout      pipelineLayout;
        uint32_t              workgroupSize;
        uint32_t              workgroupWidth;
        uint32_t              workgroupHeight;
        uint32_t              workgroupDepth;
    };

    // Each instance of a shader allocates its own buffers and descriptors
    struct ComputeShaderInstance {
        SpinLock              spinlock;
        ComputeShaderProgram* pProgram;
        IComputeBuffer**      ppBuffers;
        uint32_t              numBuffers;
        VkDescriptorSet       descriptorSet;
        VkCommandBuffer       commandBuffer;
        VkFence               fence;
    };

    static result createComputeShader( VulkanContext& context, ComputeShaderInstance* pComputeShader );
    static result createBuffer( VulkanContext& context, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
    static result recordCommandBuffer( VulkanContext& context, ComputeShaderInstance* pComputeShader );
    static result createFence( VulkanContext& context, ComputeShaderInstance* pComputeShader );
};

} // namespace pk
