#pragma once

#include "result.h"

#include <string>
#include <vulkan/vulkan.h>

namespace pk
{

struct VulkanContext {
    VkDevice         device;
    VkPhysicalDevice physicalDevice;
    VkDescriptorPool descriptorPool;
    VkCommandPool    commandPool;
    VkQueue          queue;
};


class VulkanUtils {
public:
    enum ShaderBufferType {
        BUFFER_TYPE_UNKNOWN           = 0,
        BUFFER_COMPUTE_SHADER_UNIFORM = 1,
        BUFFER_COMPUTE_SHADER_STORAGE = 2,
    };

    enum ShaderBufferVisibility {
        BUFFER_VISIBILITY_UNKNOWN = 0,
        BUFFER_SHARED             = 1,
        BUFFER_DEVICE             = 2,
    };

    struct ShaderBufferInfo {
        uint32_t               binding;
        const char*            name;
        ShaderBufferType       type;
        size_t                 size;
        VkBuffer*              pBuffer;
        VkDeviceMemory*        pBufferMemory;
        ShaderBufferVisibility visibility;
    };

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
        ComputeShaderProgram* program;
        uint32_t              numBuffers;
        ShaderBufferInfo*     pBuffers;
        VkDescriptorSet       descriptorSet;
        VkCommandBuffer       commandBuffer;
        VkFence               fence;
    };

    static result createComputeShader( VulkanContext& context, ComputeShaderInstance* pComputeShader );


    // Following functions are called by createComputeShader(); you rarely need to call them yourself

    static result createComputeShaderProgram( VulkanContext& context, ComputeShaderProgram* pComputeShaderProgram );
    static result createDescriptorSetLayout( VulkanContext& context, ComputeShaderInstance* pComputeShader );
    static result createComputePipeline( VulkanContext& context, ComputeShaderProgram* pComputeShaderProgram );

    static result createShaderBuffers( VulkanContext& context, ComputeShaderInstance* pComputeShader );
    static result createBuffers( VulkanContext& context, ShaderBufferInfo* pBuffers, uint32_t count );
    static result createBuffer( VulkanContext& context, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
    static result createDescriptorSet( VulkanContext& context, ComputeShaderInstance* pComputeShader );
    static result recordCommandBuffer( VulkanContext& context, ComputeShaderInstance* pShader );
    static result createFence( VulkanContext& context, ComputeShaderInstance* pShader );
};

} // namespace pk
