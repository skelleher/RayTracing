#pragma once

#include "compute.h"
#include "spin_lock.h"

#include <string>
#include <vulkan/vulkan.h>


namespace pk
{

// Base class for compute jobs that may be submitted to a ComputeInstance.
//
// Assumes your compute shader has:
// . entry point named main()
// . single uniform buffer for input
// . single storage buffer for output
//

class ComputeJobVulkan : public IComputeJob {
public:
    virtual void create();                         // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job
    virtual void destroy();                        // clean up resources

    SpinLock      spinLock;
    compute_job_t handle;

    // TEST:
    std::string outputPath;
    bool        enableGammaCorrection;
    uint32_t    maxIterations;

    // Inherited from the owning ComputeInstance:
    compute_t        instance;
    VkDevice         device;
    VkPhysicalDevice physicalDevice;
    VkDescriptorPool descriptorPool;
    VkCommandPool    commandPool;
    VkQueue          queue;

    ComputeJobVulkan() :
        handle( IComputeJob::nextHandle++ ),
        workgroupWidth( -1 ),
        workgroupHeight( -1 ),
        workgroupDepth( -1 ),
        // TEST:
        outputPath( "mandelbrot.ppm" ),
        enableGammaCorrection( false ),
        maxIterations( 128 )
    {
    }

    virtual ~ComputeJobVulkan()
    {
        destroy();
        handle = INVALID_COMPUTE_JOB;
    }

protected:
    // *****************************************************************************
    // Common members and utility methods
    //
    // These generally don't need to change, unless your compute shader does
    // something unusual.
    // *****************************************************************************

    VkPipeline       pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule   computeShaderModule;
    VkCommandBuffer  commandBuffer;
    VkFence          fence;

    // Caller must delete[] the returned shader buffer
    uint32_t* _loadShader( const std::string& shaderPath, size_t* pShaderLength );
    uint32_t  _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties );
    bool      _createBuffer( VkDevice device, VkPhysicalDevice physicalDevice, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
    bool      _recordCommandBuffer();
    bool      _createComputePipeline();
    bool      _createFence();

protected:
    // *****************************************************************************
    // Methods and members below are shader-specific
    // *****************************************************************************

    uint32_t workgroupWidth;
    uint32_t workgroupHeight;
    uint32_t workgroupDepth;

    // Shader inputs
    VkBuffer       uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    uint32_t       uniformBufferSize;

    // Shader outputs
    VkBuffer       outputBuffer;
    VkDeviceMemory outputBufferMemory;
    uint32_t       outputBufferSize;

    VkDescriptorSet       descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    bool _createBuffers();
    bool _createDescriptorSetLayout();
    bool _createDescriptorSet();
};

} // namespace pk
