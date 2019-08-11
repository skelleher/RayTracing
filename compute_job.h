#pragma once

#include "assert.h"
#include "compute.h"
#include "spin_lock.h"
#include "thread_pool.h"

#include <string>
#include <vulkan/vulkan.h>


namespace pk
{

//
// Example Vulkan compute job.
// You should copy and modify it to do something interesting.
//
// Assumes your compute shader has:
// . entry point named main()
// . single uniform buffer for input
// . single storage buffer for output
//

class IComputeJobVulkan {
public:
    // Mandatory Vulkan members; assigned by the Vulkan implementation of computeSubmitJob()
    VkDevice         device;
    VkPhysicalDevice physicalDevice;
    VkDescriptorPool descriptorPool;
    VkCommandPool    commandPool;
    VkQueue          queue;
};


class ComputeJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    static std::unique_ptr<ComputeJob> create( compute_t hCompute ); // factory method

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    uint32_t outputWidth;
    uint32_t outputHeight;
    void     save( const std::string path );

protected:
    ComputeJob()                    = delete;
    ComputeJob( const ComputeJob& ) = delete;
    ComputeJob& operator=( const ComputeJob& ) = delete;

    ComputeJob( compute_t hCompute ) :
        IComputeJob( hCompute ),
        initialized( false ),
        workgroupWidth( -1 ),
        workgroupHeight( -1 ),
        workgroupDepth( -1 ),
        outputWidth( 640 ),
        outputHeight( 480 )
    {
        numInstances++;
    }

public:
    virtual ~ComputeJob()
    {
        _destroy();
        handle = INVALID_COMPUTE_JOB;
    }

protected:
    // *****************************************************************************
    // Common members and utility methods
    //
    // These generally don't need to change, unless your compute shader does
    // something unusual.
    // *****************************************************************************

    // Shared by all instances of this shader
    // NOTE: making these static assumes that all ComputeJobs of a given type are never submitted to a different ComputeInstance
    static std::atomic<bool>     firstInstance;
    static std::atomic<uint32_t> numInstances;
    static std::string           shaderPath;
    static VkShaderModule        computeShaderModule;
    static VkDescriptorSetLayout descriptorSetLayout;
    static VkPipeline            pipeline;
    static VkPipelineLayout      pipelineLayout;

protected:
    // *****************************************************************************
    // Methods and members below are shader-specific
    // *****************************************************************************

    bool initialized;
    bool destroyed;

    uint32_t workgroupSize;
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

    VkDescriptorSet descriptorSet;
    VkCommandBuffer commandBuffer;
    VkFence         fence;

    bool _createBuffers();
    bool _createDescriptorSetLayout();
    bool _createDescriptorSet();
    bool _recordCommandBuffer();
    bool _createComputePipeline();
    bool _createFence();
    void _destroy();
};

} // namespace pk
