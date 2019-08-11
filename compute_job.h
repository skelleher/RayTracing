#pragma once

#include "assert.h"
#include "compute.h"
#include "spin_lock.h"
#include "thread_pool.h"
#include "vulkan_utils.h"

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
// . one storage buffer for input
// . one storage buffer for output
//

class IComputeJobVulkan {
public:
    // Mandatory Vulkan members; assigned by the Vulkan implementation of computeSubmitJob()
    VulkanContext vulkan;
};


class ComputeJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    static std::unique_ptr<ComputeJob> create( compute_t hCompute ); // factory method

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t outputWidth;
    uint32_t outputHeight;
    void     save( const std::string path );

protected:
    ComputeJob()                     = delete;
    ComputeJob( const ComputeJob & ) = delete;
    ComputeJob &operator=( const ComputeJob & ) = delete;

    ComputeJob( compute_t hCompute ) :
        IComputeJob( hCompute ),
        initialized( false ),
        inputWidth( 1 ),
        inputHeight( 1 ),
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

    void _destroy();

protected:
    static std::atomic<uint32_t> numInstances;

    // *****************************************************************************
    // The shader program (and descriptorSetLayout, pipeline, etc) are common
    // to all instances of this shader
    // *****************************************************************************
    // NOTE: making this static assumes that all ComputeJobs of a given type are only submitted to the same ComputeInstance
    static VulkanUtils::ComputeShaderProgram shaderProgram;

    // *****************************************************************************
    // Methods and members below are specific to a shader instance
    // *****************************************************************************
    bool initialized;
    bool destroyed;

    VulkanUtils::ComputeShaderInstance shader;

    // Shader inputs
    VkBuffer       uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    uint32_t       uniformBufferSize;

    VkBuffer       inputBuffer;
    VkDeviceMemory inputBufferMemory;
    uint32_t       inputBufferSize;

    // Shader outputs
    VkBuffer       outputBuffer;
    VkDeviceMemory outputBufferMemory;
    uint32_t       outputBufferSize;
};

} // namespace pk
