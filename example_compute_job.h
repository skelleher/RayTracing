#pragma once

#include "assert.h"
#include "compute.h"
#include "compute_buffer_vulkan.h"
#include "compute_job_vulkan.h"
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


class ExampleComputeJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    // factory method
    static std::unique_ptr<ExampleComputeJob> create( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight );

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    void save( const std::string path );

    virtual ~ExampleComputeJob()
    {
        _destroy();
    }

private:
    ExampleComputeJob()                            = delete;
    ExampleComputeJob( const ExampleComputeJob & ) = delete;
    ExampleComputeJob &operator=( const ExampleComputeJob & ) = delete;

    ExampleComputeJob( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight ) :
        IComputeJob( hCompute ),
        initialized( false ),
        inputWidth( inputWidth ),
        inputHeight( inputHeight ),
        outputWidth( outputWidth ),
        outputHeight( outputHeight )
    {
        numInstances++;
    }

    void _destroy();

private:
    static std::atomic<uint32_t> numInstances;

    // *****************************************************************************
    // The shader program (and descriptorSetLayout, pipeline, etc) are common
    // to all instances of this shader
    // *****************************************************************************
    // NOTE: making this static assumes that all ExampleComputeJobs of a given type are only submitted to the same ComputeInstance
    static VulkanUtils::ComputeShaderProgram shaderProgram;

    // *****************************************************************************
    // Methods and members below are specific to a shader instance
    // *****************************************************************************
    bool     initialized;
    bool     destroyed;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t outputWidth;
    uint32_t outputHeight;

    VulkanUtils::ComputeShaderInstance shader;
    ComputeBufferVulkan                uniformBuffer;
    ComputeBufferVulkan                inputBuffer;
    ComputeBufferVulkan                outputBuffer;
};

typedef std::unique_ptr<ExampleComputeJob> ExampleComputeJobPtr;

} // namespace pk
