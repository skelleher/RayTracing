#pragma once

#include "assert.h"
#include "compute.h"
#include "compute_job.h"
#include "spin_lock.h"
#include "thread_pool.h"
#include "vulkan_utils.h"

#include <string>
#include <vulkan/vulkan.h>


namespace pk
{

//
// Example instance of ComputeJob: render a Mandelbrot into a buffer and save it to disk
//

class MandelbrotComputeJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    static std::unique_ptr<MandelbrotComputeJob> create( compute_t hCompute ); // factory method

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    bool     enableGammaCorrection;
    uint32_t maxIterations;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t outputWidth;
    uint32_t outputHeight;
    void     save( const std::string path );

protected:
    MandelbrotComputeJob()                              = delete;
    MandelbrotComputeJob( const MandelbrotComputeJob& ) = delete;
    MandelbrotComputeJob& operator=( const MandelbrotComputeJob& ) = delete;

    MandelbrotComputeJob( compute_t hCompute ) :
        IComputeJob( hCompute ),
        initialized( false ),
        enableGammaCorrection( false ),
        maxIterations( 128 ),
        inputWidth( 1 ),
        inputHeight( 1 ),
        outputWidth( 3200 ),
        outputHeight( 2400 )
    {
        numInstances++;
    }

public:
    virtual ~MandelbrotComputeJob()
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
    // Methods and members below are shader-specific
    // *****************************************************************************
    bool initialized;
    bool destroyed;

    VulkanUtils::ComputeShaderInstance shader;

    // Shader inputs
    VkBuffer       uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    uint32_t       uniformBufferSize;

    // Shader outputs
    VkBuffer       outputBuffer;
    VkDeviceMemory outputBufferMemory;
    uint32_t       outputBufferSize;

    VkBuffer       inputBuffer;
    VkDeviceMemory inputBufferMemory;
    uint32_t       inputBufferSize;
};

} // namespace pk
