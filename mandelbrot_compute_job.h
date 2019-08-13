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
// Example ComputeJob: render Mandelbrot into a buffer and save it to disk
//

class MandelbrotComputeJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    // factory method
    static std::unique_ptr<MandelbrotComputeJob> create( compute_t hCompute, uint32_t outputWidth, uint32_t outputHeight );

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    bool     enableGammaCorrection;
    uint32_t maxIterations;
    void     save( const std::string path );

    virtual ~MandelbrotComputeJob()
    {
        _destroy();
    }

private:
    MandelbrotComputeJob()                              = delete;
    MandelbrotComputeJob( const MandelbrotComputeJob& ) = delete;
    MandelbrotComputeJob& operator=( const MandelbrotComputeJob& ) = delete;

    MandelbrotComputeJob( compute_t hCompute, uint32_t outputWidth, uint32_t outputHeight ) :
        IComputeJob( hCompute ),
        initialized( false ),
        enableGammaCorrection( false ),
        maxIterations( 128 ),
        inputWidth( 0 ),
        inputHeight( 0 ),
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
    // NOTE: making this static assumes that all ComputeJobs of a given type are only submitted to the same ComputeInstance
    static VulkanUtils::ComputeShaderProgram shaderProgram;

    // *****************************************************************************
    // Methods and members below are shader-specific
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

typedef std::unique_ptr<MandelbrotComputeJob> MandelbrotComputeJobPtr;

} // namespace pk
