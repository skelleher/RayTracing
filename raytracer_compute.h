#pragma once

#include "assert.h"
#include "compute.h"
#include "compute_buffer_vulkan.h"
#include "compute_job_vulkan.h"
#include "spin_lock.h"
#include "thread_pool.h"
#include "vulkan_utils.h"

// Ray tracing
#include "camera.h"
#include "scene.h"

#include <string>
#include <vulkan/vulkan.h>


namespace pk
{

//
// Raytracer implemented as a Vulkan compute job
//

class RayTracerJob final : public virtual IComputeJob, public IComputeJobVulkan {
public:
    // factory method
    static std::unique_ptr<RayTracerJob> create( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight );

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job

    int renderScene( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned blockSize, bool debug, bool recursive );

    virtual ~RayTracerJob()
    {
        _destroy();
    }

    ComputeBufferVulkan uniformBuffer;
    ComputeBufferVulkan inputBuffer;
    ComputeBufferVulkan outputBuffer;

private:
    RayTracerJob()                      = delete;
    RayTracerJob( const RayTracerJob& ) = delete;
    RayTracerJob& operator=( const RayTracerJob& ) = delete;

    RayTracerJob( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight ) :
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
    // NOTE: making this static assumes that all ComputeJobs of a given type are only submitted to the same ComputeInstance
    static VulkanUtils::ComputeShaderProgram shaderProgram;

    // *****************************************************************************
    // Methods and members below are shader-specific
    // *****************************************************************************
    bool initialized;
    bool destroyed;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t outputWidth;
    uint32_t outputHeight;

    VulkanUtils::ComputeShaderInstance shader;
};

typedef std::unique_ptr<RayTracerJob> RayTracerJobPtr;

} // namespace pk
