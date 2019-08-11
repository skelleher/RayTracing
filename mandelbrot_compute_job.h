#pragma once

#include "assert.h"
#include "compute.h"
#include "compute_job.h"
#include "spin_lock.h"
#include "thread_pool.h"

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
    virtual void destroy();                        // clean up resources

    bool     enableGammaCorrection;
    uint32_t maxIterations;
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
        workgroupWidth( -1 ),
        workgroupHeight( -1 ),
        workgroupDepth( -1 ),
        enableGammaCorrection( false ),
        maxIterations( 128 ),
        outputWidth( 3200 ),
        outputHeight( 2400 )
    {
        numInstances++;
    }

public:
    virtual ~MandelbrotComputeJob()
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

    // Shared by all instances of this shader
    // NOTE: making these static assumes that all MandelbrotComputeJobs are never submitted to a different ComputeInstance
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
};

} // namespace pk
