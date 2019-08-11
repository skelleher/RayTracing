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
// Base class for Vulkan compute jobs.
//
// Assumes your compute shader has:
// . entry point named main()
// . single uniform buffer for input
// . single storage buffer for output
//

class ComputeJob : public virtual IComputeJob {
public:
    static std::unique_ptr<ComputeJob> create( compute_t hCompute ); // factory method

    // IComputeJob
    virtual void init();                           // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit();                      // update share inputs / uniforms
    virtual void submit();                         // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ); // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job
    virtual void destroy();                        // clean up resources

    // TEST:
    void     save( const std::string path );
    bool     enableGammaCorrection;
    uint32_t maxIterations;
    uint32_t outputWidth;
    uint32_t outputHeight;
    uint32_t submitCount;
    uint32_t presubmitCount;
    uint32_t postsubmitCount;

    // Inherited from the owning ComputeInstance:
    compute_t        hCompute;
    VkDevice         device;
    VkPhysicalDevice physicalDevice;
    VkDescriptorPool descriptorPool;
    VkCommandPool    commandPool;
    VkQueue          queue;

protected:
    ComputeJob( const ComputeJob& ) = delete;
    ComputeJob& operator=( const ComputeJob& ) = delete;

    ComputeJob( compute_t hCompute ) :
        initialized( false ),
        workgroupWidth( -1 ),
        workgroupHeight( -1 ),
        workgroupDepth( -1 ),
        // TEST:
        enableGammaCorrection( false ),
        maxIterations( 128 ),
        outputWidth( 3200 ),
        outputHeight( 2400 )
    {
        numInstances++;
        this->hCompute = hCompute;
        computeAcquire( hCompute );
    }

public:
    virtual ~ComputeJob()
    {
        destroy();
        handle = INVALID_COMPUTE_JOB;
        computeRelease( hCompute );
    }

protected:
    // *****************************************************************************
    // Common members and utility methods
    //
    // These generally don't need to change, unless your compute shader does
    // something unusual.
    // *****************************************************************************

    // Shared by all hComputes of this shader
    // NOTE: making these static assumes that all ComputeJobs of a given type are never submitted to a different ComputeInstance
    static std::atomic<bool>     firstInstance;
    static std::atomic<uint32_t> numInstances;
    static std::string           shaderPath;
    static uint32_t              shaderLength;
    static uint32_t*             shaderBinary;
    static VkShaderModule        computeShaderModule;
    static VkDescriptorSetLayout descriptorSetLayout;
    static VkPipeline            pipeline;
    static VkPipelineLayout      pipelineLayout;

    // One per hCompute / invokation
    VkFence fence;
    bool    initialized;

    // Caller must delete[] the returned shader buffer
    uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength );
    uint32_t  _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties );
    bool      _createBuffer( VkDevice device, VkPhysicalDevice physicalDevice, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
    bool      _recordCommandBuffer();
    bool      _createComputePipeline();
    bool      _createFence();

protected:
    // *****************************************************************************
    // Methods and members below are shader-specific
    // *****************************************************************************

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

    virtual bool _createBuffers();
    virtual bool _createDescriptorSetLayout();
    virtual bool _createDescriptorSet();
};

} // namespace pk
