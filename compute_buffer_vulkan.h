#pragma once

#include "compute_buffer.h"
#include "spin_lock.h"
#include "utils.h"
#include "vulkan_utils.h"

#include <assert.h>


namespace pk
{

class ComputeShaderInstance;

class ComputeBufferVulkan final : virtual public IComputeBuffer {
public:
    ComputeBufferVulkan();
    virtual ~ComputeBufferVulkan();

    bool init( const VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader, uint32_t binding, const ComputeBufferDims& dims, ComputeBufferType type, ComputeBufferVisibility visibility );

    // NOTE: never call bind() or resize() while the buffer is in use by the GPU
    virtual bool   bind( void* shader );
    virtual bool   resize( const ComputeBufferDims& dims );
    virtual size_t size() const;
    virtual void   map();
    virtual void   unmap();
    virtual void   free();

    SpinLock                            spinlock;
    VulkanContext                       vulkan;
    VkBuffer                            vkBuffer;
    VkDeviceMemory                      vkBufferMemory;
    VulkanUtils::ComputeShaderInstance* pShader;

protected:
    ComputeBufferVulkan( const ComputeBufferVulkan& rhs ) = delete;
    ComputeBufferVulkan& operator=( const ComputeBufferVulkan& ) = delete;

    bool _bind();
    bool _allocate( const ComputeBufferDims& dims );
    void _deallocate();

    //SpinLock spinlock;
    bool     allocated;
    uint32_t vkUsage;
    uint32_t vkProperties;
};

typedef std::shared_ptr<ComputeBufferVulkan> ComputeBufferVulkanPtr;

} // namespace pk
