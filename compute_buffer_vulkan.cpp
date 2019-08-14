#include "compute_buffer_vulkan.h"

#include "vulkan_utils.h"

namespace pk
{

ComputeBufferVulkan::ComputeBufferVulkan() :
    IComputeBuffer( 0, COMPUTE_BUFFER_TYPE_UNKNOWN, COMPUTE_BUFFER_VISIBILITY_UNKNOWN, ComputeBufferDims(), nullptr ),
    vkBuffer( VK_NULL_HANDLE ),
    vkBufferMemory( VK_NULL_HANDLE ),
    allocated( false )
{
}


ComputeBufferVulkan::~ComputeBufferVulkan()
{
    _deallocate();
}


bool ComputeBufferVulkan::init( const VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader, uint32_t binding, const ComputeBufferDims& dims, ComputeBufferType type, ComputeBufferVisibility visibility )
{
    this->vulkan     = vulkan;
    this->pShader    = pShader;
    this->binding    = binding;
    this->dims       = dims;
    this->type       = type;
    this->visibility = visibility;

    // Can't allocate or bind 0-length buffers.
    // Rather than add a lot of checks, force dims to non-zero.
    // Caller should call .resize() later when they know the actual length.
    this->dims.width       = max( dims.width, 1 );
    this->dims.height      = max( dims.height, 1 );
    this->dims.elementSize = max( dims.elementSize, 1 );

    return _allocate( this->dims );
}


bool ComputeBufferVulkan::bind( void* pShader )
{
    if ( !pShader )
        return false;

    SpinLockGuard lock( spinlock );

    this->pShader = (VulkanUtils::ComputeShaderInstance*)pShader;
    return _bind();
}


bool ComputeBufferVulkan::resize( const ComputeBufferDims& dims )
{
    SpinLockGuard lock( spinlock );

    // TODO: still not thread-safe; can't modify a descriptorSet if it
    // is in use by a command buffer on the GPU.

    printf( "ComputeBufferVulkan::resize( %zd x %zd x %zd )\n", dims.width, dims.height, dims.elementSize );

    // Make a copy, since _deallocate() zeros .dims,
    // and caller may have passed a reference to .dims into this method
    ComputeBufferDims newDims = dims;

    if ( allocated ) {
        _deallocate();
    }

    _allocate( newDims );

    // Let the shader know it must re-generate the command buffer
    sizeHasChanged = true;

    return true;
}


size_t ComputeBufferVulkan::size() const
{
    return dims.width * dims.height * dims.elementSize;
}


void ComputeBufferVulkan::map()
{
    SpinLockGuard lock( spinlock );

    if ( !allocated )
        _allocate( dims );

    CHECK_VK( vkMapMemory( vulkan.device, vkBufferMemory, 0, size(), 0, &mapped ) );
}


void ComputeBufferVulkan::unmap()
{
    SpinLockGuard lock( spinlock );

    if ( !allocated )
        _allocate( dims );

    vkUnmapMemory( vulkan.device, vkBufferMemory );
    mapped = nullptr;
}


void ComputeBufferVulkan::free()
{
    _deallocate();
}


bool ComputeBufferVulkan::_allocate( const ComputeBufferDims& dims )
{
    size_t size = dims.width * dims.height * dims.elementSize;

    if ( allocated || size == 0 ) {
        assert( 0 );
        return false;
    }

    switch ( type ) {
        case ComputeBufferType::COMPUTE_BUFFER_UNIFORM:
            vkUsage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            break;
        case ComputeBufferType::COMPUTE_BUFFER_STORAGE:
            vkUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            break;
        default:
            assert( 0 );
    }

    switch ( visibility ) {
        case ComputeBufferVisibility::COMPUTE_BUFFER_SHARED:
            vkProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            break;
        case ComputeBufferVisibility::COMPUTE_BUFFER_DEVICE:
            vkProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        default:
            assert( 0 );
    }

    if ( R_OK == VulkanUtils::createBuffer( vulkan, size, &vkBuffer, &vkBufferMemory, vkUsage, vkProperties ) ) {
        this->dims = dims;
        allocated  = true;
        _bind();
    } else {
        printf( "ERROR: ComputeBufferVulkan::_allocate( %zd x %zd x %zd )\n", dims.width, dims.height, dims.elementSize );
    }

    return allocated;
}


bool ComputeBufferVulkan::_bind()
{
    if ( pShader->descriptorSet == VK_NULL_HANDLE )
        return false;

    size_t size = dims.width * dims.height * dims.elementSize;

    // rebind the descriptorSet
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer                 = vkBuffer;
    bufferInfo.offset                 = 0;
    bufferInfo.range                  = size;

    VkWriteDescriptorSet writeSet = {};
    writeSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.dstSet               = pShader->descriptorSet;
    writeSet.dstBinding           = binding;
    writeSet.descriptorCount      = 1;
    writeSet.pBufferInfo          = &bufferInfo;

    switch ( this->type ) {
        case ComputeBufferType::COMPUTE_BUFFER_UNIFORM:
            writeSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            break;
        case ComputeBufferType::COMPUTE_BUFFER_STORAGE:
            writeSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            break;
        default:
            assert( 0 );
    }

    //printf( "update descset 0x%llx\n", (uint64_t)pShader->descriptorSet );
    vkUpdateDescriptorSets( vulkan.device, 1, &writeSet, 0, nullptr );

    return true;
}


void ComputeBufferVulkan::_deallocate()
{
    if ( !allocated )
        return;

    //printf( "_dealloc buf %lld\n", (uint64_t)vkBuffer );

    vkFreeMemory( vulkan.device, vkBufferMemory, nullptr );
    vkDestroyBuffer( vulkan.device, vkBuffer, nullptr );
    vkBuffer       = VK_NULL_HANDLE;
    vkBufferMemory = VK_NULL_HANDLE;

    dims.width = dims.height = dims.elementSize = 0;
    allocated                                   = false;
}


} // namespace pk
