// *****************************************************************************
// Common utility methods
// *****************************************************************************

#include "vulkan_utils.h"

#include "compute_buffer_vulkan.h"
#include "utils.h"

#include <assert.h>
#include <stdint.h>


namespace pk
{

static uint32_t  _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties );
static uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength );
static result    _createShaderBuffers( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _createBuffers( VulkanContext& context, IComputeBuffer** ppBuffers, uint32_t count );
static result    _createBuffer( VulkanContext& context, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties );
static result    _createDescriptorSetLayout( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _createDescriptorSet( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _createComputeShaderProgram( VulkanContext& context, VulkanUtils::ComputeShaderProgram* pComputeShaderProgram );
static result    _createComputePipeline( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _bindBuffers( VulkanContext& context, IComputeBuffer** ppBuffers, uint32_t count, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _recordCommandBuffer( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );
static result    _createFence( VulkanContext& context, VulkanUtils::ComputeShaderInstance* pComputeShader );


// Create and bind the shader program, buffers, descriptors, layouts, and pipeline
result VulkanUtils::createComputeShader( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
    SpinLockGuard device_lock( *vulkan.pSpinlock );
    SpinLockGuard shader_lock( pShader->spinlock );

    // Create per-class resources: program binary, descriptorSetlayout, pipelineLayout, and pipeline once per class.
    if ( pShader->pProgram->shaderModule == nullptr ) {
        _createComputeShaderProgram( vulkan, pShader->pProgram );
        _createDescriptorSetLayout( vulkan, pShader );
        _createComputePipeline( vulkan, pShader );
    }

    // Crate per-instance resources
    _createShaderBuffers( vulkan, pShader );
    _recordCommandBuffer( vulkan, pShader );
    _createFence( vulkan, pShader );

    return R_OK;
}


result VulkanUtils::createBuffer( VulkanContext& vulkan, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties )
{
    if ( bufferSize == 0 )
        return R_FAIL;

    SpinLockGuard device_lock( *vulkan.pSpinlock );

    return _createBuffer( vulkan, bufferSize, pBuffer, pBufferMemory, usage, properties );
}


result VulkanUtils::recordCommandBuffer( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
    // BUG: simultaneous access to vulkan.commandPool from multi threads is not allowed
    // TODO: commandPool should be created and bound per-thread
    SpinLockGuard device_lock( *vulkan.pSpinlock );
    SpinLockGuard shader_lock( pShader->spinlock );

    return _recordCommandBuffer( vulkan, pShader );
}


result VulkanUtils::createFence( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
    SpinLockGuard device_lock( *vulkan.pSpinlock );
    SpinLockGuard shader_lock( pShader->spinlock );

    return _createFence( vulkan, pShader );
}


//
// Private Implementation
//

static uint32_t _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties )
{
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    vkGetPhysicalDeviceMemoryProperties( physicalDevice, &memoryProperties );

    for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ ) {
        if ( type & ( 1 << i ) && ( memoryProperties.memoryTypes[ i ].propertyFlags & properties ) == properties ) {
            return i;
        }
    }

    return (uint32_t)-1;
}


static uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength )
{
    FILE*   fp  = nullptr;
    errno_t err = fopen_s( &fp, shaderPath.c_str(), "rb" );
    if ( fp == nullptr || err == EINVAL ) {
        printf( "ERROR: ComputeJob: failed to load shader [%s]\n", shaderPath.c_str() );
        return nullptr;
    }

    fseek( fp, 0, SEEK_END );
    size_t filesize = ftell( fp );
    fseek( fp, 0, SEEK_SET );

    // Vulkan / SPIR-V requires shader buffer to be an array of uint32_t padded with 0
    size_t    padded = size_t( ceil( filesize / 4.0f ) * 4 );
    uint32_t* buffer = new uint32_t[ padded ];
    fread( buffer, filesize, sizeof( uint8_t ), fp );
    for ( size_t i = filesize; i < padded; i++ ) {
        buffer[ i ] = 0;
    }

    fclose( fp );

    if ( pShaderLength ) {
        *pShaderLength = (uint32_t)padded;
    }

    printf( "VulkanUtils: loaded %zd bytes of shader\n", filesize );

    return buffer;
}


result _createBuffer( VulkanContext& vulkan, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties )
{
    if ( bufferSize == 0 )
        return R_FAIL;

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size               = bufferSize;
    bufferCreateInfo.usage              = usage;
    bufferCreateInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

    CHECK_VK( vkCreateBuffer( vulkan.device, &bufferCreateInfo, nullptr, pBuffer ) );

    VkMemoryRequirements memoryRequirements = {};
    vkGetBufferMemoryRequirements( vulkan.device, *pBuffer, &memoryRequirements );

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize       = memoryRequirements.size;
    allocateInfo.memoryTypeIndex      = _findMemoryType( vulkan.physicalDevice, memoryRequirements.memoryTypeBits, properties );

    CHECK_VK( vkAllocateMemory( vulkan.device, &allocateInfo, nullptr, pBufferMemory ) );
    CHECK_VK( vkBindBufferMemory( vulkan.device, *pBuffer, *pBufferMemory, 0 ) );

    //printf( "VulkanUtils::createBuffer(): allocated %zd bytes of buffer usage 0x%x props 0x%x\n", bufferSize, usage, properties );

    return R_OK;
}


result _createComputeShaderProgram( VulkanContext& vulkan, VulkanUtils::ComputeShaderProgram* pProgram )
{
    uint32_t  shaderLength = 0;
    uint32_t* shaderBinary = nullptr;

    shaderBinary = _loadShader( pProgram->shaderPath, &shaderLength );

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = shaderBinary;
    shaderModuleCreateInfo.codeSize                 = shaderLength;
    CHECK_VK( vkCreateShaderModule( vulkan.device, &shaderModuleCreateInfo, nullptr, &( pProgram->shaderModule ) ) );
    delete[] shaderBinary;

    return R_OK;
}


result _createDescriptorSetLayout( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader )
{
    VkDescriptorSetLayoutBinding* bindings = new VkDescriptorSetLayoutBinding[ pShader->numBuffers ];

    for ( unsigned i = 0; i < pShader->numBuffers; i++ ) {
        VkDescriptorSetLayoutBinding& layout = bindings[ i ];
        ComputeBufferVulkan*          buf    = dynamic_cast<ComputeBufferVulkan*>( pShader->ppBuffers[ i ] );

        switch ( buf->type ) {
            case ComputeBufferType::COMPUTE_BUFFER_UNIFORM:
                layout.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case ComputeBufferType::COMPUTE_BUFFER_STORAGE:
                layout.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            default:
                assert( 0 );
        }

        layout.binding         = buf->binding;
        layout.descriptorCount = 1;
        layout.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount                    = pShader->numBuffers;
    createInfo.pBindings                       = bindings;

    CHECK_VK( vkCreateDescriptorSetLayout( vulkan.device, &createInfo, nullptr, &pShader->pProgram->descriptorSetLayout ) );

    delete[] bindings;

    return R_OK;
}


result _createComputePipeline( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader )
{
    VulkanUtils::ComputeShaderProgram* pProgram = pShader->pProgram;

    // Prevent race condition where ComputeJobs spawn on multiple threads, but only the first one
    // is constructing the shader
    while ( !pProgram->shaderModule ) {
    }

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module                          = pProgram->shaderModule;
    shaderStageCreateInfo.pName                           = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount             = 1;
    pipelineLayoutCreateInfo.pSetLayouts                = &( pProgram->descriptorSetLayout );
    CHECK_VK( vkCreatePipelineLayout( vulkan.device, &pipelineLayoutCreateInfo, nullptr, &( pProgram->pipelineLayout ) ) );

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = pProgram->pipelineLayout;

    CHECK_VK( vkCreateComputePipelines( vulkan.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &( pProgram->pipeline ) ) );

    printf( "VulkanUtils::createComputePipeline: [%s]\n", pProgram->shaderPath );

    return R_OK;
}


result _createShaderBuffers( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pComputeShader )
{
    _createBuffers( vulkan, pComputeShader->ppBuffers, pComputeShader->numBuffers );
    _createDescriptorSet( vulkan, pComputeShader );
    _bindBuffers( vulkan, pComputeShader->ppBuffers, pComputeShader->numBuffers, pComputeShader );

    return R_OK;
}


result _createBuffers( VulkanContext& vulkan, IComputeBuffer** ppBuffers, uint32_t count )
{
    for ( unsigned i = 0; i < count; i++ ) {
        ComputeBufferVulkan* buf = dynamic_cast<ComputeBufferVulkan*>( ppBuffers[ i ] );

        // Don't recreate a buffer that was already allocated and bound
        if ( buf->vkBuffer != VK_NULL_HANDLE )
            continue;

        VkBufferUsageFlags    usage      = 0;
        VkMemoryPropertyFlags properties = 0;

        switch ( buf->type ) {
            case ComputeBufferType::COMPUTE_BUFFER_UNIFORM:
                usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                break;
            case ComputeBufferType::COMPUTE_BUFFER_STORAGE:
                usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                break;
            default:
                assert( 0 );
        }

        switch ( buf->visibility ) {
            case ComputeBufferVisibility::COMPUTE_BUFFER_SHARED:
                properties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                break;
            case ComputeBufferVisibility::COMPUTE_BUFFER_DEVICE:
                properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                break;
            default:
                assert( 0 );
        }

        _createBuffer( vulkan, buf->size(), &buf->vkBuffer, &buf->vkBufferMemory, usage, properties );
    }

    return R_OK;
}


result _createDescriptorSet( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader )
{
    // Bind shader descriptors to buffers
    // TODO: check if descriptorPool has been exhausted

    VulkanUtils::ComputeShaderProgram& program = *( pShader->pProgram );

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool              = vulkan.descriptorPool;
    allocInfo.descriptorSetCount          = 1;
    allocInfo.pSetLayouts                 = &program.descriptorSetLayout;

    CHECK_VK( vkAllocateDescriptorSets( vulkan.device, &allocInfo, &pShader->descriptorSet ) );

    if ( pShader->descriptorSet == 0 ) {
        printf( "ERROR: VulkanUtils::createDescriptorSet(): failed to alloc descriptors (pool exhausted?): %s\n", program.shaderPath );
        return R_FAIL;
    }

    for ( unsigned i = 0; i < pShader->numBuffers; i++ ) {
        ComputeBufferVulkan* buf = dynamic_cast<ComputeBufferVulkan*>( pShader->ppBuffers[ i ] );

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer                 = buf->vkBuffer;
        bufferInfo.offset                 = 0;
        bufferInfo.range                  = buf->size();

        VkWriteDescriptorSet writeSet = {};
        writeSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSet.dstSet               = pShader->descriptorSet;
        writeSet.dstBinding           = i;
        writeSet.descriptorCount      = 1;
        writeSet.pBufferInfo          = &bufferInfo;

        switch ( buf->type ) {
            case ComputeBufferType::COMPUTE_BUFFER_UNIFORM:
                writeSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case ComputeBufferType::COMPUTE_BUFFER_STORAGE:
                writeSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            default:
                assert( 0 );
        }

        vkUpdateDescriptorSets( vulkan.device, 1, &writeSet, 0, nullptr );
    }

    //printf( "VulkanUtils::createDescriptorSet(): bound %d descriptors to set 0x%llx\n", pShader->numBuffers, (uint64_t)pShader->descriptorSet );

    return R_OK;
}


result _bindBuffers( VulkanContext& vulkan, IComputeBuffer** ppBuffers, uint32_t count, VulkanUtils::ComputeShaderInstance* pShader )
{
    bool result = true;
    for ( unsigned i = 0; i < count; i++ ) {
        ComputeBufferVulkan* buf = dynamic_cast<ComputeBufferVulkan*>( ppBuffers[ i ] );
        result &= buf->bind( pShader );
    }

    //printf( "VulkanUtils::bindBuffers(): %d buffers to descriptorSet %s:%lld\n", count, pShader->program->shaderPath, (uint64_t)pShader->descriptorSet );

    return result ? R_OK : R_FAIL;
}


result _recordCommandBuffer( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader )
{
    VulkanUtils::ComputeShaderProgram* pProgram = pShader->pProgram;

    if ( pShader->commandBuffer != VK_NULL_HANDLE ) {
        CHECK_VK( vkResetCommandBuffer( pShader->commandBuffer, 0 ) );
        vkFreeCommandBuffers( vulkan.device, vulkan.commandPool, 1, &pShader->commandBuffer );
        pShader->commandBuffer = VK_NULL_HANDLE;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool                 = vulkan.commandPool;
    allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount          = 1;
    CHECK_VK( vkAllocateCommandBuffers( vulkan.device, &allocInfo, &( pShader->commandBuffer ) ) );

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = 0;
    CHECK_VK( vkBeginCommandBuffer( pShader->commandBuffer, &beginInfo ) );

    vkCmdBindPipeline( pShader->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pProgram->pipeline );
    vkCmdBindDescriptorSets( pShader->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pProgram->pipelineLayout, 0, 1, &( pShader->descriptorSet ), 0, nullptr );
    vkCmdDispatch( pShader->commandBuffer, pProgram->workgroupWidth, pProgram->workgroupHeight, pProgram->workgroupDepth );
    CHECK_VK( vkEndCommandBuffer( pShader->commandBuffer ) );

    //printf( "VulkanUtils::recordCommandBuffer():  0x%llx [%s]\n", (uint64_t)pShader->commandBuffer, pShader->program->shaderPath );

    return R_OK;
}


result _createFence( VulkanContext& vulkan, VulkanUtils::ComputeShaderInstance* pShader )
{
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    CHECK_VK( vkCreateFence( vulkan.device, &fenceCreateInfo, nullptr, &pShader->fence ) );

    return R_OK;
}


} // namespace pk
