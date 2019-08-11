// *****************************************************************************
// Common utility methods
// *****************************************************************************

#include "vulkan_utils.h"

#include "utils.h"

#include <assert.h>
#include <stdint.h>


namespace pk
{

static uint32_t  _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties );
static uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength );


// Create and bind the shader program, buffers, descriptors, layouts, and pipeline
result VulkanUtils::createComputeShader( VulkanContext& vulkan, ComputeShaderInstance* pComputeShader )
{
    // Create per-class resources: program binary, descriptorSetlayout, pipelineLayout, and pipeline once per class.
    if ( pComputeShader->program->shaderModule == nullptr ) {
        createComputeShaderProgram( vulkan, pComputeShader->program );
        createDescriptorSetLayout( vulkan, pComputeShader );
        createComputePipeline( vulkan, pComputeShader->program );
    }

    // Crate per-instance resources
    createShaderBuffers( vulkan, pComputeShader );
    recordCommandBuffer( vulkan, pComputeShader );
    createFence( vulkan, pComputeShader );

    return R_OK;
}



result VulkanUtils::createComputeShaderProgram( VulkanContext& vulkan, ComputeShaderProgram* pProgram )
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


result VulkanUtils::createDescriptorSetLayout( VulkanContext& vulkan, ComputeShaderInstance* pComputeShader )
{
    VkDescriptorSetLayoutBinding* bindings = new VkDescriptorSetLayoutBinding[ pComputeShader->numBuffers ];

    for ( unsigned i = 0; i < pComputeShader->numBuffers; i++ ) {
        ShaderBufferInfo&             buf    = pComputeShader->pBuffers[ i ];
        VkDescriptorSetLayoutBinding& layout = bindings[ i ];

        switch ( buf.type ) {
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_UNIFORM:
                layout.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_STORAGE:
                layout.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            default:
                assert( 0 );
        }

        layout.binding         = buf.binding;
        layout.descriptorCount = 1;
        layout.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount                    = pComputeShader->numBuffers;
    createInfo.pBindings                       = bindings;

    CHECK_VK( vkCreateDescriptorSetLayout( vulkan.device, &createInfo, nullptr, &pComputeShader->program->descriptorSetLayout ) );

    delete[] bindings;

    return R_OK;
}


result VulkanUtils::createComputePipeline( VulkanContext& vulkan, ComputeShaderProgram* pProgram )
{
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
    pipelineLayoutCreateInfo.pSetLayouts                = &(pProgram->descriptorSetLayout);
    CHECK_VK( vkCreatePipelineLayout( vulkan.device, &pipelineLayoutCreateInfo, nullptr, &(pProgram->pipelineLayout) ) );

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = pProgram->pipelineLayout;

    CHECK_VK( vkCreateComputePipelines( vulkan.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &(pProgram->pipeline) ) );

    printf( "VulkanUtils::createComputePipeline: [%s]\n", pProgram->shaderPath );

    return R_OK;
}


result VulkanUtils::createShaderBuffers( VulkanContext& vulkan, ComputeShaderInstance* pComputeShader )
{
    createBuffers( vulkan, pComputeShader->pBuffers, pComputeShader->numBuffers );
    createDescriptorSet( vulkan, pComputeShader );

    return R_OK;
}


result VulkanUtils::createBuffers( VulkanContext& vulkan, ShaderBufferInfo* pBuffers, uint32_t count )
{
    for ( unsigned i = 0; i < count; i++ ) {
        ShaderBufferInfo& buf = pBuffers[ i ];

        VkBufferUsageFlags    usage      = 0;
        VkMemoryPropertyFlags properties = 0;

        switch ( buf.type ) {
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_UNIFORM:
                usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                break;
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_STORAGE:
                usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                break;
            default:
                assert( 0 );
        }

        switch ( buf.visibility ) {
            case ShaderBufferVisibility::BUFFER_SHARED:
                properties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                break;
            case ShaderBufferVisibility::BUFFER_DEVICE:
                properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                break;
            default:
                assert( 0 );
        }

        createBuffer( vulkan, buf.size, buf.pBuffer, buf.pBufferMemory, usage, properties );
    }

    return R_OK;
}


result VulkanUtils::createBuffer( VulkanContext& vulkan, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties )
{
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

    //printf( "ComputeJob[%d:%d]: allocated %zd bytes of buffer usage 0x%x props 0x%x\n", hCompute, handle, bufferSize, usage, properties );

    return R_OK;
}


result VulkanUtils::createDescriptorSet( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
    // Bind shader descriptors to buffers
    // TODO: check if descriptorPool has been exhausted

    ComputeShaderProgram& program = *( pShader->program );

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool              = vulkan.descriptorPool;
    allocInfo.descriptorSetCount          = 1;
    allocInfo.pSetLayouts                 = &program.descriptorSetLayout;

    CHECK_VK( vkAllocateDescriptorSets( vulkan.device, &allocInfo, &pShader->descriptorSet ) );
    //printf( "MandelbrotComputeJob[%d:%d]: created %d descriptor sets\n", hCompute, handle, allocInfo.descriptorSetCount );

    if ( pShader->descriptorSet == 0 ) {
        printf( "ERROR: VulkanUtils::createDescriptorSet(): failed to alloc descriptors (pool exhausted?): %s\n", program.shaderPath );
        return R_FAIL;
    }

    for ( unsigned i = 0; i < pShader->numBuffers; i++ ) {
        ShaderBufferInfo& buf = pShader->pBuffers[ i ];

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer                 = *buf.pBuffer;
        bufferInfo.offset                 = 0;
        bufferInfo.range                  = buf.size;

        VkWriteDescriptorSet writeSet = {};
        writeSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeSet.dstSet               = pShader->descriptorSet;
        writeSet.dstBinding           = i;
        writeSet.descriptorCount      = 1;
        writeSet.pBufferInfo          = &bufferInfo;

        switch ( buf.type ) {
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_UNIFORM:
                writeSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case ShaderBufferType::BUFFER_COMPUTE_SHADER_STORAGE:
                writeSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            default:
                assert( 0 );
        }

        vkUpdateDescriptorSets( vulkan.device, 1, &writeSet, 0, nullptr );
    }

    //printf( "VulkanUtils::createDescriptorSet(): bound %d descriptors\n", pShader->numBuffers );

    return R_OK;
}


result VulkanUtils::recordCommandBuffer( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
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

    vkCmdBindPipeline( pShader->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pShader->program->pipeline );
    vkCmdBindDescriptorSets( pShader->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pShader->program->pipelineLayout, 0, 1, &( pShader->descriptorSet ), 0, nullptr );
    vkCmdDispatch( pShader->commandBuffer, pShader->program->workgroupWidth, pShader->program->workgroupHeight, pShader->program->workgroupDepth );
    CHECK_VK( vkEndCommandBuffer( pShader->commandBuffer ) );

    //printf( "ComputeJob[%d:%d]: recorded command buffer, workgroup[%d x %d x %d]\n", hCompute, handle, workgroupWidth, workgroupHeight, workgroupDepth );

    return R_OK;
}


result VulkanUtils::createFence( VulkanContext& vulkan, ComputeShaderInstance* pShader )
{
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    CHECK_VK( vkCreateFence( vulkan.device, &fenceCreateInfo, nullptr, &pShader->fence ) );

    return R_OK;
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

    printf( "ComputeJob: loaded %zd bytes of shader (padded to %zd)\n", filesize, padded );

    return buffer;
}


} // namespace pk
