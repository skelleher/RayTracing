#include "compute_job.h"

#include "perf_timer.h"
#include "utils.h"
#include "vulkan_utils.h"

#include <assert.h>

namespace pk
{

//
// Example Vulkan compute job.
// You should copy and modify it to do something interesting.
//
// Assumes your compute shader has:
// . entry point named main()
// . single uniform buffer for input
// . single storage buffer for output
//

// TODO: ComputeInstance should query the GPU for best workgroupSize and pass it to create()
static const uint32_t WORK_GROUP_SIZE = 32;

std::atomic<uint32_t> ComputeJob::numInstances = 0;

// Instances of a compute job share the same shader binary and pipeline.
VulkanUtils::ComputeShaderProgram ComputeJob::shaderProgram = {
    "shaders\\test_vulkan.spv"
};

// TODO: these never change (for a given pipeline) so should be
// set at pipeline creation stage via Push Constants instead of passed as uniforms.
struct UniformBufferObject {
    alignas( 4 ) uint32_t inputWidth;
    alignas( 4 ) uint32_t inputHeight;
    alignas( 4 ) uint32_t outputWidth;
    alignas( 4 ) uint32_t outputHeight;
};


// factory method
std::unique_ptr<ComputeJob> ComputeJob::create( compute_t hCompute )
{
    std::unique_ptr<ComputeJob> ptr( new ComputeJob( hCompute ) );
    return ptr;
}


// IComputeJob
void ComputeJob::init()
{
    if ( !initialized ) {
        shaderProgram.workgroupSize   = WORK_GROUP_SIZE;
        shaderProgram.workgroupWidth  = (uint32_t)ceil( outputWidth / (float)shaderProgram.workgroupSize );
        shaderProgram.workgroupHeight = (uint32_t)ceil( outputHeight / (float)shaderProgram.workgroupSize );
        shaderProgram.workgroupDepth  = 1;

        uniformBufferSize = sizeof( UniformBufferObject );
        inputBufferSize   = inputWidth * inputHeight * sizeof( float );
        outputBufferSize  = outputWidth * outputHeight * 4 * sizeof( float );

        VulkanUtils::ShaderBufferInfo buffers[] = {
            { 0, "ubo", VulkanUtils::BUFFER_COMPUTE_SHADER_UNIFORM, sizeof( UniformBufferObject ), &uniformBuffer, &uniformBufferMemory, VulkanUtils::BUFFER_SHARED },
            { 1, "input", VulkanUtils::BUFFER_COMPUTE_SHADER_STORAGE, inputBufferSize, &inputBuffer, &inputBufferMemory, VulkanUtils::BUFFER_SHARED },
            { 2, "output", VulkanUtils::BUFFER_COMPUTE_SHADER_STORAGE, outputBufferSize, &outputBuffer, &outputBufferMemory, VulkanUtils::BUFFER_SHARED },
        };

        shader.program    = &shaderProgram;
        shader.pBuffers   = buffers;
        shader.numBuffers = ARRAY_SIZE( buffers );

        VulkanUtils::createComputeShader( vulkan, &shader );
        initialized = true;
    }
}


void ComputeJob::presubmit()
{
    //printf( "ComputeJob[%d:%d]::presubmit()\n", hCompute, handle );

    struct UniformBufferObject ubo;
    ubo.inputWidth   = inputWidth;
    ubo.inputHeight  = inputHeight;
    ubo.outputWidth  = outputWidth;
    ubo.outputHeight = outputHeight;

    void* pUniform;
    vkMapMemory( vulkan.device, uniformBufferMemory, 0, sizeof( ubo ), 0, &pUniform );
    memcpy( pUniform, &ubo, sizeof( ubo ) );
    vkUnmapMemory( vulkan.device, uniformBufferMemory );

    void* pInputBuffer;
    vkMapMemory( vulkan.device, inputBufferMemory, 0, sizeof( ubo ), 0, &pInputBuffer );
    //
    // TODO: pass your input to compute shader here
    //
    vkUnmapMemory( vulkan.device, inputBufferMemory );
}


void ComputeJob::submit()
{
    if ( !shader.commandBuffer ) {
        return;
    }

    //printf( "ComputeJob[%d:%d]::submit()\n", hCompute, handle );

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &shader.commandBuffer;

    CHECK_VK( vkResetFences( vulkan.device, 1, &shader.fence ) );
    CHECK_VK( vkQueueSubmit( vulkan.queue, 1, &submitInfo, shader.fence ) );
}


void ComputeJob::postsubmit( uint32_t timeoutMS )
{
    //printf( "ComputeJob[%d:%d]::postsubmit()\n", hCompute, handle );

    uint64_t timeoutNS = timeoutMS * 1000000;
    VkResult rval      = vkWaitForFences( vulkan.device, 1, &shader.fence, VK_TRUE, timeoutNS );

    if ( rval == VK_TIMEOUT ) {
        printf( "ERROR: ComputeJob[%d:%d]: timeout (%d ms)\n", hCompute, handle, timeoutMS );
        return;
    }

    if ( rval != VK_SUCCESS ) {
        printf( "ERROR: ComputeJob[%d:%d]: error %d\n", hCompute, handle, rval );
        return;
    }
}


void ComputeJob::save( const std::string outputPath )
{
    printf( "Saving to %s\n", outputPath.c_str() );

    // TEST: save output of mandelbrot
    void* mappedMemory = nullptr;
    vkMapMemory( vulkan.device, outputBufferMemory, 0, outputBufferSize, 0, &mappedMemory );

    struct Pixel {
        float r;
        float g;
        float b;
        float a;
    };

    Pixel* pixels = (Pixel*)mappedMemory;

    // Save image
    FILE*   file = nullptr;
    errno_t err  = fopen_s( &file, outputPath.c_str(), "w" );
    if ( !file || err != 0 ) {
        printf( "Error: failed to open [%s] for writing errno %d.\n", outputPath.c_str(), err );
        return;
    }

    fprintf( file, "P3\n" );
    fprintf( file, "%d %d\n", outputWidth, outputHeight );
    fprintf( file, "255\n" );

    for ( uint32_t y = 0; y < outputHeight; y++ ) {
        for ( uint32_t x = 0; x < outputWidth; x++ ) {
            Pixel&  rgb = pixels[ y * outputWidth + x ];
            uint8_t _r  = ( uint8_t )( rgb.r * 255 );
            uint8_t _g  = ( uint8_t )( rgb.g * 255 );
            uint8_t _b  = ( uint8_t )( rgb.b * 255 );

            fprintf( file, "%d %d %d\n", _r, _g, _b );
        }
    }

    fflush( file );
    fclose( file );

    vkUnmapMemory( vulkan.device, outputBufferMemory );

    printf( "done\n" );
}


//
// Private implementation
//
/*************
bool ComputeJob::_createComputePipeline( VulkanContext& context )
{
    // Prevent race condition where ComputeJobs spawn on multiple threads, but only the first one
    // is constructing the shader
    while ( !shaderProgram.shaderModule ) {
    }

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module                          = shaderProgram.shaderModule;
    shaderStageCreateInfo.pName                           = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount             = 1;
    pipelineLayoutCreateInfo.pSetLayouts                = &shaderProgram.descriptorSetLayout;
    CHECK_VK( vkCreatePipelineLayout( vulkan.device, &pipelineLayoutCreateInfo, nullptr, &shaderProgram.pipelineLayout ) );

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = shaderProgram.pipelineLayout;

    CHECK_VK( vkCreateComputePipelines( vulkan.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &shaderProgram.pipeline ) );

    printf( "ComputeJob[%d:%d]: created shader pipeline for [%s]\n", hCompute, handle, shaderProgram.shaderPath );

    return true;
}


bool ComputeJob::_recordCommandBuffer( VulkanContext& vulkan )
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool                 = vulkan.commandPool;
    allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount          = 1;
    CHECK_VK( vkAllocateCommandBuffers( vulkan.device, &allocInfo, &shader.commandBuffer ) );

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = 0;
    CHECK_VK( vkBeginCommandBuffer( shader.commandBuffer, &beginInfo ) );

    vkCmdBindPipeline( shader.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shaderProgram.pipeline );
    vkCmdBindDescriptorSets( shader.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shaderProgram.pipelineLayout, 0, 1, &shader.descriptorSet, 0, nullptr );
    vkCmdDispatch( shader.commandBuffer, shaderProgram.workgroupWidth, shaderProgram.workgroupHeight, shaderProgram.workgroupDepth );
    CHECK_VK( vkEndCommandBuffer( shader.commandBuffer ) );

    //printf( "ComputeJob[%d:%d]: recorded command buffer, workgroup[%d x %d x %d]\n", hCompute, handle, workgroupWidth, workgroupHeight, workgroupDepth );

    return true;
}


bool ComputeJob::_createFence( VulkanContext& vulkan )
{
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    CHECK_VK( vkCreateFence( vulkan.device, &fenceCreateInfo, nullptr, &shader.fence ) );

    return true;
}
**********************/

// *****************************************************************************
// Methods and members below are shader-specific and should be overridden by
// subclasses
// *****************************************************************************

/************************************
bool ComputeJob::_createBuffers( VulkanContext& vulkan )
{
    uniformBufferSize = sizeof( UniformBufferObject );
    VulkanUtils::createBuffer( vulkan, uniformBufferSize, &uniformBuffer, &uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    inputBufferSize = inputWidth * inputHeight * sizeof( float );
    VulkanUtils::createBuffer( vulkan, inputBufferSize, &inputBuffer, &inputBufferMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    outputBufferSize = outputWidth * outputHeight * 4 * sizeof( float );
    VulkanUtils::createBuffer( vulkan, outputBufferSize, &outputBuffer, &outputBufferMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    return true;
}


bool ComputeJob::_createDescriptorSetLayout( VulkanContext& vulkan )
{
    // Define the layout of shader resources
    // TODO: generate these dynamically when shader calls _createBuffer()
    // TODO: SPIR-V allows reflection, so we could bind to shader resources by name
    // instead of hard-coding the binding numbers

    VkDescriptorSetLayoutBinding uniformLayoutBinding = {};
    uniformLayoutBinding.binding                      = 0;
    uniformLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformLayoutBinding.descriptorCount              = 1;
    uniformLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding inputLayoutBinding = {};
    inputLayoutBinding.binding                      = 1;
    inputLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    inputLayoutBinding.descriptorCount              = 1;
    inputLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding outputLayoutBinding = {};
    outputLayoutBinding.binding                      = 2;
    outputLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    outputLayoutBinding.descriptorCount              = 1;
    outputLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {
        uniformLayoutBinding,
        inputLayoutBinding,
        outputLayoutBinding
    };

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount                    = ARRAY_SIZE( bindings );
    createInfo.pBindings                       = bindings;

    CHECK_VK( vkCreateDescriptorSetLayout( vulkan.device, &createInfo, nullptr, &shaderProgram.descriptorSetLayout ) );
    //printf( "ComputeJob[%d:%d]: defined %d descriptors\n", hCompute, handle, createInfo.bindingCount );

    return true;
}


bool ComputeJob::_createDescriptorSet( VulkanContext& vulkan )
{
    // Bind shader descriptors to buffers

    // TODO: check if descriptorPool has been exhausted

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool              = vulkan.descriptorPool;
    allocInfo.descriptorSetCount          = 1;
    allocInfo.pSetLayouts                 = &shaderProgram.descriptorSetLayout;

    CHECK_VK( vkAllocateDescriptorSets( vulkan.device, &allocInfo, &shader.descriptorSet ) );
    //printf( "ComputeJob[%d:%d]: created %d descriptor sets\n", hCompute, handle, allocInfo.descriptorSetCount );

    if ( shader.descriptorSet == 0 ) {
        printf( "ERROR: ComputeJob[%d:%d] failed to alloc descriptors (pool exhausted?)\n", hCompute, handle );
        return false;
    }

    VkDescriptorBufferInfo descriptorUniformBufferInfo = {};
    descriptorUniformBufferInfo.buffer                 = uniformBuffer;
    descriptorUniformBufferInfo.offset                 = 0;
    descriptorUniformBufferInfo.range                  = uniformBufferSize;

    VkDescriptorBufferInfo descriptorInputBufferInfo = {};
    descriptorInputBufferInfo.buffer                 = inputBuffer;
    descriptorInputBufferInfo.offset                 = 0;
    descriptorInputBufferInfo.range                  = inputBufferSize;

    VkDescriptorBufferInfo descriptorOutputBufferInfo = {};
    descriptorOutputBufferInfo.buffer                 = outputBuffer;
    descriptorOutputBufferInfo.offset                 = 0;
    descriptorOutputBufferInfo.range                  = outputBufferSize;

    VkWriteDescriptorSet writeUniformSet = {};
    writeUniformSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeUniformSet.dstSet               = shader.descriptorSet;
    writeUniformSet.dstBinding           = 0;
    writeUniformSet.descriptorCount      = 1;
    writeUniformSet.descriptorType       = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeUniformSet.pBufferInfo          = &descriptorUniformBufferInfo;
    vkUpdateDescriptorSets( vulkan.device, 1, &writeUniformSet, 0, nullptr );

    VkWriteDescriptorSet writeInputSet = {};
    writeInputSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInputSet.dstSet               = shader.descriptorSet;
    writeInputSet.dstBinding           = 1;
    writeInputSet.descriptorCount      = 1;
    writeInputSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeInputSet.pBufferInfo          = &descriptorInputBufferInfo;
    vkUpdateDescriptorSets( vulkan.device, 1, &writeInputSet, 0, nullptr );

    VkWriteDescriptorSet writeOutputSet = {};
    writeOutputSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeOutputSet.dstSet               = shader.descriptorSet;
    writeOutputSet.dstBinding           = 2;
    writeOutputSet.descriptorCount      = 1;
    writeOutputSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeOutputSet.pBufferInfo          = &descriptorOutputBufferInfo;
    vkUpdateDescriptorSets( vulkan.device, 1, &writeOutputSet, 0, nullptr );

    //unsigned int numDescriptors = 2;
    //printf( "ComputeJob[%d:%d]: bound %d descriptors\n", hCompute, handle, numDescriptors );

    return true;
}
********************************************/

void ComputeJob::_destroy()
{
    //printf( "ComputeJob[%d:%d]::destroy()\n", hCompute, handle );

    if ( destroyed )
        return;

    numInstances--;

    SpinLockGuard lock( spinLock );

    CHECK_VK( vkResetCommandBuffer( shader.commandBuffer, 0 ) );
    vkFreeCommandBuffers( vulkan.device, vulkan.commandPool, 1, &shader.commandBuffer );
    CHECK_VK( vkFreeDescriptorSets( vulkan.device, vulkan.descriptorPool, 1, &shader.descriptorSet ) );
    vkFreeMemory( vulkan.device, uniformBufferMemory, nullptr );
    vkDestroyBuffer( vulkan.device, uniformBuffer, nullptr );
    vkFreeMemory( vulkan.device, inputBufferMemory, nullptr );
    vkDestroyBuffer( vulkan.device, inputBuffer, nullptr );
    vkFreeMemory( vulkan.device, outputBufferMemory, nullptr );
    vkDestroyBuffer( vulkan.device, outputBuffer, nullptr );
    vkDestroyFence( vulkan.device, shader.fence, nullptr );

    // Free the static resources shared by all instances
    if ( numInstances == 0 && shaderProgram.pipeline ) {
        printf( "ComputeJob[%d:%d]::destroy()\n", hCompute, handle );
        vkDestroyShaderModule( vulkan.device, shaderProgram.shaderModule, nullptr );
        vkDestroyDescriptorSetLayout( vulkan.device, shaderProgram.descriptorSetLayout, nullptr );
        vkDestroyPipelineLayout( vulkan.device, shaderProgram.pipelineLayout, nullptr );
        vkDestroyPipeline( vulkan.device, shaderProgram.pipeline, nullptr );

        shaderProgram.pipeline = nullptr;
    }

    destroyed = true;
}


} // namespace pk
