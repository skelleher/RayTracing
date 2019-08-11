#include "mandelbrot_compute_job.h"

#include "perf_timer.h"
#include "utils.h"
#include "vulkan_utils.h"

#include <assert.h>

namespace pk
{

//
// This base class implements a basic Vulkan compute shader
// It assumes one input (a uniform buffer)
// and one output (a storage buffer)
//
// Subclasses should override it to do something useful
//

// TODO: ComputeInstance should query the GPU for best workgroupSize and pass it to create()
static const uint32_t WORK_GROUP_SIZE = 32;

// Instances of a compute job can share the same shader binary.
// They might be able to share the same pipeline / command buffer,
// but that means patching the I/O descriptors before enqueing to Vulkan.
std::atomic<bool>     MandelbrotComputeJob::firstInstance = true;
std::atomic<uint32_t> MandelbrotComputeJob::numInstances  = 0;
//uint32_t              MandelbrotComputeJob::shaderLength        = 0;
//uint32_t*             MandelbrotComputeJob::shaderBinary        = nullptr;
std::string           MandelbrotComputeJob::shaderPath          = "shaders\\mandelbrot.spv";
VkShaderModule        MandelbrotComputeJob::computeShaderModule = nullptr;
VkDescriptorSetLayout MandelbrotComputeJob::descriptorSetLayout;
VkPipeline            MandelbrotComputeJob::pipeline;
VkPipelineLayout      MandelbrotComputeJob::pipelineLayout;

// TODO: these never change (for a given pipeline) so should be
// set at pipeline creation stage via Push Constants instead of passed as uniforms.
struct UniformBufferObject {
    alignas( 4 ) uint32_t outputWidth;
    alignas( 4 ) uint32_t outputHeight;
    alignas( 4 ) uint32_t maxIterations;
    alignas( 4 ) bool applyGammaCorrection;
};

// factory method
std::unique_ptr<MandelbrotComputeJob> MandelbrotComputeJob::create( compute_t hCompute )
{
    std::unique_ptr<MandelbrotComputeJob> ptr( new MandelbrotComputeJob( hCompute ) );
    return ptr;
}


// IMandelbrotComputeJob
void MandelbrotComputeJob::init()
{
    // Create static resources shared by all shaders of this type
    if ( firstInstance ) {
        firstInstance = false;

        printf( "MandelbrotComputeJob[%d:%d]::create()\n", hCompute, handle );
        //shaderBinary = VulkanUtils::loadShader( shaderPath, &shaderLength );

        //VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        //shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        //shaderModuleCreateInfo.pCode                    = shaderBinary;
        //shaderModuleCreateInfo.codeSize                 = shaderLength;
        //CHECK_VK( vkCreateShaderModule( device, &shaderModuleCreateInfo, nullptr, &computeShaderModule ) );
        //delete[] shaderBinary;

        VulkanUtils::createComputeShader( device, shaderPath, &computeShaderModule );

        _createDescriptorSetLayout();
        _createComputePipeline();
    }

    workgroupSize   = WORK_GROUP_SIZE;
    workgroupWidth  = (uint32_t)ceil( outputWidth / (float)workgroupSize );
    workgroupHeight = (uint32_t)ceil( outputHeight / (float)workgroupSize );
    workgroupDepth  = 1;

    if ( !initialized ) {
        _createBuffers();
        _createDescriptorSet();
        _recordCommandBuffer();
        _createFence();

        initialized = true;
    }
}


void MandelbrotComputeJob::presubmit()
{
    //printf( "MandelbrotComputeJob[%d:%d]::presubmit()\n", hCompute, handle );

    struct UniformBufferObject ubo;
    ubo.outputWidth          = outputWidth;
    ubo.outputHeight         = outputHeight;
    ubo.maxIterations        = maxIterations;
    ubo.applyGammaCorrection = enableGammaCorrection;

    void* data;
    vkMapMemory( device, uniformBufferMemory, 0, sizeof( ubo ), 0, &data );
    memcpy( data, &ubo, sizeof( ubo ) );
    vkUnmapMemory( device, uniformBufferMemory );
}


void MandelbrotComputeJob::submit()
{
    if ( !commandBuffer ) {
        return;
    }

    //printf( "MandelbrotComputeJob[%d:%d]::submit()\n", hCompute, handle );

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffer;

    CHECK_VK( vkResetFences( device, 1, &fence ) );
    CHECK_VK( vkQueueSubmit( queue, 1, &submitInfo, fence ) );
}


void MandelbrotComputeJob::postsubmit( uint32_t timeoutMS )
{
    //printf( "MandelbrotComputeJob[%d:%d]::postsubmit()\n", hCompute, handle );

    uint64_t timeoutNS = timeoutMS * 1000000;
    VkResult rval      = vkWaitForFences( device, 1, &fence, VK_TRUE, timeoutNS );

    if ( rval == VK_TIMEOUT ) {
        printf( "ERROR: MandelbrotComputeJob[%d:%d]: timeout (%d ms)\n", hCompute, handle, timeoutMS );
        return;
    }

    if ( rval != VK_SUCCESS ) {
        printf( "ERROR: MandelbrotComputeJob[%d:%d]: error %d\n", hCompute, handle, rval );
        return;
    }
}


void MandelbrotComputeJob::save( const std::string outputPath )
{
    printf( "Saving to %s\n", outputPath.c_str() );

    // TEST: save output of mandelbrot
    void* mappedMemory = nullptr;
    vkMapMemory( device, outputBufferMemory, 0, outputBufferSize, 0, &mappedMemory );

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

    vkUnmapMemory( device, outputBufferMemory );

    printf( "done\n" );
}


//
// Private Implementation
//

bool MandelbrotComputeJob::_createComputePipeline()
{
    // Prevent race condition where MandelbrotComputeJobs spawn on multiple threads, but only the first one
    // is constructing the shader
    while ( !computeShaderModule ) {
    }

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module                          = computeShaderModule;
    shaderStageCreateInfo.pName                           = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount             = 1;
    pipelineLayoutCreateInfo.pSetLayouts                = &descriptorSetLayout;
    CHECK_VK( vkCreatePipelineLayout( device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout ) );

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = pipelineLayout;

    CHECK_VK( vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline ) );

    printf( "MandelbrotComputeJob[%d:%d]: created shader pipeline for [%s]\n", hCompute, handle, shaderPath.c_str() );

    return true;
}


bool MandelbrotComputeJob::_recordCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool                 = commandPool;
    allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount          = 1;
    CHECK_VK( vkAllocateCommandBuffers( device, &allocInfo, &commandBuffer ) );

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = 0; // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK( vkBeginCommandBuffer( commandBuffer, &beginInfo ) );

    vkCmdBindPipeline( commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline );
    vkCmdBindDescriptorSets( commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr );
    vkCmdDispatch( commandBuffer, workgroupWidth, workgroupHeight, workgroupDepth );
    CHECK_VK( vkEndCommandBuffer( commandBuffer ) );

    //printf( "MandelbrotComputeJob[%d:%d]: recorded command buffer, workgroup[%d x %d x %d]\n", hCompute, handle, workgroupWidth, workgroupHeight, workgroupDepth );

    return true;
}


bool MandelbrotComputeJob::_createFence()
{
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    CHECK_VK( vkCreateFence( device, &fenceCreateInfo, nullptr, &fence ) );

    return true;
}


// *****************************************************************************
// Methods and members below are shader-specific and should be overridden by
// subclasses
// *****************************************************************************

bool MandelbrotComputeJob::_createBuffers()
{
    outputBufferSize = outputWidth * outputHeight * 4 * sizeof( float );
    VulkanUtils::createBuffer( device, physicalDevice, outputBufferSize, &outputBuffer, &outputBufferMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    uniformBufferSize = sizeof( UniformBufferObject );
    VulkanUtils::createBuffer( device, physicalDevice, uniformBufferSize, &uniformBuffer, &uniformBufferMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    return true;
}


bool MandelbrotComputeJob::_createDescriptorSetLayout()
{
    // Define the layout of shader resources
    // Can we assume all compute shaders will have one uniform input buffer and one storage output buffer?

    VkDescriptorSetLayoutBinding uniformLayoutBinding = {};
    uniformLayoutBinding.binding                      = 0;
    uniformLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformLayoutBinding.descriptorCount              = 1;
    uniformLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding storageLayoutBinding = {};
    storageLayoutBinding.binding                      = 1;
    storageLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageLayoutBinding.descriptorCount              = 1;
    storageLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {
        uniformLayoutBinding,
        storageLayoutBinding
    };

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount                    = ARRAY_SIZE( bindings );
    createInfo.pBindings                       = bindings;

    CHECK_VK( vkCreateDescriptorSetLayout( device, &createInfo, nullptr, &descriptorSetLayout ) );
    //printf( "MandelbrotComputeJob[%d:%d]: defined %d descriptors\n", hCompute, handle, createInfo.bindingCount );

    return true;
}


bool MandelbrotComputeJob::_createDescriptorSet()
{
    // Bind shader descriptors to buffers

    // TODO: check if descriptorPool has been exhausted

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool              = descriptorPool;
    allocInfo.descriptorSetCount          = 1;
    allocInfo.pSetLayouts                 = &descriptorSetLayout;

    CHECK_VK( vkAllocateDescriptorSets( device, &allocInfo, &descriptorSet ) );
    //printf( "MandelbrotComputeJob[%d:%d]: created %d descriptor sets\n", hCompute, handle, allocInfo.descriptorSetCount );

    if ( descriptorSet == 0 ) {
        printf( "ERROR: MandelbrotComputeJob[%d:%d] failed to alloc descriptors (pool exhausted?)\n", hCompute, handle );
        return false;
    }

    VkDescriptorBufferInfo descriptorUniformBufferInfo = {};
    descriptorUniformBufferInfo.buffer                 = uniformBuffer;
    descriptorUniformBufferInfo.offset                 = 0;
    descriptorUniformBufferInfo.range                  = uniformBufferSize;

    VkDescriptorBufferInfo descriptorStorageBufferInfo = {};
    descriptorStorageBufferInfo.buffer                 = outputBuffer;
    descriptorStorageBufferInfo.offset                 = 0;
    descriptorStorageBufferInfo.range                  = outputBufferSize;

    VkWriteDescriptorSet writeUniformSet = {};
    writeUniformSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeUniformSet.dstSet               = descriptorSet;
    writeUniformSet.dstBinding           = 0;
    writeUniformSet.descriptorCount      = 1;
    writeUniformSet.descriptorType       = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeUniformSet.pBufferInfo          = &descriptorUniformBufferInfo;
    vkUpdateDescriptorSets( device, 1, &writeUniformSet, 0, nullptr );

    VkWriteDescriptorSet writeStorageSet = {};
    writeStorageSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeStorageSet.dstSet               = descriptorSet;
    writeStorageSet.dstBinding           = 1;
    writeStorageSet.descriptorCount      = 1;
    writeStorageSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeStorageSet.pBufferInfo          = &descriptorStorageBufferInfo;
    vkUpdateDescriptorSets( device, 1, &writeStorageSet, 0, nullptr );

    unsigned int numDescriptors = 2;
    //printf( "MandelbrotComputeJob[%d:%d]: bound %d descriptors\n", hCompute, handle, numDescriptors );

    return true;
}


void MandelbrotComputeJob::_destroy()
{
    //printf( "MandelbrotComputeJob[%d:%d]::destroy()\n", hCompute, handle );

    if ( destroyed )
        return;

    numInstances--;

    SpinLockGuard lock( spinLock );

    CHECK_VK( vkResetCommandBuffer( commandBuffer, 0 ) );
    vkFreeCommandBuffers( device, commandPool, 1, &commandBuffer );
    CHECK_VK( vkFreeDescriptorSets( device, descriptorPool, 1, &descriptorSet ) );
    vkFreeMemory( device, outputBufferMemory, nullptr );
    vkDestroyBuffer( device, outputBuffer, nullptr );
    vkFreeMemory( device, uniformBufferMemory, nullptr );
    vkDestroyBuffer( device, uniformBuffer, nullptr );
    vkDestroyFence( device, fence, nullptr );

    // Free the static resources shared by all instances
    if ( numInstances == 0 && pipeline ) {
        printf( "MandelbrotComputeJob[%d:%d]::destroy()\n", hCompute, handle );
        vkDestroyShaderModule( device, computeShaderModule, nullptr );
        vkDestroyDescriptorSetLayout( device, descriptorSetLayout, nullptr );
        vkDestroyPipelineLayout( device, pipelineLayout, nullptr );
        vkDestroyPipeline( device, pipeline, nullptr );

        pipeline = nullptr;
    }

    destroyed = true;
}


} // namespace pk
