#include "raytracer_compute.h"

#include "log.h"
#include "perf_timer.h"
#include "raytracer_glsl.h"
#include "utils.h"
#include "vulkan_utils.h"

#include <assert.h>

namespace pk
{

//
// Implementation of ray tracing as a GPU compute job
//

// TODO: ComputeInstance should query the GPU for best workgroupSize and pass it to create()
static const uint32_t WORK_GROUP_SIZE = 32;

std::atomic<uint32_t> RayTracerJob::numInstances = 0;

// Instances of a compute job share the same shader binary and pipeline.
VulkanUtils::ComputeShaderProgram RayTracerJob::shaderProgram = {
    "shaders\\raytracer.spv"
};


// factory method
std::unique_ptr<RayTracerJob> RayTracerJob::create( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight )
{
    std::unique_ptr<RayTracerJob> pJob( new RayTracerJob( hCompute, inputWidth, inputHeight, outputWidth, outputHeight ) );

    computeBindJob( *pJob, hCompute );

    return pJob;
}


// IComputeJob
void RayTracerJob::init()
{
    if ( initialized )
        return;

    ComputeBufferDims uniformBufferDims  = { 1, 1, sizeof( render_context_glsl_t ) };
    //ComputeBufferDims sceneBufferDims    = { inputWidth, inputHeight, sizeof( uint8_t ) };
    ComputeBufferDims sceneBufferDims    = { 1, 1, sizeof( sphere_glsl_t ) };
    ComputeBufferDims materialBufferDims = { 1, 1, sizeof( material_glsl_t ) };
    ComputeBufferDims outputBufferDims   = { outputWidth, outputHeight, sizeof( pixel ) };

    uniformBuffer.init  ( vulkan, &shader, 0, uniformBufferDims,  COMPUTE_BUFFER_UNIFORM, COMPUTE_BUFFER_SHARED );
    sceneBuffer.init    ( vulkan, &shader, 1, sceneBufferDims,    COMPUTE_BUFFER_STORAGE, COMPUTE_BUFFER_SHARED );
    materialsBuffer.init( vulkan, &shader, 2, materialBufferDims, COMPUTE_BUFFER_STORAGE, COMPUTE_BUFFER_SHARED );
    outputBuffer.init   ( vulkan, &shader, 3, outputBufferDims,   COMPUTE_BUFFER_STORAGE, COMPUTE_BUFFER_SHARED );

    IComputeBuffer* buffers[] = {
        &uniformBuffer, &sceneBuffer, &materialsBuffer, &outputBuffer
    };

    shaderProgram.workgroupSize   = WORK_GROUP_SIZE;
    shaderProgram.workgroupWidth  = (uint32_t)ceil( outputWidth / (float)shaderProgram.workgroupSize );
    shaderProgram.workgroupHeight = (uint32_t)ceil( outputHeight / (float)shaderProgram.workgroupSize );
    shaderProgram.workgroupDepth  = 1;

    shader.pProgram   = &shaderProgram;
    shader.ppBuffers  = buffers;
    shader.numBuffers = ARRAY_SIZE( buffers );

    VulkanUtils::createComputeShader( vulkan, &shader );
    initialized = true;
}


void RayTracerJob::presubmit()
{
    if ( uniformBuffer.sizeHasChanged || sceneBuffer.sizeHasChanged || materialsBuffer.sizeHasChanged || outputBuffer.sizeHasChanged ) {
        VulkanUtils::recordCommandBuffer( vulkan, &shader );

        uniformBuffer.sizeHasChanged   = false;
        sceneBuffer.sizeHasChanged     = false;
        materialsBuffer.sizeHasChanged = false;
        outputBuffer.sizeHasChanged    = false;
    }
}


void RayTracerJob::submit()
{
    if ( !shader.commandBuffer ) {
        return;
    }

    //printf( "RayTracerJob[%d:%d]::submit()\n", hCompute, handle );

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &shader.commandBuffer;

    CHECK_VK( vkResetFences( vulkan.device, 1, &shader.fence ) );
    CHECK_VK( vkQueueSubmit( vulkan.queue, 1, &submitInfo, shader.fence ) );
}


void RayTracerJob::postsubmit( uint32_t timeoutMS )
{
    //printf( "RayTracerJob[%d:%d]::postsubmit()\n", hCompute, handle );

    uint64_t timeoutNS = timeoutMS * 1'000'000;
    VkResult rval      = vkWaitForFences( vulkan.device, 1, &shader.fence, VK_TRUE, timeoutNS );

    if ( rval == VK_TIMEOUT ) {
        printf( "ERROR: RayTracerJob[%d:%d]: timeout (%d ms)\n", hCompute, handle, timeoutMS );
        return;
    }

    if ( rval != VK_SUCCESS ) {
        printf( "ERROR: RayTracerJob[%d:%d]: error %d\n", hCompute, handle, rval );
        return;
    }
}


//
// Private Implementation
//

void RayTracerJob::_destroy()
{
    //printf( "RayTracerJob[%d:%d]::destroy()\n", hCompute, handle );

    if ( destroyed )
        return;

    numInstances--;

    SpinLockGuard lock( spinLock );

    CHECK_VK( vkResetCommandBuffer( shader.commandBuffer, 0 ) );
    vkFreeCommandBuffers( vulkan.device, vulkan.commandPool, 1, &shader.commandBuffer );
    CHECK_VK( vkFreeDescriptorSets( vulkan.device, vulkan.descriptorPool, 1, &shader.descriptorSet ) );
    vkDestroyFence( vulkan.device, shader.fence, nullptr );

    uniformBuffer.free();
    sceneBuffer.free();
    materialsBuffer.free();
    outputBuffer.free();

    // Free the static resources shared by all instances
    if ( numInstances == 0 && shaderProgram.pipeline ) {
        printf( "RayTracerJob[%d:%d]::destroy()\n", hCompute, handle );
        vkDestroyShaderModule( vulkan.device, shaderProgram.shaderModule, nullptr );
        vkDestroyDescriptorSetLayout( vulkan.device, shaderProgram.descriptorSetLayout, nullptr );
        vkDestroyPipelineLayout( vulkan.device, shaderProgram.pipelineLayout, nullptr );
        vkDestroyPipeline( vulkan.device, shaderProgram.pipeline, nullptr );

        shaderProgram.pipeline = nullptr;
    }

    destroyed = true;
}


} // namespace pk
