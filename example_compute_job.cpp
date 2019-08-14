#include "example_compute_job.h"

#include "log.h"
#include "compute_buffer.h"
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

std::atomic<uint32_t> ExampleComputeJob::numInstances = 0;

// Instances of a compute job share the same shader binary and pipeline.
VulkanUtils::ComputeShaderProgram ExampleComputeJob::shaderProgram = {
    "shaders\\test_vulkan.spv"
};

// TODO: these never change (for a given pipeline) so could be
// set at pipeline creation stage via Push Constants instead of passed as uniforms.
struct UniformBufferObject {
    alignas( 4 ) uint32_t inputWidth;
    alignas( 4 ) uint32_t inputHeight;
    alignas( 4 ) uint32_t outputWidth;
    alignas( 4 ) uint32_t outputHeight;
};

struct Pixel {
    float value[ 4 ];
};

// factory method
std::unique_ptr<ExampleComputeJob> ExampleComputeJob::create( compute_t hCompute, uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight )
{
    std::unique_ptr<ExampleComputeJob> pJob( new ExampleComputeJob( hCompute, inputWidth, inputHeight, outputWidth, outputHeight ) );
    computeBindJob( *pJob, hCompute );

    return pJob;
}


// IComputeJob
void ExampleComputeJob::init()
{
    if ( initialized )
        return;

    ComputeBufferDims uniformBufferDims = { 1, 1, sizeof( UniformBufferObject ) };
    ComputeBufferDims inputBufferDims   = { inputWidth, inputHeight, sizeof( uint8_t ) };
    ComputeBufferDims outputBufferDims  = { outputWidth, outputHeight, sizeof( Pixel ) };

    uniformBuffer.init( vulkan, &shader, 0, uniformBufferDims, COMPUTE_BUFFER_UNIFORM, COMPUTE_BUFFER_SHARED );
    inputBuffer.init( vulkan, &shader, 1, inputBufferDims, COMPUTE_BUFFER_STORAGE, COMPUTE_BUFFER_SHARED );
    outputBuffer.init( vulkan, &shader, 2, outputBufferDims, COMPUTE_BUFFER_STORAGE, COMPUTE_BUFFER_SHARED );

    IComputeBuffer* buffers[] = {
        &uniformBuffer, &inputBuffer, &outputBuffer
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


void ExampleComputeJob::presubmit()
{
    //printf( "ExampleComputeJob[%d:%d]::presubmit()\n", hCompute, handle );

    // TEST: just to ensure that resize doesn't break anything:
    uniformBuffer.resize( uniformBuffer.dims );
    inputBuffer.resize( inputBuffer.dims );
    outputBuffer.resize( outputBuffer.dims );


    if ( uniformBuffer.sizeHasChanged || inputBuffer.sizeHasChanged || outputBuffer.sizeHasChanged ) {
        VulkanUtils::recordCommandBuffer( vulkan, &shader );

        uniformBuffer.sizeHasChanged = false;
        inputBuffer.sizeHasChanged   = false;
        outputBuffer.sizeHasChanged  = false;
    }

    // NOTE: if these never change you could do this in init()
    // instead of presubmit()
    struct UniformBufferObject ubo;
    ubo.inputWidth   = inputWidth;
    ubo.inputHeight  = inputHeight;
    ubo.outputWidth  = outputWidth;
    ubo.outputHeight = outputHeight;

    uniformBuffer.map();
    memcpy( uniformBuffer.mapped, &ubo, sizeof( ubo ) );
    uniformBuffer.unmap();

    inputBuffer.map();
    //
    // TODO: pass your input to compute shader here
    //
    inputBuffer.unmap();
}


void ExampleComputeJob::submit()
{
    if ( !shader.commandBuffer ) {
        return;
    }

    //printf( "ExampleComputeJob[%d:%d]::submit()\n", hCompute, handle );

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &shader.commandBuffer;

    CHECK_VK( vkResetFences( vulkan.device, 1, &shader.fence ) );

    // TODO: vkQueueSubmit() works better if we pass lots of command buffers, instead
    // of calling once per command buffer.
    // Compute should concatenate them and then periodically call submit?

    CHECK_VK( vkQueueSubmit( vulkan.queue, 1, &submitInfo, shader.fence ) );
}


void ExampleComputeJob::postsubmit( uint32_t timeoutMS )
{
    //printf( "ExampleComputeJob[%d:%d]::postsubmit()\n", hCompute, handle );

    uint64_t timeoutNS = timeoutMS * 1000000;
    VkResult rval      = vkWaitForFences( vulkan.device, 1, &shader.fence, VK_TRUE, timeoutNS );

    if ( rval == VK_TIMEOUT ) {
        printf( "ERROR: ExampleComputeJob[%d:%d]: timeout (%d ms)\n", hCompute, handle, timeoutMS );
        return;
    }

    if ( rval != VK_SUCCESS ) {
        printf( "ERROR: ExampleComputeJob[%d:%d]: error %d\n", hCompute, handle, rval );
        return;
    }
}


void ExampleComputeJob::save( const std::string outputPath )
{
    printf( "Saving to %s\n", outputPath.c_str() );

    outputBuffer.map();

    struct Pixel {
        float r;
        float g;
        float b;
        float a;
    };

    Pixel* pixels = (Pixel*)outputBuffer.mapped;

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

    outputBuffer.unmap();

    printf( "done\n" );
}


//
// Private implementation
//

void ExampleComputeJob::_destroy()
{
    //printf( "ExampleComputeJob[%d:%d]::destroy()\n", hCompute, handle );

    if ( destroyed )
        return;

    numInstances--;

    SpinLockGuard lock( spinLock );

    CHECK_VK( vkResetCommandBuffer( shader.commandBuffer, 0 ) );
    vkFreeCommandBuffers( vulkan.device, vulkan.commandPool, 1, &shader.commandBuffer );
    CHECK_VK( vkFreeDescriptorSets( vulkan.device, vulkan.descriptorPool, 1, &shader.descriptorSet ) );
    vkDestroyFence( vulkan.device, shader.fence, nullptr );

    uniformBuffer.free();
    inputBuffer.free();
    outputBuffer.free();

    // Free the static resources shared by all instances
    if ( numInstances == 0 && shaderProgram.pipeline ) {
        printf( "ExampleComputeJob[%d:%d]::destroy()\n", hCompute, handle );
        vkDestroyShaderModule( vulkan.device, shaderProgram.shaderModule, nullptr );
        vkDestroyDescriptorSetLayout( vulkan.device, shaderProgram.descriptorSetLayout, nullptr );
        vkDestroyPipelineLayout( vulkan.device, shaderProgram.pipelineLayout, nullptr );
        vkDestroyPipeline( vulkan.device, shaderProgram.pipeline, nullptr );

        shaderProgram.pipeline = nullptr;
    }

    destroyed = true;
}


} // namespace pk
