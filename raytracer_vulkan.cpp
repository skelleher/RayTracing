
#include "compute.h"
#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "raytracer.h"
#include "raytracer_compute.h"
#include "raytracer_glsl.h"
#include "scene.h"
#include "sphere.h"
#include "vector_cuda.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vulkan/vulkan.h>


namespace pk
{

static void      _createCamera( Camera* pdCamera );
static compute_t hCompute;

int renderSceneVulkan( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned blockSize, bool debug )
{
    PerfTimer t;

    hCompute = computeAcquire();

    uint32_t        inputWidth  = (uint32_t)scene.objects.size() * ( sizeof( sphere_glsl_t ) + sizeof( material_glsl_t ) );
    uint32_t        inputHeight = 1;
    RayTracerJobPtr job         = RayTracerJob::create( hCompute, inputWidth, inputHeight, cols, rows );

    printf( "Allocated %zd device bytes for context\n", job->uniformBuffer.size() );
    job->uniformBuffer.map();
    render_context_glsl_t* pdContext = (render_context_glsl_t*)job->uniformBuffer.mapped;
    //pdContext->camera         = pdCamera;
    //pdContext->scene          = pdScene;
    pdContext->numSceneObjects = (uint32_t)scene.objects.size();
    pdContext->outputHeight    = rows;
    pdContext->outputWidth     = cols;
    pdContext->num_aa_samples  = num_aa_samples;
    pdContext->max_ray_depth   = max_ray_depth;
    pdContext->gammaCorrection = true;
    pdContext->debug           = debug;

    // Create a copy of the Camera on the device [ gross hack because Camera is created in main() since before I refactored for CUDA ]
    //Camera* pdCamera = nullptr;
    //CHECK_CUDA( cudaMallocManaged( &pdCamera, sizeof( Camera ) * 2 ) );
    //    memcpy( &pdCamera[ 1 ], &camera, sizeof( camera ) );
    //    printf( "Allocated %zd device bytes for camera\n", sizeof( camera ) );
    //    _createCamera<<<1, 1>>>( pdCamera );

    //struct UniformBufferObject ubo;
    //_initCamera(&ubo.camera, camera);
    //ubo.numMaterials = 0;
    //ubo.numSpheres = 0;

    // Copy the Scene to device
    // Flatten the Scene object to an array of sphere_t, which is what Scene should've been in the first place
    job->inputBuffer.map();
    sphere_t* pdScene   = (sphere_t*)job->inputBuffer.mapped;
    size_t    sceneSize = sizeof( sphere_t ) * scene.objects.size();
    assert( sceneSize == job->inputBuffer.size() );
    printf( "Allocated %zd device bytes / %zd objects\n", sceneSize, scene.objects.size() );

    sphere_t* p = pdScene;
    for ( IVisible* obj : scene.objects ) {
        Sphere*   s1 = dynamic_cast<Sphere*>( obj );
        sphere_t* s2 = (sphere_t*)p;
        s2->center   = s1->center;
        s2->radius   = s1->radius;

        // Deep copy material to the GPU
        s2->material = *( s1->material );

        p++;
    }
    printf( "Copied %zd objects to device\n", scene.objects.size() );

    // Render the scene
    computeSubmitJob( *job, hCompute );
    computeWaitForJob( job->handle, COMPUTE_NO_TIMEOUT, hCompute );

    job->outputBuffer.map();
    uint32_t framebufferSize = rows * cols * sizeof( uint32_t );
    //memcpy( framebuffer, job->outputBuffer.mapped, framebufferSize );

    struct Pixel {
        float r;
        float g;
        float b;
        float a;
    };

    Pixel* pixels = (Pixel*)job->outputBuffer.mapped;

    // Convert image from floats to uint32_t
    // TODO: compute shader should operate directly on the framebuffer
    for ( uint32_t y = 0; y < job->outputBuffer.dims.height; y++ ) {
        for ( uint32_t x = 0; x < job->outputBuffer.dims.width; x++ ) {
            Pixel&  rgb = pixels[ y * job->outputBuffer.dims.width + x ];
            uint8_t _r  = ( uint8_t )( rgb.r * 255 );
            uint8_t _g  = ( uint8_t )( rgb.g * 255 );
            uint8_t _b  = ( uint8_t )( rgb.b * 255 );

            uint32_t color = ( _r << 24 ) | ( _g << 16 ) | ( _b << 8 );

            framebuffer[ y * job->outputBuffer.dims.width + x ] = color;
        }
    }

    computeRelease( hCompute );

    printf( "renderSceneVulkan: %f s\n", t.ElapsedSeconds() );

    return 0;
}


} // namespace pk
