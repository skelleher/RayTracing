
#include "compute.h"
#include "log.h"
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

static void      _cameraInit( const Camera& camera, camera_glsl_t* p_camera );
static compute_t hCompute;

int renderSceneVulkan( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned blockSize, bool debug, unsigned gpu )
{
    PerfTimer t;

    hCompute = computeAcquire( gpu );

    uint32_t        inputWidth  = 1;
    uint32_t        inputHeight = 1;
    RayTracerJobPtr job         = RayTracerJob::create( hCompute, inputWidth, inputHeight, cols, rows );

    ComputeBufferDims sceneBufferDims    = { scene.objects.size() + 1, 1, sizeof( sphere_glsl_t ) };
    ComputeBufferDims materialBufferDims = { scene.objects.size() + 1, 1, sizeof( material_glsl_t ) };
    job->sceneBuffer.resize( sceneBufferDims );
    job->materialsBuffer.resize( materialBufferDims );

    //size_t sceneSize     = scene.objects.size() * sizeof( sphere_glsl_t );
    size_t materialsSize = scene.objects.size() * sizeof( material_glsl_t );
    printf( "Allocated %zd device bytes : %zd objects\n", job->sceneBuffer.size(), scene.objects.size() );
    printf( "Allocated %zd device bytes : %zd materials\n", job->materialsBuffer.size(), scene.objects.size() );

    printf( "Allocated %zd device bytes : context\n", job->uniformBuffer.size() );
    job->uniformBuffer.map();
    render_context_glsl_t* pdContext = (render_context_glsl_t*)job->uniformBuffer.mapped;

    _cameraInit( camera, &pdContext->camera );

    pdContext->sceneSize            = (uint32_t)scene.objects.size();
    pdContext->outputHeight         = rows;
    pdContext->outputWidth          = cols;
    pdContext->num_aa_samples       = num_aa_samples;
    pdContext->max_ray_depth        = max_ray_depth;
    pdContext->applyGammaCorrection = true;
    pdContext->debug                = debug;
    pdContext->monochrome           = true;
    pdContext->magic                = 0xDEADBEEF;
    pdContext->clock_ticks          = (uint32_t)PerfTimer::SystemTimeMilliseconds();

    // Copy the Scene to device
    // Flatten the Scene object to an array of sphere_t, which is what Scene should've been in the first place
    job->sceneBuffer.map();
    uint32_t* pSceneMagic = (uint32_t*)job->sceneBuffer.mapped;
    *pSceneMagic          = 0xDEADBEEF;

    //sphere_glsl_t* pdScene = (sphere_glsl_t*)job->sceneBuffer.mapped;
    uint8_t* pdScene = (uint8_t*)( (uint8_t*)job->sceneBuffer.mapped + 4 ); // + 16 if std140

    uint8_t* p = pdScene;
    unsigned i = 0;
    for ( IVisible* obj : scene.objects ) {
        Sphere*        s1 = dynamic_cast<Sphere*>( obj );
        sphere_glsl_t* s2 = (sphere_glsl_t*)p;
        s2->center_x      = s1->center.x;
        s2->center_y      = s1->center.y;
        s2->center_z      = s1->center.z;
        s2->radius        = s1->radius;
        s2->materialID    = i;

        i++;
        //p++;
        p += sizeof( sphere_glsl_t ); // TODO: What is the alignment of each struct in an SSBO array w/ std140?
    }
    job->sceneBuffer.unmap();
    printf( "Copied %zd objects to device\n", scene.objects.size() );

    job->materialsBuffer.map();
    uint32_t* pMaterialsMagic = (uint32_t*)job->materialsBuffer.mapped;
    *pMaterialsMagic          = 0xC001C0DE;

    //material_glsl_t* pdMaterials = (material_glsl_t*)job->sceneBuffer.mapped;
    material_glsl_t* pdMaterials = (material_glsl_t*)( (uint8_t*)job->materialsBuffer.mapped + 4 ); // + 16 if std140

    material_glsl_t* m = pdMaterials;
    for ( IVisible* obj : scene.objects ) {
        Sphere*          s1 = dynamic_cast<Sphere*>( obj );
        material_glsl_t* m2 = (material_glsl_t*)m;

        // Deep copy material to the GPU
        m2->type            = (uint32_t)s1->material->type;
        m2->albedo_r        = s1->material->albedo.r();
        m2->albedo_g        = s1->material->albedo.g();
        m2->albedo_b        = s1->material->albedo.b();
        m2->blur            = s1->material->blur;
        m2->refractionIndex = s1->material->refractionIndex;

        m++;
    }
    job->materialsBuffer.unmap();
    printf( "Copied %zd materials to device\n", scene.objects.size() );

    PerfTimer frame;
    // Render the scene
    computeSubmitJob( *job, hCompute );
    computeWaitForJob( job->handle, COMPUTE_NO_TIMEOUT, hCompute );

    float    msPerFrame = frame.ElapsedMilliseconds();
    uint64_t rays       = rows * cols * num_aa_samples;
    printf( "renderSceneVulkan: %f ms (%f ms per frame, %0.2f M rays / s)\n", t.ElapsedMilliseconds(), msPerFrame, (rays / (msPerFrame/1'000.0f))/1'000'000 );

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

    return 0;
}


static void _cameraInit( const Camera& camera, camera_glsl_t* p_camera )
{
    p_camera->origin.x      = camera.origin.x;
    p_camera->origin.y      = camera.origin.y;
    p_camera->origin.z      = camera.origin.z;
    p_camera->lookat.x      = camera.lookat.x;
    p_camera->lookat.y      = camera.lookat.y;
    p_camera->lookat.z      = camera.lookat.z;
    p_camera->vfov          = camera.vfov;
    p_camera->aspect        = camera.aspect;
    p_camera->aperture      = camera.aperture;
    p_camera->focusDistance = camera.focusDistance;

    p_camera->lensRadius = p_camera->aperture / 2.0f;

    float theta      = RADIANS( p_camera->vfov );
    float halfHeight = float( tan( theta / 2.0f ) );
    float halfWidth  = p_camera->aspect * halfHeight;

    p_camera->w.x = p_camera->origin.x - p_camera->lookat.x;
    p_camera->w.y = p_camera->origin.y - p_camera->lookat.y;
    p_camera->w.z = p_camera->origin.z - p_camera->lookat.z;
    //p_camera->w   = normalize( p_camera->w );
    p_camera->w.normalize();

    vec3 up     = { 0, 1, 0 };
    p_camera->u = cross( up, p_camera->w );
    //p_camera->u   = normalize( p_camera->u );
    p_camera->u.normalize();
    p_camera->v = cross( p_camera->w, p_camera->u );

    p_camera->leftCorner.x = p_camera->origin.x - halfWidth * p_camera->focusDistance * p_camera->u.x - halfHeight * p_camera->focusDistance * p_camera->v.x - p_camera->focusDistance * p_camera->w.x;
    p_camera->leftCorner.y = p_camera->origin.y - halfWidth * p_camera->focusDistance * p_camera->u.y - halfHeight * p_camera->focusDistance * p_camera->v.y - p_camera->focusDistance * p_camera->w.y;
    p_camera->leftCorner.z = p_camera->origin.z - halfWidth * p_camera->focusDistance * p_camera->u.z - halfHeight * p_camera->focusDistance * p_camera->v.z - p_camera->focusDistance * p_camera->w.z;

    p_camera->horizontal.x = 2 * halfWidth * p_camera->focusDistance * p_camera->u.x;
    p_camera->horizontal.y = 2 * halfWidth * p_camera->focusDistance * p_camera->u.y;
    p_camera->horizontal.z = 2 * halfWidth * p_camera->focusDistance * p_camera->u.z;

    p_camera->vertical.x = 2 * halfHeight * p_camera->focusDistance * p_camera->v.x;
    p_camera->vertical.y = 2 * halfHeight * p_camera->focusDistance * p_camera->v.y;
    p_camera->vertical.z = 2 * halfHeight * p_camera->focusDistance * p_camera->v.z;
}


} // namespace pk
