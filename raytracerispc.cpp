#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "raytracer.h"
#include "raytracer_ispc.h"
#include "sphere.h"
#include "thread_pool.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

namespace pk
{

typedef struct _RenderThreadContext {
    const Camera*          camera;
    const ispc::sphere_t*  scene;
    uint32_t               sceneSize;
    uint32_t*              framebuffer;
    uint32_t               rows;
    uint32_t               cols;
    uint32_t               num_aa_samples;
    uint32_t               max_ray_depth;
    uint32_t               blockID;
    uint32_t               blockSize;
    uint32_t               xOffset;
    uint32_t               yOffset;
    std::atomic<uint32_t>* blockCount;
    uint32_t               totalBlocks;
    bool                   debug;

    _RenderThreadContext() :
        scene( nullptr ),
        camera( nullptr ),
        framebuffer( nullptr ),
        blockSize( 0 ),
        xOffset( 0 ),
        yOffset( 0 ),
        debug( false )
    {
    }
} RenderThreadContext;


static bool _renderJobISPC( void* context, uint32_t tid );


int renderSceneISPC( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    // Spin up a pool of render threads
    // Allocate width+1 and height+1 blocks to handle case where image is not an even multiple of block size
    uint32_t      widthBlocks  = uint32_t( float( cols / blockSize ) ) + 1;
    uint32_t      heightBlocks = uint32_t( float( rows / blockSize ) ) + 1;
    uint32_t      numBlocks    = heightBlocks * widthBlocks;
    thread_pool_t tp           = threadPoolInit( numThreads );

    printf( "Render %d x %d: blockSize %d x %d, %d blocks, [%d:%d] threads \n",
        cols, rows, blockSize, blockSize, numBlocks, tp, numThreads );

    // Flatten the Scene object to an array of ispc::sphere_t
    // TODO: convert to SoA
    size_t          sceneSize = sizeof( ispc::sphere_t ) * scene.objects.size();
    ispc::sphere_t* pScene    = (ispc::sphere_t*)new uint8_t[ sceneSize ];
    //printf( "Allocated %zd bytes for %zd objects\n", sceneSize, scene.objects.size() );

    ispc::sphere_t* p = pScene;
    for ( IVisible* obj : scene.objects ) {
        Sphere*         s1 = dynamic_cast<Sphere*>( obj );
        ispc::sphere_t* s2 = (ispc::sphere_t*)p;

        s2->center[ 0 ] = s1->center.x;
        s2->center[ 1 ] = s1->center.y;
        s2->center[ 2 ] = s1->center.z;
        s2->radius      = s1->radius;

        s2->material.type            = (ispc::material_type_t)s1->material->type;
        s2->material.albedo[ 0 ]     = s1->material->albedo.r();
        s2->material.albedo[ 1 ]     = s1->material->albedo.g();
        s2->material.albedo[ 2 ]     = s1->material->albedo.b();
        s2->material.blur            = s1->material->blur;
        s2->material.refractionIndex = s1->material->refractionIndex;

        p++;
    }
    printf( "Flattened %zd scene objects to ISPC array\n", scene.objects.size() );

    // Initialize the camera
    ispc::RenderGangContext ispc_ctx;

    ispc_ctx.camera_origin[ 0 ]   = camera.origin.x;
    ispc_ctx.camera_origin[ 1 ]   = camera.origin.y;
    ispc_ctx.camera_origin[ 2 ]   = camera.origin.z;
    ispc_ctx.camera_vfov          = camera.vfov;
    ispc_ctx.camera_aspect        = camera.aspect;
    ispc_ctx.camera_aperture      = camera.aperture;
    ispc_ctx.camera_focusDistance = camera.focusDistance;
    ispc_ctx.camera_lookat[ 0 ]   = camera.lookat.x;
    ispc_ctx.camera_lookat[ 1 ]   = camera.lookat.y;
    ispc_ctx.camera_lookat[ 2 ]   = camera.lookat.z;
    ispc::cameraInitISPC( &ispc_ctx );

    memset( framebuffer, 0x00, rows * cols * sizeof( uint32_t ) );

    // Allocate a render context to pass to each worker job
    RenderThreadContext* contexts = new RenderThreadContext[ numBlocks ];

    std::atomic<uint32_t> blockCount = 0;
    uint32_t              blockID    = 0;
    uint32_t              yOffset    = 0;
    for ( uint32_t y = 0; y < heightBlocks; y++ ) {
        uint32_t xOffset = 0;
        for ( uint32_t x = 0; x < widthBlocks; x++ ) {
            RenderThreadContext* ctx = &contexts[ blockID ];
            ctx->scene               = pScene;
            ctx->sceneSize           = (uint32_t)scene.objects.size();
            ctx->camera              = &camera;
            ctx->framebuffer         = framebuffer;
            ctx->blockID             = blockID;
            ctx->blockSize           = blockSize;
            ctx->xOffset             = xOffset;
            ctx->yOffset             = yOffset;
            ctx->rows                = rows;
            ctx->cols                = cols;
            ctx->num_aa_samples      = num_aa_samples;
            ctx->max_ray_depth       = max_ray_depth;
            ctx->blockCount          = &blockCount;
            ctx->totalBlocks         = numBlocks;
            ctx->debug               = debug;

            threadPoolSubmitJob( tp, _renderJobISPC, ctx );

            //printf( "Submit block %d of %d\n", blockID, numBlocks );

            blockID++;
            xOffset += blockSize;
        }
        yOffset += blockSize;
    }

    // Wait for threads to complete
    while ( blockCount != numBlocks ) {
        delay( 1000 );
        printf( "." );
    }
    printf( "\n" );

    threadPoolDeinit( tp );
    delete[] contexts;
    ///////delete[] pScene;

    printf( "renderSceneISPC: %f s\n", t.ElapsedSeconds() );

    return 0;
}


static bool _renderJobISPC( void* context, uint32_t tid )
{
    RenderThreadContext* ctx = (RenderThreadContext*)context;

    // NOTE: there are TWO render contexts at play here:
    // RenderThreadContext is a single thread on the CPU
    // ispc::RenderGangContext is a single gang on the SIMD unit
    ispc::RenderGangContext ispc_ctx;

    ispc_ctx.scene          = ctx->scene;
    ispc_ctx.sceneSize      = ctx->sceneSize;
    ispc_ctx.framebuffer    = ctx->framebuffer;
    ispc_ctx.blockID        = ctx->blockID;
    ispc_ctx.blockSize      = ctx->blockSize;
    ispc_ctx.totalBlocks    = ctx->totalBlocks;
    ispc_ctx.xOffset        = ctx->xOffset;
    ispc_ctx.yOffset        = ctx->yOffset;
    ispc_ctx.rows           = ctx->rows;
    ispc_ctx.cols           = ctx->cols;
    ispc_ctx.num_aa_samples = ctx->num_aa_samples;
    ispc_ctx.max_ray_depth  = ctx->max_ray_depth;
    ispc_ctx.debug          = ctx->debug;

    bool rval = ispc::renderISPC( &ispc_ctx ); // blocking call

    // Notify main thread that we have completed the work
    if ( ctx->blockID == ctx->totalBlocks - 1 ) {
        std::atomic<uint32_t>* blockCount = ctx->blockCount;
        blockCount->exchange( ctx->totalBlocks );
    }

    return rval;
}

} // namespace pk
