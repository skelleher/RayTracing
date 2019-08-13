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
    const Camera*           camera;
    const ispc::sphere_t*   scene;
    const ispc::material_t* materials;
    uint32_t                sceneSize;
    uint32_t*               framebuffer;
    uint32_t                rows;
    uint32_t                cols;
    uint32_t                num_aa_samples;
    uint32_t                max_ray_depth;
    uint32_t                blockID;
    uint32_t                blockSize;
    uint32_t                xOffset;
    uint32_t                yOffset;
    std::atomic<uint32_t>*  blockCount;
    uint32_t                totalBlocks;
    bool                    debug;

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


int renderSceneISPC( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned blockSize, bool debug )
{
    PerfTimer t;

    // Spin up a pool of render threads
    // Allocate width+1 and height+1 blocks to handle case where image is not an even multiple of block size
    uint32_t      widthBlocks  = uint32_t( float( cols / blockSize ) ) + 1;
    uint32_t      heightBlocks = uint32_t( float( rows / blockSize ) ) + 1;
    uint32_t      numBlocks    = heightBlocks * widthBlocks;

    printf( "Render %d x %d: blockSize %d x %d, %d blocks\n",
        cols, rows, blockSize, blockSize, numBlocks );

    // Flatten the Scene object to an SoA ispc::sphere_t
    size_t sceneSize = scene.objects.size();

    ispc::sphere_t _scene;
    _scene.center_x   = new float[ sceneSize ];
    _scene.center_y   = new float[ sceneSize ];
    _scene.center_z   = new float[ sceneSize ];
    _scene.radius     = new float[ sceneSize ];
    _scene.materialID = new uint32_t[ sceneSize ];
#ifdef MATERIAL_SHADE
    ispc::material_t _materials;
    _materials.type            = new ispc::material_type_t[ sceneSize ];
    _materials.albedo_r        = new float[ sceneSize ];
    _materials.albedo_g        = new float[ sceneSize ];
    _materials.albedo_b        = new float[ sceneSize ];
    _materials.blur            = new float[ sceneSize ];
    _materials.refractionIndex = new float[ sceneSize ];
#endif

    uint32_t sphereID   = 0;
    uint32_t materialID = 0;
    for ( IVisible* obj : scene.objects ) {
        Sphere* s1 = dynamic_cast<Sphere*>( obj );

        _scene.center_x[ sphereID ] = s1->center.x;
        _scene.center_y[ sphereID ] = s1->center.y;
        _scene.center_z[ sphereID ] = s1->center.z;
        _scene.radius[ sphereID ]   = s1->radius;

#ifdef MATERIAL_SHADE
        _scene.materialID[ sphereID ] = materialID;

        _materials.type[ materialID ]            = (ispc::material_type_t)s1->material->type;
        _materials.albedo_r[ materialID ]        = s1->material->albedo.r();
        _materials.albedo_g[ materialID ]        = s1->material->albedo.g();
        _materials.albedo_b[ materialID ]        = s1->material->albedo.b();
        _materials.blur[ materialID ]            = s1->material->blur;
        _materials.refractionIndex[ materialID ] = s1->material->refractionIndex;
        materialID++;
#endif

        sphereID++;
    }
    printf( "Flattened %d scene objects and %d materials to ISPC array\n", sphereID, materialID );

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
            ctx->scene               = &_scene;
            ctx->materials           = &_materials;
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

            threadPoolSubmitJob( Function(_renderJobISPC, ctx) );

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

    delete[] contexts;
    delete[] _scene.center_x;
    delete[] _scene.center_y;
    delete[] _scene.center_z;
    delete[] _scene.radius;
#ifdef MATERIAL_SHADE
    delete[] _scene.materialID;
#endif

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
    ispc_ctx.materials      = ctx->materials;
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
