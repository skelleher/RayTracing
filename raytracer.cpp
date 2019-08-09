#include "raytracer.h"

#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "sphere.h"
#include "thread_pool.h"
#include "vector_cuda.h"

#include <assert.h>
#include <atomic>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <thread>

namespace pk
{

typedef struct _RenderThreadContext {
    const Camera*          camera;
    const sphere_t*        scene;
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
    bool                   recursive;

    _RenderThreadContext() :
        scene( nullptr ),
        camera( nullptr ),
        framebuffer( nullptr ),
        blockSize( 0 ),
        xOffset( 0 ),
        yOffset( 0 ),
        debug( false ),
        recursive( false )
    {
    }
} RenderThreadContext;


static bool    _sceneHit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit );
static vector3 _color_recursive( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth );
static vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth );
static vector3 _background( const ray& r );
static bool    _renderJob( void* context, uint32_t tid );


int renderScene( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    // Spin up a pool of render threads
    // Allocate width+1 and height+1 blocks to handle case where image is not an even multiple of block size
    uint32_t      widthBlocks  = uint32_t( float( cols / blockSize ) ) + 1;
    uint32_t      heightBlocks = uint32_t( float( rows / blockSize ) ) + 1;
    uint32_t      numBlocks    = heightBlocks * widthBlocks;

    printf( "Render %d x %d: blockSize %d x %d, %d blocks\n",
        cols, rows, blockSize, blockSize, numBlocks );


    // Flatten the Scene object to an array of sphere_t, which is what Scene should've been in the first place
    size_t    sceneSize = sizeof( sphere_t ) * scene.objects.size();
    sphere_t* pScene    = (sphere_t*)new uint8_t[ sceneSize ];
    //printf( "Allocated %zd bytes for %zd objects\n", sceneSize, scene.objects.size() );

    sphere_t* p = pScene;
    for ( IVisible* obj : scene.objects ) {
        Sphere*   s1 = dynamic_cast<Sphere*>( obj );
        sphere_t* s2 = (sphere_t*)p;
        s2->center   = s1->center;
        s2->radius   = s1->radius;
        s2->material = *( s1->material );

        p++;
    }
    printf( "Flattened %zd scene objects to array\n", scene.objects.size() );


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
            ctx->recursive           = recursive;

            threadPoolSubmitJob( Function(_renderJob, ctx) );

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
    //delete[] pScene;

    printf( "renderScene: %f s\n", t.ElapsedSeconds() );

    return 0;
}

static bool _renderJob( void* context, uint32_t tid )
{
    UNUSED( tid );

    const RenderThreadContext* ctx = (const RenderThreadContext*)context;

    //printf( "start %dx%d block %d of %d AA:%d MD:%d R:%d %d x %d x %d\n",
    //    ctx->cols, ctx->rows,
    //    ctx->blockID, ctx->totalBlocks, ctx->num_aa_samples, ctx->max_ray_depth, ctx->recursive,
    //    ctx->xOffset, ctx->yOffset, ctx->blockSize
    //    );

    for ( uint32_t y = ctx->yOffset; y < ctx->yOffset + ctx->blockSize; y++ ) {
        // Don't render out of bounds (in case where image is not an even multiple of block size)
        if ( y >= ctx->rows )
            break;

        for ( uint32_t x = ctx->xOffset; x < ctx->xOffset + ctx->blockSize; x++ ) {
            // Don't render out of bounds (in case where image is not an even multiple of block size)
            if ( x >= ctx->cols )
                break;

            // TEST
            if ( ctx->debug && ( y == ctx->yOffset || y == ctx->yOffset + ctx->blockSize - 1 || x == ctx->xOffset || x == ctx->xOffset + ctx->blockSize - 1 ) ) {
                ctx->framebuffer[ y * ctx->cols + x ] = 0xFF000000;
                continue;
            }

            // Sample each pixel in image space, with anti-aliasing

            vector3 color( 0, 0, 0 );
            for ( uint32_t s = 0; s < ctx->num_aa_samples; s++ ) {
                float u = float( x + random() ) / float( ctx->cols );
                float v = float( y + random() ) / float( ctx->rows );
                ray   r = ctx->camera->getRay( u, v );

                if ( ctx->recursive ) {
                    color += _color_recursive( r, ctx->scene, ctx->sceneSize, 0, ctx->max_ray_depth );
                } else {
                    color += _color( r, ctx->scene, ctx->sceneSize, 0, ctx->max_ray_depth );
                }
            }
            color /= float( ctx->num_aa_samples );

            // Apply 2.0 Gamma correction
            color = vector3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

            uint8_t  _r  = ( uint8_t )( 255.99 * color.x );
            uint8_t  _g  = ( uint8_t )( 255.99 * color.y );
            uint8_t  _b  = ( uint8_t )( 255.99 * color.z );
            uint32_t rgb = ( (uint32_t)_r << 24 ) | ( (uint32_t)_g << 16 ) | ( (uint32_t)_b << 8 );

            ctx->framebuffer[ y * ctx->cols + x ] = rgb;
        }
    }

    // Notify main thread that we have completed the work
    if ( ctx->blockID == ctx->totalBlocks - 1 ) {
        std::atomic<uint32_t>* blockCount = ctx->blockCount;
        blockCount->exchange( ctx->totalBlocks );
    }

    //printf( "block %d of %d (%d) DONE\n", ctx->blockID, ctx->totalBlocks, ctx->blockCount->load() );

    return true;
}

// Recursively trace each ray through objects/materials
static vector3 _color_recursive( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth )
{
    hit_info hit;

    if ( _sceneHit( scene, sceneSize, r, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
        vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
        return 0.5f * vector3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
        if ( depth < max_depth ) {
            vector3 target = hit.point + hit.normal + randomInUnitSphere();
            return 0.5f * _color_recursive( ray( hit.point, target - hit.point ), scene, sceneSize, depth + 1, max_depth );
        } else {
            return vector3( 0, 0, 0 );
        }
#else
        ray     scattered;
        vector3 attenuation;
        if ( depth < max_depth && materialScatter( hit.material, r, hit, &attenuation, &scattered ) ) {
            return attenuation * _color_recursive( scattered, scene, sceneSize, depth + 1, max_depth );
        } else {
            return vector3( 0, 0, 0 );
        }
#endif
    }

    return _background( r );
}

// Non-recursive version
static vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( _sceneHit( scene, sceneSize, scattered, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
            vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            return 0.5f * vector3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
            vector3 target = hit.point + hit.normal + randomInUnitSphere();
            scattered      = ray( hit.point, target - hit.point );
            color *= 0.5f;
#else
            if ( materialScatter( hit.material, scattered, hit, &attenuation, &scattered ) ) {
                color *= attenuation;
            } else {
                break;
            }
#endif
        } else {
            color *= _background( scattered );
            break;
        }
    }

    return color;
}


static vector3 _background( const ray& r )
{
    vector3 unitDirection = r.direction.normalized();
    float   t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vector3( 1.0f, 1.0f, 1.0f ) + t * vector3( 0.5f, 0.7f, 1.0f );
}


static bool _sceneHit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit )
{
    bool     rval         = false;
    float    closestSoFar = max;
    hit_info hit;

    for ( unsigned i = 0; i < sceneSize; i++ ) {
        const sphere_t& sphere = scene[ i ];

        hit_info tmp;
        if ( sphereHit( sphere, r, min, closestSoFar, &tmp ) ) {
            rval         = true;
            closestSoFar = tmp.distance;
            hit          = tmp;
        }
    }

    *p_hit = hit;
    return rval;
}

} // namespace pk
