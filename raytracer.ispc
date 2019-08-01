
//#define DIFFUSE_SHADE
//#define NORMAL_SHADE

// Use the built-in prng,
// otherwise use a custom one.
#define ISPC_RANDOM


// TODO: a vector3 struct with operator overloading for +-*/
typedef float<3> vector3;

#define M_PI 3.14159265358979323846f
#define RADIANS( x ) ( (x)*M_PI / 180.0f )

#define FLT_MAX  3.402823466e+38F
#define FLT_MIN -FLT_MAX


// NOTE: any struct that is typedef'd MUST have a _tag in order to match a function signature
// NOTE: ISPC struct fields are unbound by default;
// you can declare an instance of struct as either uniform or varying

// TODO: operator overloading for +-*/
struct ray {
    vector3 origin;
    vector3 direction;
};


enum material_type_t {
    MATERIAL_NONE    = 0,
    MATERIAL_DIFFUSE = 1,
    MATERIAL_METAL   = 2,
    MATERIAL_GLASS   = 3,
}; 


struct material_t {
    material_type_t type;
    float           albedo[3];
    float           blur;
    float           refractionIndex;
};


struct hit_info {
    float      distance;
    vector3    point;
    vector3    normal;
    material_t   material;
};


struct sphere_t {
    float      center[3];
    float      radius;
    material_t material; // TODO: should be an index into a materials array
};


struct camera_t {
    vector3 origin;
    float   vfov;
    float   aspect;
    float   aperture;
    vector3 lookat;
    float   focusDistance;

    vector3 leftCorner;
    vector3 horizontal;
    vector3 vertical;
    vector3 u;
    vector3 v;
    vector3 w;
    float   lensRadius;
};


struct RenderGangContext {
    float   camera_origin[3];
    float   camera_vfov;
    float   camera_aspect;
    float   camera_aperture;
    float   camera_lookat[3];
    float   camera_focusDistance;

    const sphere_t*      scene;
    unsigned int32       sceneSize;
    unsigned int32*      framebuffer;
    unsigned int32       rows;
    unsigned int32       cols;
    unsigned int32       num_aa_samples;
    unsigned int32       max_ray_depth;
    unsigned int32       blockID;
    unsigned int32       blockSize;
    unsigned int32       totalBlocks;
    unsigned int32       xOffset;
    unsigned int32       yOffset;
    bool                 debug;
};

static uniform camera_t s_camera;


static inline varying float   _length( varying vector3& v );
static inline varying vector3 _normalize( varying vector3& v );
static inline varying float   _dot( varying vector3& v1, varying vector3& v2 );
static inline varying vector3 _cross( varying vector3 &v1, varying vector3 &v2 );
static inline varying vector3 _pointOnRay( varying ray& r, varying float distance );

static inline uniform float   _length( uniform vector3& v );
static inline uniform vector3 _normalize( uniform vector3& v );
static inline uniform float   _dot( uniform vector3& v1, uniform vector3& v2 );
static inline uniform vector3 _cross( uniform vector3 &v1, uniform vector3 &v2 );
static inline uniform vector3 _pointOnRay( uniform ray& r, uniform float distance );

static ray _cameraGetRay( float u, float v );

static vector3 _blockColor( int blockID, int totalBlocks );
static vector3 _randomColor( float u, float v );
static vector3 _gradient( float u, float v );
static vector3 _background( ray& r );
static vector3 _sky( float u, float v );
static vector3 _color( ray& r, const sphere_t* scene, unsigned int32 sceneSize, unsigned int32 max_depth );
static bool    _sceneHit( ray& r, const uniform sphere_t* varying scene, unsigned int32 sceneSize, float t_min, float t_max, varying hit_info* uniform p_hit );
static bool    _sphereHit( ray& r, const uniform sphere_t* varying sphere, float t_min, float t_max, varying hit_info* uniform p_hit );

static bool _materialScatter( ray& r, material_t& d, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered  );
static bool _diffuseScatter( ray& r, material_t& d, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered  );
static bool _metalScatter( ray& r, material_t& m, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered  );
static bool _glassScatter( ray& r, material_t& g, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered  );

static vector3 _reflect( vector3& v, vector3& normal );
static bool    _refract( vector3& v, vector3& normal, float ni_over_nt, varying vector3* uniform refracted );
static float   _schlick( float cosine, float refractionIndex );

#ifdef ISPC_RANDOM
static struct RNGState _rngstate;
static inline void    _randomInit( uniform int32 seed );
#else
static inline void    _randomInit( varying unsigned int32 seed );
#endif

static inline vector3 _randomInUnitSphere();
static inline vector3 _randomOnUnitDisk();
static inline float   _random();


export void cameraInitISPC( uniform RenderGangContext * uniform ctx )
{
    //print( "cameraInitISPC\n" );

    s_camera.origin.x      = ctx->camera_origin[0];
    s_camera.origin.y      = ctx->camera_origin[1];
    s_camera.origin.z      = ctx->camera_origin[2];
    s_camera.vfov          = ctx->camera_vfov;
    s_camera.aspect        = ctx->camera_aspect;
    s_camera.aperture      = ctx->camera_aperture;
    s_camera.lookat.x      = ctx->camera_lookat[0];
    s_camera.lookat.y      = ctx->camera_lookat[1];
    s_camera.lookat.z      = ctx->camera_lookat[2];
    s_camera.focusDistance = ctx->camera_focusDistance;

    s_camera.lensRadius = s_camera.aperture / 2.0f;

    uniform float theta      = RADIANS( s_camera.vfov );
    uniform float halfHeight = (float)tan( theta / 2.0f );
    uniform float halfWidth  = s_camera.aspect * halfHeight;

    s_camera.w.x = s_camera.origin.x - s_camera.lookat.x;
    s_camera.w.y = s_camera.origin.y - s_camera.lookat.y;
    s_camera.w.z = s_camera.origin.z - s_camera.lookat.z;
    s_camera.w   = _normalize( s_camera.w );

    uniform vector3 up = { 0, 1, 0 };
    s_camera.u   = _cross( up, s_camera.w );
    s_camera.u   = _normalize( s_camera.u );
    s_camera.v   = _cross( s_camera.w, s_camera.u );

    s_camera.leftCorner.x = s_camera.origin.x - halfWidth * s_camera.focusDistance * s_camera.u.x - halfHeight * s_camera.focusDistance * s_camera.v.x - s_camera.focusDistance * s_camera.w.x;
    s_camera.leftCorner.y = s_camera.origin.y - halfWidth * s_camera.focusDistance * s_camera.u.y - halfHeight * s_camera.focusDistance * s_camera.v.y - s_camera.focusDistance * s_camera.w.y;
    s_camera.leftCorner.z = s_camera.origin.z - halfWidth * s_camera.focusDistance * s_camera.u.z - halfHeight * s_camera.focusDistance * s_camera.v.z - s_camera.focusDistance * s_camera.w.z;

    s_camera.horizontal.x = 2 * halfWidth * s_camera.focusDistance * s_camera.u.x;
    s_camera.horizontal.y = 2 * halfWidth * s_camera.focusDistance * s_camera.u.y;
    s_camera.horizontal.z = 2 * halfWidth * s_camera.focusDistance * s_camera.u.z;

    s_camera.vertical.x   = 2 * halfHeight * s_camera.focusDistance * s_camera.v.x;
    s_camera.vertical.y   = 2 * halfHeight * s_camera.focusDistance * s_camera.v.y;
    s_camera.vertical.z   = 2 * halfHeight * s_camera.focusDistance * s_camera.v.z;

    //print( "ISPC Camera(): \nfov % \naspect % \naperture % \n(%, %, %) -> (%, %, %) \nfocusDistance: %\n",
    //    s_camera.vfov,
    //    s_camera.aspect,
    //    s_camera.aperture,
    //    s_camera.origin.x, s_camera.origin.y, s_camera.origin.z,
    //    s_camera.lookat.x, s_camera.lookat.y, s_camera.lookat.z,
    //    s_camera.focusDistance
    //    );

    //print( "u[%, %, %] \nv[%, %, %] \nw[%, %, %] \nhorizontal[%, %, %] \nvertical[%, %, %]\n",
    //    s_camera.u.x, s_camera.u.y, s_camera.u.z,
    //    s_camera.v.x, s_camera.v.y, s_camera.v.z,
    //    s_camera.w.x, s_camera.w.y, s_camera.w.z,
    //    s_camera.horizontal.x, s_camera.horizontal.y, s_camera.horizontal.z,
    //    s_camera.vertical.x, s_camera.vertical.y, s_camera.vertical.z
    //);
}


export uniform bool renderISPC( uniform RenderGangContext * uniform ctx )
{
    //print( "renderISPC: blockID % of %\n", ctx->blockID, ctx->totalBlocks );

#ifdef ISPC_RANDOM
    _randomInit( ctx->blockID );
#else
    _randomInit( programIndex * ctx->blockID * ctx->xOffset * ctx->yOffset );
#endif

    for ( int y = ctx->yOffset; y < ctx->yOffset + ctx->blockSize; y += 1 )
    {
        for ( int x = ctx->xOffset + programIndex; x < ctx->xOffset + ctx->blockSize; x += programCount )
        {
            if ( x >= ctx->cols || y >= ctx->rows ) {
                continue;
            }

            //if ( ctx->debug && ( y == ctx->yOffset || y == ctx->yOffset + ctx->blockSize - 1 || x == ctx->xOffset || x == ctx->xOffset + ctx->blockSize - 1 ) ) {
            //    ctx->framebuffer[ y * ctx->cols + x ] = 0xFF000000;
            //    continue;
            //}

            unsigned int32 p = y * ctx->cols + x;

            vector3 color = { 0, 0, 0 };

            for ( unsigned int32 s = 0; s < ctx->num_aa_samples; s++ )
            {
                float u = ((float)x) / ctx->cols;
                float v = ((float)y) / ctx->rows;

                ray r = _cameraGetRay( u, v );

                vector3 _sample  = _color( r, ctx->scene, ctx->sceneSize, ctx->max_ray_depth );
                //vector3 _sample = _background( r );
                //vector3 _sample = _gradient( u, v );
                //vector3 _sample = _randomColor( u, v );
                //vector3 _sample = _blockColor( ctx->blockID, ctx->totalBlocks );

                color.r += _sample.r;
                color.g += _sample.g;
                color.b += _sample.b;
            }

            color.r /= ctx->num_aa_samples;
            color.g /= ctx->num_aa_samples;
            color.b /= ctx->num_aa_samples;

            // Apply 2.0 Gamma correction
            vector3 _color = { sqrt( color.r ), sqrt( color.g ), sqrt( color.b ) };
            color = _color;

            ctx->framebuffer[ p ] = ( ( int32 )( color.r * 255.99f ) << 24 ) | ( ( int32 )( color.g * 255.99f ) << 16 ) | ( ( int32 )( color.b * 255.99f ) << 8 );
        }
    }

    return true;
}


//
// Renderer
//


static vector3 _blockColor( int blockID, int totalBlocks )
{
    float c = ((float)blockID / (float)totalBlocks);
    vector3 color = { c, c, c };

    return color;
}


static vector3 _randomColor( float u, float v )
{
    float c = _random();
    vector3 color = { c, c, c };

    return color;
}


static vector3 _gradient( float u, float v )
{
    float r = u;
    float g = v;
    float b = 0.2f;

    //vector3 color = { r * 255.9f, g * 255.9f, b * 255.8f };
    vector3 color = { r, g, b };

    return color;
}


static vector3 _background( ray& r )
{
    vector3 unitDirection = _normalize( r.direction );
    float   t             = 0.5f * ( unitDirection.y + 1.0f );

    //return ( 1.0f - t ) * vector3( 1.0f, 1.0f, 1.0f ) + t * vector3( 0.5f, 0.7f, 1.0f );

    vector3 v1 = { 1.0f, 1.0f, 1.0f };
    vector3 v2 = { 0.5f, 0.7f, 1.0f  };

    vector3 color;
    color.r = (1.0f - t) * v1.x + t * v2.x;
    color.g = (1.0f - t) * v1.y + t * v2.y;
    color.b = (1.0f - t) * v1.z + t * v2.z;

    return color;
}


static vector3 _sky( float u, float v )
{
    vector3 horizontal = { 4, 0, 0 };
    vector3 vertical = { 0, 2, 0 };
    vector3 origin = { 0, 0, 0 };
    vector3 leftCorner = { -2, -1, -1 };

    ray r;
    r.origin = s_camera.origin;
//    r.direction.x = s_camera.leftCorner.x + u * s_camera.horizontal.x + v * s_camera.vertical.x;
//    r.direction.y = s_camera.leftCorner.y + u * s_camera.horizontal.y + v * s_camera.vertical.y;
//    r.direction.z = s_camera.leftCorner.z + u * s_camera.horizontal.z + v * s_camera.vertical.z;

    r.origin = origin;
    r.direction.x = leftCorner.x + u * horizontal.x + v * vertical.x;
    r.direction.y = leftCorner.y + u * horizontal.y + v * vertical.y;
    r.direction.z = leftCorner.z + u * horizontal.z + v * vertical.z;

    vector3 unit_direction = _normalize( r.direction );
    float t = 0.5f * (unit_direction.y + 1.0f);
    vector3 v1 = { 1, 1, 1 };
    vector3 v2 = { 0.5f, 0.7f, 1.0f };
    vector3 color;
    color.x = (1.0 - t) * v1.x + t * v2.x;
    color.y = (1.0 - t) * v1.y + t * v2.y;
    color.z = (1.0 - t) * v1.z + t * v2.z;

    return color;
}


static vector3 _color( ray& r, const sphere_t* scene, unsigned int32 sceneSize, unsigned int32 max_depth )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color = { 1.0f, 1.0f, 1.0f };

    for ( unsigned int32 i = 0; i < max_depth; i++ ) {
        if ( _sceneHit( scattered, scene, sceneSize, 0.001f, FLT_MAX, &hit ) ) {
#if defined( NORMAL_SHADE )
            //vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            //return 0.5f * vector3( normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f );
            vector3 normal;
            vector3 p = _pointOnRay( r, hit.distance );
            normal.x = p.x - 0;
            normal.y = p.y - 0;
            normal.z = p.z = -1;
            normal = _normalize( normal );
            normal.x = 0.5f * (normal.x + 1.0f);
            normal.y = 0.5f * (normal.y + 1.0f);
            normal.z = 0.5f * (normal.z + 1.0f);
            return normal;
#elif defined( DIFFUSE_SHADE )
            vector3 randBounce = _randomInUnitSphere();

            vector3 target;
            target.x = hit.point.x + hit.normal.x + randBounce.x;
            target.y = hit.point.y + hit.normal.y + randBounce.y;
            target.z = hit.point.z + hit.normal.z + randBounce.z;

            ray s;
            s.origin = hit.point;
            s.direction.x = target.x - hit.point.x;
            s.direction.y = target.y - hit.point.y;
            s.direction.z = target.z - hit.point.z;
            scattered = s;

            color.x = color.x * 0.5f;
            color.y = color.y * 0.5f;
            color.z = color.z * 0.5f;
#else
            if ( _materialScatter( scattered, hit.material, hit, &attenuation, &scattered ) ) {
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


static bool _sphereHit( ray& r, const uniform sphere_t* varying sphere, float t_min, float t_max, varying hit_info* uniform p_hit )
{
    assert( p_hit );
    
    //float center_x;
    //float center_y;
    //float center_z;
    //aos_to_soa3(&sphere->center, &center_x, &center_y, &center_z);

    vector3 oc;
    oc.x = r.origin.x - sphere->center[0];
    oc.y = r.origin.y - sphere->center[1];
    oc.z = r.origin.z - sphere->center[2];
    //oc.x = r.origin.x - center_x;
    //oc.y = r.origin.y - center_y;
    //oc.z = r.origin.z - center_z;

    float   a  = _dot( r.direction, r.direction );
    float   b  = _dot( oc, r.direction );
    float c = _dot( oc, oc ) - ( sphere->radius * sphere->radius );

    float discriminant = b * b - a * c;

    if ( discriminant > 0 ) {
        float t = ( -b - sqrt( discriminant ) ) / a;
        if ( t < t_max && t > t_min ) {
            p_hit->distance = t;
            p_hit->point    = _pointOnRay( r, t );
            p_hit->normal.x = ( p_hit->point.x - sphere->center[0] ) / sphere->radius;
            p_hit->normal.y = ( p_hit->point.y - sphere->center[1] ) / sphere->radius;
            p_hit->normal.z = ( p_hit->point.z - sphere->center[2] ) / sphere->radius;
            p_hit->material = sphere->material;
            return true;
        }

        t = ( -b + sqrt( discriminant ) ) / a;
        if ( t < t_max && t > t_min ) {
            p_hit->distance = t;
            p_hit->point    = _pointOnRay( r, t );
            p_hit->normal.x = ( p_hit->point.x - sphere->center[0] ) / sphere->radius;
            p_hit->normal.y = ( p_hit->point.y - sphere->center[1] ) / sphere->radius;
            p_hit->normal.z = ( p_hit->point.z - sphere->center[2] ) / sphere->radius;
            p_hit->material = sphere->material;
            return true;
        }
    }

    return false;
}


static bool  _sceneHit( ray& r, const uniform sphere_t* varying scene, unsigned int32 sceneSize, float t_min, float t_max, varying hit_info* uniform p_hit )
{
    bool     rval         = false;
    float    closestSoFar = t_max;
    hit_info hit;

    for ( int i = 0; i < sceneSize; i++ ) {
        const sphere_t* sphere = &scene[ i ];

        hit_info tmp;
        if ( _sphereHit( r, sphere, t_min, closestSoFar, &tmp ) ) {
            rval         = true;
            closestSoFar = tmp.distance;
            hit          = tmp;
        }
    }

    *p_hit = hit;
    return rval;
}


//
// Material implementations
//


static bool _materialScatter( ray& r, material_t& m, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered )
{
    bool rval = false;

    switch ( m.type ) {
        case MATERIAL_DIFFUSE:
            rval = _diffuseScatter( r, m, hit, p_attenuation, p_scattered );
            break;

        case MATERIAL_METAL:
            rval = _metalScatter( r, m, hit, p_attenuation, p_scattered );
            break;

        case MATERIAL_GLASS:
            rval = _glassScatter( r, m, hit, p_attenuation, p_scattered );
            break;

        default:
            //print("unknown material %d\n", m.type);
            assert( 0 );
    }

    return rval;
}


static bool _diffuseScatter( ray& r, material_t& d, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered )
{
    vector3 target    = hit.point + hit.normal;
    target = target + _randomInUnitSphere();
    vector3 hit_point = hit.point;

    ray scatter;
    scatter.origin = hit.point;
    scatter.direction = target.xyz - hit_point.xyz;

    *p_scattered     = scatter;
    p_attenuation->r = d.albedo[0];
    p_attenuation->g = d.albedo[1];
    p_attenuation->b = d.albedo[2];

    return true;
}


static bool _metalScatter( ray& r, material_t& m, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered )
{
    vector3 direction = _normalize( r.direction );
    vector3 reflected = _reflect( direction, hit.normal );

    ray scatter;
    scatter.origin = hit.point;
    scatter.direction = reflected.xyz + (m.blur * _randomInUnitSphere() );

    *p_scattered     = scatter;
    p_attenuation->r = m.albedo[0];
    p_attenuation->g = m.albedo[1];
    p_attenuation->b = m.albedo[2];

    float d = _dot( p_scattered->direction, hit.normal );
    return ( d > 0 );
}


static bool _glassScatter( ray& r, material_t& g, hit_info& hit, varying vector3 * uniform p_attenuation, varying ray* uniform p_scattered )
{
    vector3 outwardNormal;
    vector3 reflected = _reflect( r.direction, hit.normal );
    float   niOverNt;
    vector3 refracted;
    float   probability;
    float   cosine;

    // no color shift; "white" glass
    p_attenuation->r = 1.0;
    p_attenuation->g = 1.0;
    p_attenuation->b = 1.0;

    if ( _dot( r.direction, hit.normal ) > 0 ) {
        outwardNormal = -hit.normal;
        niOverNt      = g.refractionIndex;
        cosine        = g.refractionIndex * _dot(r.direction, hit.normal ) / _length( r.direction );
    } else {
        outwardNormal = hit.normal;
        niOverNt      = 1.0f / g.refractionIndex;
        cosine        = (-1 * _dot(r.direction, hit.normal ) ) / _length( r.direction );
    }

    if ( _refract( r.direction, outwardNormal, niOverNt, &refracted ) ) {
        probability = _schlick( cosine, g.refractionIndex );
    } else {
        probability = 1.0f;
    }

    float p = _random();

    if ( p < probability ) {
        ray scatter;
        scatter.origin = hit.point;
        scatter.direction = reflected;
        *p_scattered = scatter;
    } else {
        ray scatter;
        scatter.origin = hit.point;
        scatter.direction = refracted;
        *p_scattered = scatter;
    }

    return true;
}



//
// Camera functions
//

static ray _cameraGetRay( float s, float t )
{
    vector3 rand   = s_camera.lensRadius * _randomOnUnitDisk(); // calling _randomOnUnitDisk() cuts frame rate in half
    vector3 offset = s_camera.u * rand.x + s_camera.v * rand.y;

    vector3 origin    = { s_camera.origin.x + offset.x, s_camera.origin.y + offset.y, s_camera.origin.z + offset.z };

    vector3 direction;
    direction.x = s_camera.leftCorner.x + ( s * s_camera.horizontal.x ) + ( ( 1.0f - t ) * s_camera.vertical.x ) - s_camera.origin.x - offset.x;
    direction.y = s_camera.leftCorner.y + ( s * s_camera.horizontal.y ) + ( ( 1.0f - t ) * s_camera.vertical.y ) - s_camera.origin.y - offset.y;
    direction.z = s_camera.leftCorner.z + ( s * s_camera.horizontal.z ) + ( ( 1.0f - t ) * s_camera.vertical.z ) - s_camera.origin.z - offset.z;

    ray r = { origin, direction };

    return r;
}



//
// Helper functions
//

static double _dseed = 1.0;
static unsigned int32 _seed;

#ifdef ISPC_RANDOM

static inline void _randomInit( uniform int seed )
{
    varying int s = (seed << programIndex ) + programIndex;
    seed_rng( &_rngstate, s );

    _dseed = seed;
}

#else

static inline void _randomInit( varying unsigned int32 seed )
{
    _seed = seed;
}

#endif

static inline float drand48( float u, float v )
{
    double d     = 12.9898 * u + 78.233 * v;
    double x     = sin( d ) * 43758.5453;
    double fract = x - floor( x );
    double rval  = 2.0 * (fract)-1.0;
    rval         = ( 1.0 - abs( rval * _dseed ) );

    _dseed = rval;

    return rval;
}

static inline float xorrand( unsigned int32 seed )
{
    // Xorshift algorithm from George Marsaglia's paper
    seed ^= ( seed << 13 );
    seed ^= ( seed >> 17 );
    seed ^= ( seed << 5 );

    return seed;
}

static inline unsigned int32 wanghash( unsigned int32 seed )
{
    seed = ( seed ^ 61 ) ^ ( seed >> 16 );
    seed *= 9;
    seed = seed ^ ( seed >> 4 );
    seed *= 0x27d4eb2d;
    seed = seed ^ ( seed >> 15 );
    return seed;
}

#ifdef ISPC_RANDOM
static inline float _random()
{
    return frandom(&_rngstate);
}
#else
static inline float _random()
{
    //return drand48( _seed * programIndex, _seed * programIndex / programCount );

    _seed = wanghash( _seed );
    _seed = xorrand( _seed );

    // Generate a random float in [0, 1)...
    float f = ((float)_seed ) * ( 1.0 / 4294967296.0 );
    return f;
}
#endif

static inline float _length( vector3& v )
{
    float length = (float)sqrt( v.x * v.x + v.y * v.y + v.z * v.z );
    return length;
}


static inline vector3 _normalize( vector3& v )
{
    vector3 _v;
    _v.x = v.x;
    _v.y = v.y;
    _v.z = v.z;

    float length = (float)sqrt( _v.x * _v.x + _v.y * _v.y + _v.z * _v.z );
    _v.x /= length;
    _v.y /= length;
    _v.z /= length;

    return _v;
}


static inline float _dot( vector3& v1, vector3& v2 )
{
    return ( v1.x * v2.x ) + ( v1.y * v2.y ) + ( v1.z * v2.z );
}


static inline vector3 _cross( vector3 &v1, vector3 &v2 )
{
    vector3 v3 = {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
        };

    return v3;
}


static inline varying vector3 _pointOnRay( varying ray& r, varying float distance )
{
    vector3 p;
    p.x = r.origin.x + (r.direction.x * distance);
    p.y = r.origin.y + (r.direction.y * distance);
    p.z = r.origin.z + (r.direction.z * distance);

    return p;
}


//--------------------------------------------------------------

static inline uniform float _length( uniform vector3& v )
{
    uniform float length = (float)sqrt( v.x * v.x + v.y * v.y + v.z * v.z );

    return length;
}


static uniform inline vector3 _normalize( uniform vector3& v )
{
    uniform vector3 _v;
    _v.x = v.x;
    _v.y = v.y;
    _v.z = v.z;

    uniform float length = (float)sqrt( _v.x * _v.x + _v.y * _v.y + _v.z * _v.z );
    _v.x /= length;
    _v.y /= length;
    _v.z /= length;

    return _v;
}


static inline uniform float _dot( uniform vector3& v1, uniform vector3& v2 )
{
    return ( v1.x * v2.x ) + ( v1.y * v2.y ) + ( v1.z * v2.z );
}


static inline uniform vector3 _cross( uniform vector3 &v1, uniform vector3 &v2 )
{
    uniform vector3 v3 = {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
        };

    return v3;
}


static inline uniform vector3 _pointOnRay( uniform ray& r, uniform float distance )
{
    uniform vector3 p;
    p.x = r.origin.x + (r.direction.x * distance);
    p.y = r.origin.y + (r.direction.y * distance);
    p.z = r.origin.z + (r.direction.z * distance);

    return p;
}


//--------------------------------------------------------------

static inline vector3 _randomInUnitSphere()
{
    vector3 v;
    v.x = _random();
    v.y = _random();
    v.z = _random();

    return v;
}


static inline vector3 _randomOnUnitDisk()
{
    vector3      point;
    unsigned int maxTries = 20;
    do {
        vector3 v1 = { _random(), _random(), 0.0f };
        vector3 v2 = { 1.0f, 1.0f, 0.0f };
        point.x = 2.0f * v1.x - v2.x;
        point.y = 2.0f * v1.x - v2.y;
        point.z = 2.0f * v1.x - v2.z;
    } while ( _dot( point, point ) >= 1.0f && maxTries-- );

    return point;
}


static inline vector3 _reflect( vector3& v, vector3& normal )
{
    return v.xyz - 2 * _dot( v,  normal ) * normal.xyz;
}


static bool _refract( vector3& v, vector3& normal, float ni_over_nt, varying vector3* uniform p_refracted )
{
    vector3 _v           = _normalize( v );
    float   dt           = _dot( _v,  normal );
    float   discriminant = 1.0f - ni_over_nt * ni_over_nt * ( 1.0f - dt * dt );

    if ( discriminant > 0 ) {
        float sqrt_d = sqrt( discriminant );
        vector3 tmp;
        tmp.x = ni_over_nt * (_v.x - normal.x * dt) - normal.x * sqrt_d;
        tmp.y = ni_over_nt * (_v.y - normal.y * dt) - normal.y * sqrt_d;
        tmp.z = ni_over_nt * (_v.z - normal.z * dt) - normal.z * sqrt_d;
        *p_refracted = tmp;

        return true;
    } else {
        return false;
    }
}


static float _schlick( float cosine, float refractionIndex )
{
    float r0 = ( 1.0f - refractionIndex ) / ( 1.0f + refractionIndex );
    r0       = r0 * r0;

    return r0 + ( 1.0f - r0 ) * pow( ( 1.0f - cosine ), 5 );
}



//
// Test a simple call from C++ -> ISPC
//

static bool _testFunction( ray& r, material_t& g, hit_info& hit, vector3 attenuation, varying ray* uniform scattered )
{
    return false;
}


export void testISPC()
{
    print( "testISPC: % lanes\n", programCount );

    // Declare uniform to get a single instance per SIMD lane,
    // else you get the default (varying x8 on AVX).
    vector3 v1 = { 1.0f, 0.0f, 0.0f };
    vector3 v2 = { 1.0f, 0.0f, 0.5f };
    print("v1 = %, %, %\n", v1.x, v1.y, v1.z);
    print("v2 = %, %, %\n", v2.x, v2.y, v2.z);

    float a = _dot( _normalize(v1), _normalize(v2) );
    print( "_dot( [%, %, %], [%, %, %] ) = %\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, a );


    ray r;
    material_t material;
    hit_info hit;
    vector3 attenuation;
    ray scattered;

    bool rval = _testFunction( r, material, hit, attenuation, &scattered );

    print("_testFunction() = %\n", rval);
}


