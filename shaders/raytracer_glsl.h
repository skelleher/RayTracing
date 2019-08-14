//
// Uniform and input / output buffers for raytracer.comp
//

//
//  **** NOTE ****
//
//  Passing structs, and arrays of structs, from C++ to GLSL requires great care.
//  All fields should be 4-byte aligned.
//
//  Don't pass arrays of floats. They will NOT have the same stride in C++ and GLSL.
//  Use a vec4 or mat4 instead.
// 
//  Don't pass vec3 (unless it is aligned to 16-bytes).
//
//  Don't pass bool (unless it is aligned to 4 bytes).
//  sizeof(bool) = 4 on GLSL but may be 1 or 8 on the host.
//
//
// For the difference between GLSL layouts such as std140 and std430,
// see: https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
//


#ifdef GLSL

#define _ALIGN( x )
layout(row_major) uniform;
layout(row_major) buffer;

#else

#include "vector_cuda.h"
#include <stdint.h>
typedef uint32_t uint;
typedef vector3  vec3;
typedef vector4  vec4;
typedef uint     material_t;

#define _ALIGN( x ) alignas( x )

#endif

struct pixel {
    vec4 rgba;
};

struct ray {
    vec3 origin;
    vec3 direction;
};


//enum material_type_t
const uint MATERIAL_NONE    = 0;
const uint MATERIAL_DIFFUSE = 1;
const uint MATERIAL_METAL   = 2;
const uint MATERIAL_GLASS   = 3;

// Any GLSL struct used in an array must be 4-byte aligned
struct material_glsl_t {
    uint  type;
    float albedo_r;
    float albedo_g;
    float albedo_b;
    float blur;
    float refractionIndex;
};


struct hit_info_glsl_t {
    float distance;
    vec3  point;
    vec3  normal;
    uint  materialID;
};


// Any GLSL struct used in an array must be 4-byte aligned
struct sphere_glsl_t {
    _ALIGN(4 ) float center_x;
    _ALIGN(4 ) float center_y;
    _ALIGN(4 ) float center_z;
    _ALIGN(4 ) float radius;
    _ALIGN(4 ) uint  materialID;
};

// Nope, no vec3 at C++ / GLSL boundaries
struct camera_glsl_t {
    _ALIGN(16) vec3  origin;
    _ALIGN(16) vec3  lookat;
    _ALIGN(4 ) float vfov;
    _ALIGN(4 ) float aspect;
    _ALIGN(4 ) float aperture;
    _ALIGN(4 ) float focusDistance;
    _ALIGN(4 ) float lensRadius;

    _ALIGN(16) vec3  leftCorner;
    _ALIGN(16) vec3  horizontal;
    _ALIGN(16) vec3  vertical;
    _ALIGN(16) vec3  u;
    _ALIGN(16) vec3  v;
    _ALIGN(16) vec3  w;
};


//
// Shader inputs and outputs
// Do NOT use vec3 for Uniform or Storage buffers,
// when combined with std140
//
#ifndef GLSL
struct render_context_glsl_t {
#else
struct render_context_glsl_t {
#endif
    _ALIGN(4 ) uint outputHeight;
    _ALIGN(4 ) uint outputWidth;

    _ALIGN(4 ) float camera_vfov;
    _ALIGN(4 ) float camera_aspect;
    _ALIGN(4 ) float camera_aperture;
    _ALIGN(4 ) float camera_focusDistance;
    _ALIGN(16) vec3 camera_origin;
    _ALIGN(16) vec3 camera_lookat;

    _ALIGN(4 ) uint numSceneObjects;
    _ALIGN(4 ) uint num_aa_samples;
    _ALIGN(4 ) uint max_ray_depth;

    _ALIGN(4 ) bool applyGammaCorrection;
    _ALIGN(4 ) bool debug;
    _ALIGN(4 ) bool monochrome;

    // Check for mis-aligned fields; set to 0xDEADBEEF by host
    _ALIGN(4 ) uint magic;
};

#ifdef GLSL
layout( std140, binding = 0 ) uniform _ubo
{
    render_context_glsl_t ubo;
};
#endif

#ifdef GLSL
layout( std140, binding = 1 ) buffer _inputBuffer
{
    sphere_glsl_t scene[];
    //material_glsl_t        materials[];
};
#endif

#ifdef GLSL
layout( std140, binding = 2 ) buffer _outputBuffer
{
    pixel imageData[];
};
#endif
