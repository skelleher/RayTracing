//
// Uniform and buffer parameters to raytracer.comp
//

#ifndef GLSL
#include "vector_cuda.h"
#include <stdint.h>
typedef vector3  vec3;
typedef vector4  vec4;
typedef uint32_t uint;
typedef uint     material_t;
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


struct sphere_glsl_t {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    uint  materialID;
};


struct camera_glsl_t {
    vec3  origin;
    float vfov;
    float aspect;
    float aperture;
    vec3  lookat;
    float focusDistance;

    vec3  leftCorner;
    vec3  horizontal;
    vec3  vertical;
    vec3  u;
    vec3  v;
    vec3  w;
    float lensRadius;
};


struct render_context_glsl_t {
    uint outputHeight;
    uint outputWidth;
    float camera_origin[ 3 ];
    float camera_vfov;
    float camera_aspect;
    float camera_aperture;
    float camera_lookat[ 3 ];
    float camera_focusDistance;

    //uint vkFramebuffer;
    //    sphere_glsl_t          scene[];
    //    material_glsl_t        materials[];
    uint numSceneObjects;
    uint num_aa_samples;
    uint max_ray_depth;
    bool gammaCorrection;
    bool debug;
};
