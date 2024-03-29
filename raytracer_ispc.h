//
// raytracer_ispc.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#pragma once
#include <stdint.h>



#ifdef __cplusplus
namespace ispc { /* namespace */
#endif // __cplusplus
///////////////////////////////////////////////////////////////////////////
// Enumerator types with external visibility from ispc code
///////////////////////////////////////////////////////////////////////////

#ifndef __ISPC_ENUM_material_type_t__
#define __ISPC_ENUM_material_type_t__
enum material_type_t {
    MATERIAL_NONE = 0,
    MATERIAL_DIFFUSE = 1,
    MATERIAL_METAL = 2,
    MATERIAL_GLASS = 3 
};
#endif


#ifndef __ISPC_ALIGN__
#if defined(__clang__) || !defined(_MSC_VER)
// Clang, GCC, ICC
#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))
#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)
#else
// Visual Studio
#define __ISPC_ALIGN__(s) __declspec(align(s))
#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct
#endif
#endif

#ifndef __ISPC_STRUCT_RenderGangContext__
#define __ISPC_STRUCT_RenderGangContext__
struct RenderGangContext {
    float camera_origin[3];
    float camera_vfov;
    float camera_aspect;
    float camera_aperture;
    float camera_lookat[3];
    float camera_focusDistance;
    const struct sphere_t * scene;
    const struct material_t * materials;
    uint32_t sceneSize;
    uint32_t * framebuffer;
    uint32_t rows;
    uint32_t cols;
    uint32_t num_aa_samples;
    uint32_t max_ray_depth;
    uint32_t blockID;
    uint32_t blockSize;
    uint32_t totalBlocks;
    uint32_t xOffset;
    uint32_t yOffset;
    bool debug;
};
#endif

#ifndef __ISPC_STRUCT_sphere_t__
#define __ISPC_STRUCT_sphere_t__
struct sphere_t {
    float * center_x;
    float * center_y;
    float * center_z;
    float * radius;
    uint32_t * materialID;
};
#endif

#ifndef __ISPC_STRUCT_material_t__
#define __ISPC_STRUCT_material_t__
struct material_t {
    enum material_type_t * type;
    float * albedo_r;
    float * albedo_g;
    float * albedo_b;
    float * blur;
    float * refractionIndex;
};
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void cameraInitISPC(struct RenderGangContext * ctx);
    extern bool renderISPC(struct RenderGangContext * ctx);
    extern void testISPC();
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus
