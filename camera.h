#pragma once

#include "log.h"
#include "ray.h"
#include "utils.h"
#include "vector_cuda.h"

#include <float.h> // for FLT_MAX
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

namespace pk
{

class Camera {
public:
    __host__ __device__ Camera() :
        Camera( 50.0f, 2.0f ) {}

    __host__ __device__ Camera( float vfov, float aspect, float aperture = 1.0f, float focusDistance = FLT_MAX, const vector3& pos = vector3( 0, 0, 0 ), const vector3& up = vector3( 0, 1, 0 ), const vector3& lookat = vector3( 0, 0, -1 ) )
    {
        this->vfov          = vfov;
        this->aspect        = aspect;
        this->aperture      = aperture;
        this->lookat        = lookat;
        this->focusDistance = focusDistance;

        lensRadius = aperture / 2.0f;

        float theta      = RADIANS( vfov );
        float halfHeight = (float)tan( theta / 2 );
        float halfWidth  = aspect * halfHeight;

        origin = pos;
        w      = ( pos - lookat ).normalized();
        u      = up.cross( w ).normalized();
        v      = w.cross( u );

        leftCorner = origin - halfWidth * focusDistance * u - halfHeight * focusDistance * v - focusDistance * w;
        horizontal = 2 * halfWidth * focusDistance * u;
        vertical   = 2 * halfHeight * focusDistance * v;

        //printf( "Camera(): \nfov %4.1f \naspect %4.1f \naperture %4.1f \n(%f, %f, %f) -> (%f, %f, %f) \n focusDistance %f\n",
        //    vfov, aspect, aperture, origin.x, origin.y, origin.z, lookat.x, lookat.y, lookat.z, focusDistance );

        //printf( "u[%f, %f, %f] \nv[%f, %f, %f] \nw[%f, %f, %f] \nhorizontal[%f, %f, %f] \nvertical[%f, %f, %f]\n",
        //    u.x, u.y, u.z,
        //    v.x, v.y, v.z,
        //    w.x, w.y, w.z,
        //    horizontal.x, horizontal.y, horizontal.z,
        //    vertical.x, vertical.y, vertical.z
        //);
    }

    __host__ __device__ Camera( const Camera& rhs ) :
        origin( rhs.origin ),
        leftCorner( rhs.leftCorner ),
        horizontal( rhs.horizontal ),
        vertical( rhs.vertical ),
        u( rhs.u ),
        v( rhs.v ),
        w( rhs.w ),
        lensRadius( rhs.lensRadius ),

        vfov( rhs.vfov ),
        aspect( rhs.aspect ),
        aperture( rhs.aperture ),
        lookat( rhs.lookat ),
        focusDistance( rhs.focusDistance )
    {
        printf( "Camera(): fov %4.1f aspect %4.1f aperture %4.1f (%f, %f, %f) -> (%f, %f, %f) (%f : %f)\n",
            vfov, aspect, aperture, origin.x, origin.y, origin.z, lookat.x, lookat.y, lookat.z, focusDistance, ( origin - lookat ).length() );
    }

    __host__ __device__ ray getRay( float s, float t ) const
    {
#ifdef __CUDA_ARCH__
        vector3 rand   = lensRadius * randomOnUnitDisk();
        vector3 offset = u * rand.x + v * rand.y;
        return ray( origin + offset, leftCorner + ( s * horizontal ) + ( ( 1.0f - t ) * vertical ) - origin - offset );
#else
        vector3 rand   = lensRadius * randomOnUnitDisk();
        vector3 offset = u * rand.x + v * rand.y;
        return ray( origin + offset, leftCorner + ( s * horizontal ) + ( ( 1.0f - t ) * vertical ) - origin - offset );
#endif
    }

    float   vfov;
    float   aspect;
    float   aperture;
    vector3 lookat;
    float   focusDistance;

    vector3 origin;
    vector3 leftCorner;
    vector3 horizontal;
    vector3 vertical;
    vector3 u;
    vector3 v;
    vector3 w;
    float   lensRadius;
};

} // namespace pk
