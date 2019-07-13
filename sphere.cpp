#include "sphere.h"

#include <assert.h>
#include <stdio.h>


namespace pk
{

__host__ __device__ Sphere::Sphere() :
    center( 0, 0, 0 ),
    radius( 1.0f ),
    material( nullptr )
{
}

__host__ __device__ Sphere::Sphere( const vector3& pos, float r, material_t* material ) :
    center( pos ),
    radius( r ),
    material( material )
{
}

__host__ __device__ bool Sphere::hit( const ray& r, float min, float max, hit_info* p_hit ) const
{
    assert( p_hit );

    vector3 oc = r.origin - center;
    float   a  = r.direction.dot( r.direction );
    float   b  = oc.dot( r.direction );
    float   c  = oc.dot( oc ) - ( radius * radius );

    float discriminant = b * b - a * c;

    if ( discriminant > 0 ) {
        float t = ( -b - sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - center ) / radius;
            p_hit->material = material;

            return true;
        }

        t = ( -b + sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - center ) / radius;
            p_hit->material = material;

            return true;
        }
    }

    return false;
}

} // namespace pk
