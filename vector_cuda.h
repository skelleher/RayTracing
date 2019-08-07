#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

class vector3 {


public:
    __host__ __device__ vector3() {}
    __host__ __device__ vector3( float e0, float e1, float e2 )
    {
        x = e0;
        y = e1;
        z = e2;
    }

    __host__ __device__ ~vector3() {}

    __host__ __device__ inline float r() const { return x; }
    __host__ __device__ inline float g() const { return y; }
    __host__ __device__ inline float b() const { return z; }

    __host__ __device__ inline const vector3 &operator+() const { return *this; }
    __host__ __device__ inline vector3 operator-() const { return vector3( -x, -y, -z ); }

    __host__ __device__ inline vector3 &operator+=( const vector3 &v2 );
    __host__ __device__ inline vector3 &operator-=( const vector3 &v2 );
    __host__ __device__ inline vector3 &operator*=( const vector3 &v2 );
    __host__ __device__ inline vector3 &operator/=( const vector3 &v2 );
    __host__ __device__ inline vector3 &operator*=( const float t );
    __host__ __device__ inline vector3 &operator/=( const float t );

    __host__ __device__ inline float length() const { return sqrt( x * x + y * y + z * z ); }
    __host__ __device__ inline float squared_length() const { return x * x + y * y + z * z; }

    __host__ __device__ inline vector3 normalized() const
    {
        vector3 v = *this;
        v.normalize();

        return v;
    }

    __host__ __device__ inline void normalize()
    {
        float length = (float)sqrt( x * x + y * y + z * z );
        x /= length;
        y /= length;
        z /= length;
    }

    __host__ __device__ inline vector3 cross( const vector3& v ) const
    {
        return vector3( y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x );
    }

    __host__ __device__ inline float dot( const vector3& v ) const
    {
        return (x * v.x) + (y * v.y) + (z * v.z);
    }

    float x;
    float y;
    float z;
};


inline std::istream &operator>>( std::istream &is, vector3 &t )
{
    is >> t.x >> t.y >> t.z;
    return is;
}

inline std::ostream &operator<<( std::ostream &os, const vector3 &t )
{
    os << t.x << " " << t.y << " " << t.z;
    return os;
}

__host__ __device__ inline vector3 operator+( const vector3 &v1, const vector3 &v2 )
{
    return vector3( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z );
}

__host__ __device__ inline vector3 operator-( const vector3 &v1, const vector3 &v2 )
{
    return vector3( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z );
}

__host__ __device__ inline vector3 operator*( const vector3 &v1, const vector3 &v2 )
{
    return vector3( v1.x * v2.x, v1.y * v2.y, v1.z * v2.z );
}

__host__ __device__ inline vector3 operator/( const vector3 &v1, const vector3 &v2 )
{
    return vector3( v1.x / v2.x, v1.y / v2.y, v1.z / v2.z );
}

__host__ __device__ inline vector3 operator*( float t, const vector3 &v )
{
    return vector3( t * v.x, t * v.y, t * v.z );
}

__host__ __device__ inline vector3 operator/( vector3 v, float t )
{
    return vector3( v.x / t, v.y / t, v.z / t );
}

__host__ __device__ inline vector3 operator*( const vector3 &v, float t )
{
    return vector3( t * v.x, t * v.y, t * v.z );
}

__host__ __device__ inline float dot( const vector3 &v1, const vector3 &v2 )
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline vector3 cross( const vector3 &v1, const vector3 &v2 )
{
    return vector3( ( v1.y * v2.z - v1.z * v2.y ),
        ( -( v1.x * v2.z - v1.z * v2.x ) ),
        ( v1.x * v2.y - v1.y * v2.x ) );
}


__host__ __device__ inline vector3 &vector3::operator+=( const vector3 &v )
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

__host__ __device__ inline vector3 &vector3::operator*=( const vector3 &v )
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

__host__ __device__ inline vector3 &vector3::operator/=( const vector3 &v )
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

__host__ __device__ inline vector3 &vector3::operator-=( const vector3 &v )
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

__host__ __device__ inline vector3 &vector3::operator*=( const float t )
{
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

__host__ __device__ inline vector3 &vector3::operator/=( const float t )
{
    float k = 1.0f / t;

    x *= k;
    y *= k;
    z *= k;
    return *this;
}

__host__ __device__ inline vector3 unit_vector( vector3 v )
{
    return v / v.length();
}
