#include "log.h"
#include "utils.h"
#include "result.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#include <psapi.h> // must come after windows.h
#include <locale>
#include <codecvt>
#endif


namespace pk
{
static std::random_device                    _rd;
static std::mt19937                          _gen( _rd() );
static std::uniform_real_distribution<float> _random( 0.0f, 1.0f );

__device__ float xorrand( uint32_t seed );
__device__ uint32_t wanghash( uint32_t seed );
__device__ float drand48( const vector3& v );
__device__ float cudarand();
__device__ float cudarand2();


bool delay( size_t ms )
{
    std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );

    return true;
}


result threadSetName(const char* name)
{
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    //std::string narrow = converter.to_bytes(wide_utf16_source_string);
    std::wstring wchar_name = converter.from_bytes(name);

    SetThreadDescription(
        GetCurrentThread(),
        wchar_name.data()
    );

    return R_OK;
#else
    reutrn R_NOTIMPL;
#endif
}



__device__ __host__ float random()
{
#ifdef __CUDA_ARCH__
    return cudarand();
#else
    return _random( _gen );
#endif
}

__device__ __host__ vector3 randomInUnitSphere()
{
    vector3      point;
    unsigned int maxTries = 20;
    do {
        point = 2.0f * vector3( random(), random(), random() ) - vector3( 1, 1, 1 );
    } while ( point.squared_length() >= 1.0f && maxTries-- );

    return point;
}

__device__ __host__ vector3 randomOnUnitDisk()
{
    vector3      point;
    unsigned int maxTries = 20;
    do {
        point = 2.0f * vector3( random(), random(), 0.0f ) - vector3( 1.0f, 1.0f, 0.0f );
    } while ( point.dot( point ) >= 1.0f && maxTries-- );

    return point;
}


__device__ static double dseed = 1.0;
__device__ float         drand48( const vector3& v )
{
    double d     = 12.9898 * v.x + 78.233 * v.y;
    double x     = sin( d ) * 43758.5453;
    double fract = x - floor( x );
    double rval  = 2.0 * (fract)-1.0;
    rval         = ( 1.0 - abs( rval * dseed ) );

    dseed = rval;

    return rval;
}

__device__ float xorrand( uint32_t seed )
{
    // Xorshift algorithm from George Marsaglia's paper
    seed ^= ( seed << 13 );
    seed ^= ( seed >> 17 );
    seed ^= ( seed << 5 );

    return seed;
}

__device__ uint32_t wanghash( uint32_t seed )
{
    seed = ( seed ^ 61 ) ^ ( seed >> 16 );
    seed *= 9;
    seed = seed ^ ( seed >> 4 );
    seed *= 0x27d4eb2d;
    seed = seed ^ ( seed >> 15 );
    return seed;
}

__device__ float cudarand()
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t seed;
    seed = wanghash( x + y * clock() );
    seed = xorrand( seed );

    // Generate a random float in [0, 1)...
    float f = float( seed ) * ( 1.0 / 4294967296.0 );
    return f;
}

__device__ float cudarand2()
{
    curandState state;
    int         idx = threadIdx.x + ( blockIdx.x * blockDim.x );
    curand_init( (unsigned long long)clock() + idx, 0, 0, &state );

    return curand_uniform_double( &state );
}

void check_cuda( cudaError_t result, char const* const function, const char* const filename, int const line )
{
    if ( result ) {
        printf( "error 0x%x [%s:%d]: ", result, filename, line );
        printf( "[%s] ", function );
        printf( "%s : %s\n", cudaGetErrorName( result ), cudaGetErrorString( result ) );
        /*DEBUGCHK(0);*/
        //cudaDeviceReset();
    }
}

} // namespace pk
