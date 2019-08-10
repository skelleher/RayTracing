#pragma once

#include "result.h"
#include "vector_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <signal.h>


namespace pk
{

#define UNUSED( x ) ( x );

#if defined( SHIP_BUILD )
#define DEBUG_BREAK()
#elif defined( __arm__ )
#define SOFTWARE_INTERRUPT( signal )                                          \
    __asm__ __volatile__( "mov r0, %0\nmov r1, %1\nmov.w r12, #37\nswi 128\n" \
                          :                                                   \
                          : "r"( getpid() ), "r"( signal )                    \
                          : "r12", "r0", "r1", "cc" )

#ifndef SHIPBUILD
#define DEBUG_BREAK()                                                                       \
    {                                                                                       \
        do {                                                                                \
            printf( "- DEBUG_BREAK -" );                                                    \
            /***int trapSignal = Z::Platform::IsDebuggerAttached() ? SIGINT : SIGSTOP; ***/ \
            int trapSignal = SIGINT;                                                        \
            SOFTWARE_INTERRUPT( trapSignal );                                               \
            if ( trapSignal == SIGSTOP ) {                                                  \
                SOFTWARE_INTERRUPT( SIGINT );                                               \
            }                                                                               \
        } while ( false );                                                                  \
    }
#else
#define DEBUG_BREAK()
#endif

#elif defined( _WIN32 )
#define DEBUG_BREAK() \
    {                 \
        DebugBreak(); \
    }

#else
#define DEBUG_BREAK() assert( 0 )
#endif


#define DEBUGCHK( x )                                                        \
    {                                                                        \
        if ( !( x ) ) {                                                      \
            printf( "DEBUGCHK [%s:%d:%s]: ", __FILE__, __LINE__, __func__ ); \
            printf( "[%s]\n", #x );                                          \
            DEBUG_BREAK();                                                   \
        }                                                                    \
    }

#ifndef SHIP_BUILD
void check_cuda( cudaError_t result, char const* const function, const char* const filename, int const line );
#define CHECK_CUDA( x ) check_cuda( ( x ), #x, __FILE__, __LINE__ )
#else
#define CHECK_CUDA( x ) ( x )
#endif

#ifndef SHIP_BUILD
#define CHECK_VK( f )                                                                          \
    {                                                                                          \
        VkResult res = ( f );                                                                  \
        if ( res != VK_SUCCESS ) {                                                             \
            printf( "CHECK_VK [%s:%d:%s] VkResult: %d\n", __FILE__, __LINE__, __func__, res ); \
        }                                                                                      \
    }
#else
#define CHECK_VK( x ) ( x )
#endif

#define M_PI 3.14159265358979323846f
#define RADIANS( x ) ( (x)*M_PI / 180.0f )

#define ARRAY_SIZE( x ) ( sizeof( x ) / sizeof( ( x )[ 0 ] ) )

#ifndef CLAMP
#define CLAMP( x, min, max ) ( MIN( ( max ), MAX( ( x ), ( min ) ) ) )
#endif

#define STRINGIFY( x ) #x
#define XSTRINGIFY( s ) STRINGIFY( s )

bool delay( size_t ms );

// Give thread a sensible name in the debugger
result threadSetName( const char* name );
#define SET_THREAD_NAME()            \
    {                                \
        char name[ 32 ];             \
        sprintf_s( name, __func__ ); \
        threadSetName( name );       \
    }


__host__ __device__ float random();
__host__ __device__ vector3 randomInUnitSphere();
__host__ __device__ vector3 randomOnUnitDisk();

} // namespace pk
