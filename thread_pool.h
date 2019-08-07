#pragma once

//
// Trivial job system using thread pools
//

#include "result.h"

#include <functional>
#include <stdint.h>

namespace pk
{

typedef uint32_t thread_pool_t;
typedef uint64_t job_t;
typedef uint64_t job_group_t;

#define INVALID_THREAD_POOL ( thread_pool_t( -1 ) )
#define DEFAULT_THREAD_POOL ( thread_pool_t( 0 ) )
#define INVALID_JOB ( job_t( -1 ) )
#define INVALID_JOB_GROUP ( job_group_t( -1 ) )
#define INFINITE_TIMEOUT ( uint32_t( -1 ) )

//
// All jobs, whether object methods or naked functions, must conform to this signature:
//
typedef bool ( *jobFunction )( void* context, uint32_t tid );


//
// Convenience wrappers for constructing jobs:
//
// threadPoolSubmitJob( Function( func, context ) );
// threadPoolSubmitJOb( Method( this, method, context ) );
//
class Invokable {
public:
    Invokable()
    {
        functor = nullptr;
    }

    bool invoke( uint32_t tid )
    {
        if ( functor )
            return functor( context, tid );
        else {
            printf( "WARN: null Job.functor\n" );
            return false;
        }
    }

    std::function<bool( void*, uint32_t )> functor;
    void*                                  context;
};


class Function : public Invokable {
public:
    Function( jobFunction function, void* ctx )
    {
        functor = std::bind( function, std::placeholders::_1, std::placeholders::_2 );
        context = ctx;
    }
};


template<class TYPE>
class _Method : public Invokable {
public:
    _Method( TYPE* object, bool ( TYPE::*method )( void*, uint32_t ), void* ctx )
    {
        functor = std::bind( method, object, std::placeholders::_1, std::placeholders::_2 );
        context = ctx;
    }
};

// C++11 can't infer template type from constructor parameters (C++17 can).
// So work around it with a little hack:
template<typename TYPE>
_Method<TYPE> Method( TYPE* object, bool ( TYPE::*method )( void*, uint32_t ), void* ctx )
{
    return _Method<TYPE>( object, method, ctx );
}


typedef enum {
    THREAD_POOL_SUBMIT_BLOCKING    = 0,
    THREAD_POOL_SUBMIT_NONBLOCKING = 1,
} thread_pool_blocking_t;


thread_pool_t threadPoolCreate( uint32_t numThreads );
job_t         threadPoolSubmitJob( const Invokable& job, thread_pool_t pool = DEFAULT_THREAD_POOL, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );
job_group_t   threadPoolSubmitJobs( const Invokable* jobs, size_t numJobs, thread_pool_t pool = DEFAULT_THREAD_POOL, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );
result        threadPoolWaitForJob( job_t, uint32_t timeout_ms = INFINITE_TIMEOUT, thread_pool_t pool = DEFAULT_THREAD_POOL );
result        threadPoolWaitForJobs( job_group_t, uint32_t timeout_ms = INFINITE_TIMEOUT, thread_pool_t pool = DEFAULT_THREAD_POOL );
bool          threadPoolDestroy( thread_pool_t pool );

void testThreadPool();

} // namespace pk
