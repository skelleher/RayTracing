//
// Trivial job system using thread pool
//

#include "thread_pool.h"

#include "object_queue.h"
#include "perf_timer.h"
#include "spin_lock.h"
#include "utils.h"

#include <assert.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pk
{

static const int MAX_THREAD_POOLS = 4;
static const int MAX_QUEUE_DEPTH  = 1024;


class Job {
public:
    Job() :
        pFunction( nullptr ),
        pContext( nullptr )
    {
    }

    virtual bool invoke( uint32_t tid )
    {
        if ( pFunction )
            return pFunction( pContext, tid );
        else {
            printf( "WARN: null Job.pFunction\n" );
            return false;
        }
    }

    std::function<bool( void*, uint32_t )> pFunction;
    void*                                  pContext;
    job_t                                  handle;
    job_group_t                            groupHandle;
};


typedef struct _thread {
    uint32_t          tid;
    thread_pool_t     hPool;
    std::thread*      thread;
    std::atomic<bool> shouldExit;

    // For perf debugging
    std::chrono::steady_clock::time_point startTick;
    std::chrono::steady_clock::time_point stopTick;
    uint64_t                              jobsExecuted;

    _thread() :
        tid( -1 ),
        hPool( INVALID_THREAD_POOL ),
        thread( nullptr ),
        shouldExit( false ),
        jobsExecuted( 0 )
    {
    }

    _thread( const _thread& rhs )
    {
        tid          = rhs.tid;
        hPool        = rhs.hPool;
        thread       = std::move( rhs.thread );
        shouldExit   = false;
        jobsExecuted = rhs.jobsExecuted;
    }
} _thread_t;


typedef struct _thread_pool {
    thread_pool_t          hPool;
    std::vector<_thread_t> threads;
    std::atomic<uint64_t>  nexthandle;
    Job*                   jobQueueBuffer;
    obj_queue_t            jobQueue;

    SpinLock                                        spinLock;
    std::unordered_map<job_t, std::atomic_bool>     jobCompletion;
    std::unordered_map<job_t, std::atomic_uint32_t> groupCompletion;

    _thread_pool() :
        hPool( INVALID_THREAD_POOL ),
        jobQueueBuffer( nullptr ),
        jobQueue( INVALID_QUEUE ),
        nexthandle( 0 ) {}
} _thread_pool_t;


static std::mutex     s_pools_mutex;
static _thread_pool_t s_pools[ MAX_THREAD_POOLS ];

static bool _valid( thread_pool_t pool );
static void _threadWorker( void* context );


//
// Public
//

thread_pool_t threadPoolInit( uint32_t numThreads )
{
    assert( numThreads );

    _thread_pool_t* tp     = nullptr;
    thread_pool_t   handle = INVALID_THREAD_POOL;

    std::lock_guard<std::mutex> lock( s_pools_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_pools ); i++ ) {
        SpinLockGuard( s_pools[ i ].spinLock );

        if ( s_pools[ i ].hPool == INVALID_THREAD_POOL ) {
            tp        = &s_pools[ i ];
            handle    = (thread_pool_t)i;
            tp->hPool = handle;
            break;
        }
    }

    if ( !tp )
        return INVALID_THREAD_POOL;

    tp->threads.reserve( numThreads );

    tp->jobQueueBuffer = new Job[ MAX_QUEUE_DEPTH ];
    tp->jobQueue       = Queue<Job>::create( MAX_QUEUE_DEPTH, tp->jobQueueBuffer );

    for ( uint32_t i = 0; i < numThreads; i++ ) {
        _thread_t t;
        t.hPool = handle;
        t.tid   = (uint32_t)i;

        tp->threads.push_back( t );
        tp->threads[ i ].thread = new std::thread( _threadWorker, (void*)&tp->threads[ i ] );
        assert( tp->threads[ i ].thread->joinable() );
    }

    //printf( "Created pool %d, %d threads\n", handle, numThreads );

    return handle;
}


// Submit a raw function to job system
job_t threadPoolSubmitJob( thread_pool_t pool, jobFunction function, void* context, thread_pool_blocking_t blocking )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    Job job;
    job.pContext    = context;
    job.pFunction   = std::bind( function, std::placeholders::_1, std::placeholders::_2 );
    job.handle      = (job_t)tp->nexthandle++;
    job.groupHandle = INVALID_JOB_GROUP;

    // NOTE: do NOT hold the spinlock when calling queue_send_blocking();
    // you'll block the worker threads and deadlock.
    tp->spinLock.lock();
    tp->jobCompletion[ job.handle ] = false;
    tp->spinLock.release();

    result rval = R_OK;
    if ( blocking == THREAD_POOL_SUBMIT_BLOCKING ) {
        rval = Queue<Job>::sendBlocking( tp->jobQueue, &job );
    } else {
        rval = Queue<Job>::send( tp->jobQueue, &job );
    }

    return job.handle;
}


// Submit an object method to job system
template<class TYPE>
job_t threadPoolSubmitJob( thread_pool_t pool, TYPE* object, bool ( TYPE::*method )( void*, uint32_t ), void* context, thread_pool_blocking_t blocking )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    Job job;
    job.pContext    = context;
    job.pFunction   = std::bind( method, object, std::placeholders::_1, std::placeholders::_2 );
    job.handle      = tp->nexthandle++;
    job.groupHandle = INVALID_JOB_GROUP;

    // NOTE: do NOT hold the spinlock when calling queue_send_blocking();
    // you'll block the worker threads and deadlock.
    tp->spinLock.lock();
    tp->jobCompletion[ job.handle ] = false;
    tp->spinLock.release();

    result rval = R_OK;
    if ( blocking == THREAD_POOL_SUBMIT_BLOCKING ) {
        rval = Queue<Job>::sendBlocking( tp->jobQueue, &job );
    } else {
        rval = Queue<Job>::send( tp->jobQueue, &job );
    }

    return job.handle;
}


job_group_t threadPoolSubmitJobs( thread_pool_t pool, jobFunction* functions, void** contexts, size_t numJobs, thread_pool_blocking_t blocking )
{
    return INVALID_JOB_GROUP;
}


template<class TYPE>
job_group_t threadPoolSubmitJobs( thread_pool_t pool, TYPE** objects, jobFunction* methods, void** contexts, size_t numJobs, thread_pool_blocking_t blocking )
{
    return INVALID_JOB_GROUP;
}


result threadPoolWaitForJob( thread_pool_t pool, job_t job, uint32_t timeout_ms )
{
    if ( !_valid( pool ) )
        return R_INVALID_ARG;

    _thread_pool_t* tp = &s_pools[ pool ];

    PerfTimer timer;

    while ( true ) {

        // TODO: should block on mutex and/or condition variable in case job is long-running

        tp->spinLock.lock();

        if ( tp->jobCompletion[ job ] ) {
            tp->jobCompletion.erase( job );
            tp->spinLock.release();
            return R_OK;
        }
        tp->spinLock.release();

        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) ); // gross
        if ( timer.ElapsedMilliseconds() >= timeout_ms )
            return R_TIMEOUT;
    }
}


result threadPoolWaitForJobs( thread_pool_t pool, job_group_t group, uint32_t timeout_ms )
{
    if ( !_valid( pool ) )
        return R_INVALID_ARG;

    _thread_pool_t* tp = &s_pools[ pool ];

    PerfTimer timer;

    while ( true ) {

        // TODO: should block on mutex and/or condition variable in case job is long-running

        tp->spinLock.lock();
        if ( tp->groupCompletion[ group ] ) {
            tp->groupCompletion.erase( group );
            tp->spinLock.release();
            return R_OK;
        }
        tp->spinLock.release();

        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) ); // gross
        if ( timer.ElapsedMilliseconds() >= timeout_ms )
            return R_TIMEOUT;
    }
}


bool threadPoolDeinit( thread_pool_t pool )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    tp->spinLock.lock();
    for ( int i = 0; i < tp->threads.size(); i++ ) {
        _thread_t& t = tp->threads[ i ];
        t.shouldExit = true;
    }
    tp->spinLock.release();

    Queue<Job>::notifyAll( tp->jobQueue );

    for ( int i = 0; i < tp->threads.size(); i++ ) {
        _thread_t& t = tp->threads[ i ];
        t.thread->join();
    }

    Queue<Job>::destroy( tp->jobQueue );
    delete[] tp->jobQueueBuffer;

    // Print perf metrics
    for ( int i = 0; i < tp->threads.size(); i++ ) {
        _thread_t& t = tp->threads[ i ];

        std::chrono::steady_clock::duration elapsedTicks = t.stopTick - t.startTick;
        auto                                duration     = std::chrono::duration_cast<std::chrono::seconds>( elapsedTicks ).count();
        double                              seconds      = std::chrono::duration<double>( duration ).count();

        printf( "Thread [%d:%d] %zd jobs %f seconds %f jobs/second\n", t.hPool, t.tid, t.jobsExecuted, seconds, t.jobsExecuted / seconds );
    }

    return true;
}


//
// Private implementation
//

static bool _valid( thread_pool_t pool )
{
    if ( pool == INVALID_THREAD_POOL || pool >= ARRAY_SIZE( s_pools ) ) {
        return false;
    }

    return true;
}


// Call the user-supplied function, passing a thread ID (informational) and the user-supplied function context
static void _threadWorker( void* context )
{
    SET_THREAD_NAME();

    _thread_t*      thread = (_thread_t*)context;
    _thread_pool_t* tp     = &s_pools[ thread->hPool ];

    thread->startTick = std::chrono::steady_clock::now();
    //printf( "_threadWorker[%d:%d] started\n", thread->pool, thread->tid );

    while ( true ) {
        if ( thread->shouldExit )
            goto Exit;

        Job job;
        if ( R_OK == Queue<Job>::receive( tp->jobQueue, &job, sizeof( job ), ( std::numeric_limits<unsigned int>::max )() ) ) {
            if ( thread->shouldExit )
                goto Exit;

            uint32_t tid = uint32_t( thread->hPool << 16 | thread->tid );
            job.invoke( tid );
            thread->jobsExecuted++;

            // Signal that job has completed
            SpinLockGuard lock( tp->spinLock );

            if ( job.handle != INVALID_JOB ) {
                tp->jobCompletion[ job.handle ] = true;
            }

            if ( job.groupHandle != INVALID_JOB_GROUP ) {
                tp->groupCompletion[ job.groupHandle ]++;
            }
        }
    }

Exit:
    thread->stopTick = std::chrono::steady_clock::now();
}


//
// Self-test
//

struct TestContext {
    int*         array1;
    int*         array2;
    unsigned int offset;
    unsigned int blockSize;
    job_t        handle;
};


class TestObject {
public:
    static bool static_method( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "_method1[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }


    bool method1( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method1[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }


    bool method2( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method2[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }

    bool method3( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method3[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }
};


bool _job( void* context, uint32_t tid )
{
    TestContext* ctx = (TestContext*)context;

    //printf( "_job[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

    for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
        ctx->array2[ i ] = ctx->array1[ i ] * 2;
    }

    return true;
}


void testThreadPool()
{
    std::cout << "test thread: " << std::this_thread::get_id() << std::endl;

    enum test_case_t : uint8_t {
        TEST_FUNCTION,
        TEST_STATIC_METHOD,
        TEST_METHOD_1,
        TEST_METHOD_2,
        TEST_METHOD_3,

        TEST_MAX
    };

    int        numThreads  = std::thread::hardware_concurrency() - 1;
    int        numElements = 1 << 20;
    int        blockSize   = 128;
    int        numBlocks   = numElements / blockSize;
    TestObject obj;

    thread_pool_t tp = threadPoolInit( numThreads );

    int* array1 = new int[ numElements ];
    int* array2 = new int[ numElements ];

    for ( int i = 0; i < ARRAY_SIZE( array1 ); i++ ) {
        array1[ i ] = i;
    }

    for ( int i = 0; i < ARRAY_SIZE( array2 ); i++ ) {
        array2[ i ] = -1;
    }

    TestContext* jobs = (TestContext*)new uint8_t[ sizeof( TestContext ) * numBlocks ];

    for ( uint8_t test = TEST_FUNCTION; test < TEST_MAX; test++ ) {
        printf( "[%d] Submitting %d jobs\n", test, numBlocks );

        PerfTimer timer;

        for ( int i = 0; i < numBlocks; i++ ) {
            jobs[ i ].array1    = array1;
            jobs[ i ].array2    = array2;
            jobs[ i ].offset    = i * blockSize;
            jobs[ i ].blockSize = blockSize;

            switch ( test ) {
                case TEST_FUNCTION:
                    jobs[ i ].handle = threadPoolSubmitJob( tp, _job, &jobs[ i ], THREAD_POOL_SUBMIT_BLOCKING );
                    break;

                case TEST_STATIC_METHOD:
                    jobs[ i ].handle = threadPoolSubmitJob( tp, TestObject::static_method, &jobs[ i ], THREAD_POOL_SUBMIT_BLOCKING );
                    break;

                case TEST_METHOD_1:
                    jobs[ i ].handle = threadPoolSubmitJob( tp, &obj, &TestObject::method1, &jobs[ i ], THREAD_POOL_SUBMIT_BLOCKING );
                    break;

                case TEST_METHOD_2:
                    jobs[ i ].handle = threadPoolSubmitJob( tp, &obj, &TestObject::method2, &jobs[ i ], THREAD_POOL_SUBMIT_BLOCKING );
                    break;

                case TEST_METHOD_3:
                    jobs[ i ].handle = threadPoolSubmitJob( tp, &obj, &TestObject::method3, &jobs[ i ], THREAD_POOL_SUBMIT_BLOCKING );
                    break;

                default:
                    assert( 0 );
                    break;
            }

            printf( "." );
        }

        printf( "\nSubmitted %d jobs in %f msec\n", numBlocks, timer.ElapsedMilliseconds() );

        printf( "[%d] Waiting for %d jobs\n", test, numBlocks );
        timer.Reset();
        for ( int i = 0; i < numBlocks; i++ ) {
            threadPoolWaitForJob( tp, jobs[ i ].handle, 5000 );
            printf( "." );
        }
        printf( " %f msec\n", timer.ElapsedMilliseconds() );

        int error = 0;
        for ( int i = 0; i < ARRAY_SIZE( array2 ); i++ ) {
            error += array2[ i ] - ( array1[ i ] * 2 );
        }
        assert( error == 0 );
    }

    threadPoolDeinit( tp );

    delete[] jobs;
    delete[] array1;
    delete[] array2;
}

} // namespace pk
