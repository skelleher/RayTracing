//
// Trivial job system using thread pool
//

#include "thread_pool.h"

#include "event_object.h"
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

//
// Private types and data
//

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
    thread_pool_t     pool;
    std::thread*      thread;
    std::atomic<bool> shouldExit;

    // For perf debugging
    std::chrono::steady_clock::time_point startTick;
    std::chrono::steady_clock::time_point stopTick;
    uint64_t                              jobsExecuted;

    _thread() :
        tid( -1 ),
        pool( INVALID_THREAD_POOL ),
        thread( nullptr ),
        shouldExit( false ),
        jobsExecuted( 0 )
    {
    }

    _thread( const _thread& rhs )
    {
        tid          = rhs.tid;
        pool         = rhs.pool;
        thread       = std::move( rhs.thread );
        shouldExit   = false;
        jobsExecuted = rhs.jobsExecuted;
    }
} _thread_t;


typedef struct _thread_pool {
    thread_pool_t                handle;
    std::vector<_thread_t>       threads;
    std::vector<std::thread::id> threadIDs;
    std::atomic<uint64_t>        nexthandle;
    Job*                         jobQueueBuffer;
    obj_queue_t                  jobQueue;

    SpinLock spinLock;
    std::unordered_map<job_t, std::atomic_uint32_t> groupCompletion;
    std::unordered_map<job_t, Event>                activeJobEvents;

    _thread_pool() :
        handle( INVALID_THREAD_POOL ),
        jobQueueBuffer( nullptr ),
        jobQueue( INVALID_QUEUE ),
        nexthandle( 0 ) {}
} _thread_pool_t;


static std::mutex     s_pools_mutex;
static _thread_pool_t s_pools[ MAX_THREAD_POOLS ];

static bool _valid( thread_pool_t pool );
static void _threadWorker( void* context );
static bool _calledFromWorkerThread( thread_pool_t pool );


//
// Public
//

thread_pool_t threadPoolCreate( uint32_t numThreads )
{
    assert( numThreads );

    _thread_pool_t* tp     = nullptr;
    thread_pool_t   handle = INVALID_THREAD_POOL;

    std::lock_guard<std::mutex> lock( s_pools_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_pools ); i++ ) {
        SpinLockGuard lock( s_pools[ i ].spinLock );

        if ( s_pools[ i ].handle == INVALID_THREAD_POOL ) {
            tp         = &s_pools[ i ];
            handle     = (thread_pool_t)i;
            tp->handle = handle;
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
        t.pool = handle;
        t.tid  = (uint32_t)i;

        tp->threads.push_back( t );
        tp->threads[ i ].thread  = new std::thread( _threadWorker, (void*)&tp->threads[ i ] );
        std::thread::id threadID = tp->threads[ i ].thread->get_id();
        tp->threadIDs.push_back( threadID );
        assert( tp->threads[ i ].thread->joinable() );
    }

    //printf( "Created pool %d, %d threads\n", handle, numThreads );

    return handle;
}


job_t threadPoolSubmitJob( const Invokable& i, thread_pool_t pool, thread_pool_blocking_t blocking )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    Job job;
    job.pFunction   = i.functor;
    job.pContext    = i.context;
    job.handle      = (job_t)tp->nexthandle++;
    job.groupHandle = INVALID_JOB_GROUP;

    // NOTE: do NOT hold the spinlock when calling queue_send_blocking();
    // you'll block the worker threads and deadlock.
    tp->spinLock.lock();
    tp->activeJobEvents[ job.handle ].reset();
    tp->spinLock.release();

    result rval = R_OK;
    if ( blocking == THREAD_POOL_SUBMIT_BLOCKING ) {
        rval = Queue<Job>::sendBlocking( tp->jobQueue, &job );
    } else {
        rval = Queue<Job>::send( tp->jobQueue, &job );
    }

    return job.handle;
}


job_group_t threadPoolSubmitJobs( const Invokable* jobs, size_t numJobs, thread_pool_t pool, thread_pool_blocking_t blocking )
{
    return INVALID_JOB_GROUP;
}


result threadPoolWaitForJob( job_t job, uint32_t timeout_ms, thread_pool_t pool )
{
    if ( !_valid( pool ) )
        return R_INVALID_ARG;

    // Don't allow jobs to block on other jobs; all the worker threads can grind to a halt.
    if ( _calledFromWorkerThread( pool ) )
        return R_FAIL;

    _thread_pool_t* tp = &s_pools[ pool ];

    tp->spinLock.lock();
    if ( tp->activeJobEvents.find( job ) == tp->activeJobEvents.end() ) {
        printf( "ERROR: ThreadPool[%d]: waitForJob: handle %d is not owned by this pool\n", tp->handle, (uint32_t)job );
        tp->spinLock.release();

        return R_FAIL;
    }

    while ( true ) {

        //tp->spinLock.lock();
        //if ( tp->jobCompletion[ job ] ) {
        //    tp->activeJobEvents.erase( job );
        //    tp->spinLock.release();
        //    return R_OK;
        //}
        //tp->spinLock.release();

        result rval = tp->activeJobEvents[ job ].wait( timeout_ms );
        tp->spinLock.lock();
        tp->activeJobEvents.erase( job );
        tp->spinLock.release();

        return rval;
    }
}


result threadPoolWaitForJobs( job_group_t group, uint32_t timeout_ms, thread_pool_t pool )
{
    if ( !_valid( pool ) )
        return R_INVALID_ARG;

    _thread_pool_t* tp = &s_pools[ pool ];

    while ( true ) {

        //tp->spinLock.lock();
        //if ( tp->groupCompletion[ group ] ) {
        //    tp->groupCompletion.erase( group );
        //    tp->spinLock.release();
        //    return R_OK;
        //}
        //tp->spinLock.release();

        result rval = tp->activeJobEvents[ group ].wait( timeout_ms );
        tp->spinLock.lock();
        tp->activeJobEvents.erase( group );
        tp->spinLock.release();
    }
}


bool threadPoolDestroy( thread_pool_t pool )
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

        printf( "Thread [%d:%d] %zd jobs %f seconds %f jobs/second\n", t.pool, t.tid, t.jobsExecuted, seconds, t.jobsExecuted / seconds );
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


static bool _calledFromWorkerThread( thread_pool_t pool )
{
    std::thread::id id = std::this_thread::get_id();
    _thread_pool_t* tp = &s_pools[ pool ];

    if ( std::find( tp->threadIDs.begin(), tp->threadIDs.end(), id ) != tp->threadIDs.end() )
        return true;

    return false;
}


// Call the user-supplied function, passing a thread ID (informational) and the user-supplied function context
static void _threadWorker( void* context )
{
    SET_THREAD_NAME();

    _thread_t*      thread = (_thread_t*)context;
    _thread_pool_t* tp     = &s_pools[ thread->pool ];

    thread->startTick = std::chrono::steady_clock::now();
    //printf( "_threadWorker[%d:%d] started\n", thread->pool, thread->tid );

    while ( true ) {
        if ( thread->shouldExit )
            goto Exit;

        Job job;
        if ( R_OK == Queue<Job>::receive( tp->jobQueue, &job, sizeof( Job ), ( std::numeric_limits<unsigned int>::max )() ) ) {
            if ( thread->shouldExit )
                goto Exit;

            uint32_t tid = uint32_t( thread->pool << 16 | thread->tid );
            job.invoke( tid );
            thread->jobsExecuted++;

            // Signal that job has completed
            SpinLockGuard lock( tp->spinLock );

            if ( job.handle != INVALID_JOB ) {
                //tp->jobCompletion[ job.handle ] = true;
                tp->activeJobEvents[ job.handle ].set();
            }

            if ( job.groupHandle != INVALID_JOB_GROUP ) {
                tp->groupCompletion[ job.groupHandle ]++;
                // TODO: signal group completion event
                //tp->activeJobEvents[job.groupHandle].set();
            }
        }
    }

Exit:
    thread->stopTick = std::chrono::steady_clock::now();
}


} // namespace pk
