#include "msg_queue.h"

#include "perf_timer.h"
#include "utils.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>


namespace pk
{

const int MAX_MSG_QUEUES = 5;

typedef struct _queue {
    std::mutex              mutex;
    std::condition_variable notification;
    std::atomic<bool>       notified;
    size_t                  msg_size;
    uint32_t                length;
    std::atomic<uint32_t>   used;
    uint8_t*                head;
    uint8_t*                tail;
    uint8_t*                msgs;

    // Some messages are blocking, and may return a result to the sender
    std::mutex              response_mutex;
    std::condition_variable response;
    result                  response_rval;

    _queue() :
        head( nullptr ),
        tail( nullptr ),
        notified( false ),
        msg_size( 0 ),
        length( 0 ),
        used( 0 ),
        msgs( nullptr ),
        response_rval( R_FAIL )
    {
    }
} _queue_t;

static std::mutex s_queues_mutex;
static _queue_t   s_queues[ MAX_MSG_QUEUES ];

static bool _valid( queue_t queue );
static bool _queue_is_empty( queue_t queue );
static bool _queue_is_full( queue_t queue );
static bool _queue_push_back( _queue_t* p_queue, const void* p_msg );
static void _queue_pop_front( _queue_t* p_queue, void* p_msg );


queue_t queue_create( size_t msg_size, uint32_t queue_length, void* queue_mem )
{
    assert( msg_size );
    assert( queue_length );
    assert( queue_mem );

    _queue_t* p_queue = nullptr;
    queue_t   handle  = INVALID_QUEUE;

    std::lock_guard<std::mutex> lock( s_queues_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_queues ); i++ ) {
        std::lock_guard<std::mutex> lock_queue( s_queues[ i ].mutex );

        if ( s_queues[ i ].msg_size == 0 ) {
            p_queue           = &s_queues[ i ];
            p_queue->msg_size = msg_size;
            p_queue->length   = queue_length;
            p_queue->used     = 0;
            p_queue->msgs     = (uint8_t*)queue_mem;

            p_queue->head = p_queue->tail = p_queue->msgs;

            handle = (queue_t)i;
            break;
        }
    }

    return handle;
}


result queue_destroy( queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    std::lock_guard<std::mutex> lock( s_queues_mutex );
    std::lock_guard<std::mutex> lock_queue( s_queues[ queue ].mutex );

    _queue_t* p_queue = &s_queues[ queue ];

    p_queue->head     = nullptr;
    p_queue->tail     = nullptr;
    p_queue->msg_size = 0;
    p_queue->length   = 0;
    p_queue->used     = 0;
    p_queue->msgs     = nullptr;

    return R_OK;
}


result queue_send( queue_t queue, const void* msg )
{
    if ( !_valid( queue ) || _queue_is_full( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    std::lock_guard<std::mutex> queue_lock( p_queue->mutex );
    _queue_push_back( p_queue, msg );
    p_queue->notified = true;
    p_queue->notification.notify_one();

    return R_OK;
}


result queue_send_blocking( queue_t queue, const void* msg, uint32_t timeout_ms )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    PerfTimer timer;
    result    rval = R_OK;

    p_queue->mutex.lock();
    while ( !_queue_push_back( p_queue, msg ) ) {
        // Block calling thread while queue is full; release lock so receiver can receive
        p_queue->mutex.unlock();
        delay( 10 );
        if ( timer.ElapsedMilliseconds() >= timeout_ms ) {
            rval = R_TIMEOUT;
            break;
        }

        static unsigned int timeout_warning_ms = 1000;
        if ( (unsigned int)timer.ElapsedMilliseconds() >= timeout_warning_ms ) {
            printf( "queue_send_blocking(%d) hung for %d seconds\n", queue, (unsigned)timer.ElapsedSeconds() );
            timeout_warning_ms *= 2;
        }

        p_queue->mutex.lock();
    };
    p_queue->notification.notify_one();
    p_queue->mutex.unlock();

    return rval;
}


result queue_send_and_wait_for_response( queue_t queue, const void* msg )
{
    if ( !_valid( queue ) || _queue_is_full( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    p_queue->mutex.lock();
    _queue_push_back( p_queue, msg );
    p_queue->notified = true;
    p_queue->notification.notify_one();
    p_queue->mutex.unlock();

    // Block on response
    std::unique_lock<std::mutex> response_lock( p_queue->response_mutex );
    uint32_t                     timeout_ms = 1000;
    std::chrono::milliseconds    ms( timeout_ms );
    std::cv_status               status = p_queue->response.wait_for( response_lock, ms );

    if ( status == std::cv_status::timeout ) {
        return R_TIMEOUT;
    }

    return p_queue->response_rval;
}


result queue_receive( queue_t queue, void* p_msg, size_t msg_size, unsigned int timeout_ms )
{
    assert( p_msg );
    assert( msg_size );

    if ( !_valid( queue ) ) {
        return R_FAIL;
    }

    result    rval    = R_FAIL;
    _queue_t* p_queue = &s_queues[ queue ];

    std::unique_lock<std::mutex> queue_lock( p_queue->mutex );

    std::chrono::milliseconds ms( timeout_ms );
    while ( _queue_is_empty( queue ) && !p_queue->notified ) {
        std::cv_status status = p_queue->notification.wait_for( queue_lock, ms );

        if ( status == std::cv_status::timeout ) {
            rval = R_TIMEOUT;
            goto Exit;
        }

        rval = R_OK;
        break;
    }

    if ( p_queue->notified || !_queue_is_empty( queue ) )
        rval = R_OK;

    _queue_pop_front( p_queue, p_msg );

Exit:
    p_queue->notified = false;

    return rval;
}


result queue_notify_sender( queue_t queue, result rval )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _queue_t*                   p_queue = &s_queues[ queue ];
    std::lock_guard<std::mutex> queue_lock( p_queue->response_mutex );
    p_queue->response_rval = rval;
    p_queue->response.notify_one();

    return R_OK;
}


result queue_notify( queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    std::lock_guard<std::mutex> queue_lock( p_queue->mutex );
    p_queue->notified = true;
    p_queue->notification.notify_one();

    return R_OK;
}


result queue_notify_all( queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    std::lock_guard<std::mutex> queue_lock( p_queue->mutex );
    p_queue->notification.notify_all();

    return R_OK;
}


size_t queue_size( queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _queue_t* p_queue = &s_queues[ queue ];

    std::lock_guard<std::mutex> queue_lock( p_queue->mutex );
    return p_queue->used;
}

//
// Private methods; these assume the queue mutex is already held
//

static bool _queue_push_back( _queue_t* p_queue, const void* p_msg )
{
    if ( p_queue->used >= p_queue->length ) {
        return false;
    }

    memcpy( p_queue->tail, p_msg, p_queue->msg_size ); // problem: need to call placement_new ctor()?
    p_queue->tail += p_queue->msg_size;
    p_queue->used++;

    //printf( "+%d\n", (uint32_t)p_queue->used );

    if ( p_queue->tail >= p_queue->msgs + ( p_queue->msg_size * p_queue->length ) )
        p_queue->tail = p_queue->msgs;

    return true;
}


static void _queue_pop_front( _queue_t* p_queue, void* p_msg )
{
    if ( p_queue->used == 0 )
        return;

    memcpy( p_msg, p_queue->head, p_queue->msg_size ); // problem: need to call dtor()?
    p_queue->head += p_queue->msg_size;
    p_queue->used--;

    //printf( "-%d\n", (uint32_t)p_queue->used );

    if ( p_queue->head >= p_queue->msgs + ( p_queue->msg_size * p_queue->length ) )
        p_queue->head = p_queue->msgs;
}


static bool _valid( queue_t queue )
{
    if ( queue == INVALID_QUEUE || queue >= ARRAY_SIZE( s_queues ) ) {
        return false;
    }

    return true;
}


static bool _queue_is_full( queue_t queue )
{
    return s_queues[ queue ].used >= s_queues[ queue ].length - 1;
}


static bool _queue_is_empty( queue_t queue )
{
    return s_queues[ queue ].used == 0;
}

} // namespace pk
