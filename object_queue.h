#pragma once

#include "perf_timer.h"
#include "result.h"
#include "spin_lock.h"
#include "utils.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>


//
// A simple thread-safe message queue for sending messages from a producer thread to a consumer thread.
// Copy-on-send, copy-on-receive.
//
// Messages can be objects, structs, or primitives.
//
// See also: msg_queue.h for a vanilla C version that only handles structs and primitives
//


namespace pk
{

typedef uint32_t  obj_queue_t;
const obj_queue_t INVALID_QUEUE = ( obj_queue_t )( -1 );

template<class TYPE>
class Queue {
public:
    static obj_queue_t create( uint32_t queue_length, TYPE* queue_mem );
    static result      destroy( obj_queue_t queue );
    static result      send( obj_queue_t queue, const TYPE* msg );
    static result      sendBlocking( obj_queue_t queue, const TYPE* msg, uint32_t timeout_ms = ( std::numeric_limits<uint32_t>::max )() );
    static result      sendAndWaitForResponse( obj_queue_t queue, const TYPE* msg );
    static result      receive( obj_queue_t queue, TYPE* p_msg, size_t msg_size, unsigned int timeout_ms = ( std::numeric_limits<unsigned int>::max )() );
    static result      notifySender( obj_queue_t queue, result rval );
    static result      notify( obj_queue_t queue );
    static result      notifyAll( obj_queue_t queue );
    static size_t      size( obj_queue_t queue );

private:
    typedef struct _obj_queue {
        SpinLock                spinLock; // used for fast lock during send/receive/query
        std::mutex              mutex;    // used for blocking thread on receive when queue is empty
        std::condition_variable notification;
        std::atomic<bool>       notified;
        size_t                  msg_size;
        uint32_t                length;
        std::atomic<uint32_t>   used;
        TYPE*                   head;
        TYPE*                   tail;
        TYPE*                   msgs;
        obj_queue_t             handle;

        // Some messages are blocking, and may return a result to the sender
        std::mutex              response_mutex;
        std::condition_variable response;
        result                  response_rval;

        _obj_queue() :
            head( nullptr ),
            tail( nullptr ),
            notified( false ),
            msg_size( 0 ),
            length( 0 ),
            used( 0 ),
            msgs( nullptr ),
            handle( INVALID_QUEUE ),
            response_rval( R_FAIL )
        {
        }
    } _obj_queue_t;

    static const int    MAX_OBJECT_QUEUES = 5;
    static std::mutex   s_queues_mutex;
    static _obj_queue_t s_queues[ MAX_OBJECT_QUEUES ];

    static bool _valid( obj_queue_t queue );
    static bool _queue_is_empty( obj_queue_t queue );
    static bool _queue_is_full( obj_queue_t queue );
    static bool _queue_push_back( _obj_queue_t* p_queue, const TYPE* p_msg );
    static bool _queue_pop_front( _obj_queue_t* p_queue, TYPE* p_msg );
};


//
// Implementation
//

template<class TYPE>
std::mutex Queue<TYPE>::s_queues_mutex;

template<class TYPE>
typename Queue<TYPE>::_obj_queue_t Queue<TYPE>::s_queues[ MAX_OBJECT_QUEUES ];


template<class TYPE>
obj_queue_t Queue<TYPE>::create( uint32_t queue_length, TYPE* queue_mem )
{
    assert( queue_length );
    assert( queue_mem );

    _obj_queue_t* p_queue = nullptr;
    obj_queue_t   handle  = INVALID_QUEUE;

    std::lock_guard<std::mutex> lock( s_queues_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_queues ); i++ ) {
        SpinLockGuard lock( s_queues[ i ].spinLock );

        if ( s_queues[ i ].msg_size == 0 ) {
            handle = (obj_queue_t)i;

            p_queue           = &s_queues[ i ];
            p_queue->msg_size = sizeof( TYPE );
            p_queue->length   = queue_length;
            p_queue->used     = 0;
            p_queue->msgs     = (TYPE*)queue_mem;
            p_queue->handle   = handle;
            p_queue->head = p_queue->tail = p_queue->msgs;

            break;
        }
    }

    return handle;
}


template<class TYPE>
result Queue<TYPE>::destroy( obj_queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];
    SpinLockGuard lock( p_queue->spinLock );

    p_queue->head     = nullptr;
    p_queue->tail     = nullptr;
    p_queue->msg_size = 0;
    p_queue->length   = 0;
    p_queue->used     = 0;
    p_queue->msgs     = nullptr;
    p_queue->handle   = INVALID_QUEUE;

    return R_OK;
}


template<class TYPE>
result Queue<TYPE>::send( obj_queue_t queue, const TYPE* msg )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];
    SpinLockGuard lock( p_queue->spinLock );

    if ( _queue_is_full( queue ) )
        return R_FAIL;

    bool rval = _queue_push_back( p_queue, msg );

    p_queue->notified = true;
    p_queue->notification.notify_one();

    return rval ? R_OK : R_FAIL;
}


template<class TYPE>
result Queue<TYPE>::sendBlocking( obj_queue_t queue, const TYPE* msg, uint32_t timeout_ms )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    PerfTimer timer;
    result    rval = R_OK;

    p_queue->spinLock.lock();
    while ( !_queue_push_back( p_queue, msg ) ) {
        // Block calling thread while queue is full; release lock so receiver can receive
        p_queue->spinLock.release();
        delay( 1 ); // Gross hack

        if ( timer.ElapsedMilliseconds() >= timeout_ms ) {
            rval = R_TIMEOUT;
            break;
        }

        static unsigned int timeout_warning_ms = 1000;
        if ( (unsigned int)timer.ElapsedMilliseconds() >= timeout_warning_ms ) {
            printf( "queue_sendBlocking(%d) hung for %d seconds\n", queue, (unsigned)timer.ElapsedSeconds() );
            timeout_warning_ms *= 2;
        }

        p_queue->spinLock.lock();
    };
    p_queue->notification.notify_one();
    p_queue->spinLock.release();

    return rval;
}


// TODO: we should also implement the blocking flavor of this
template<class TYPE>
result Queue<TYPE>::sendAndWaitForResponse( obj_queue_t queue, const TYPE* msg )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    p_queue->spinLock.lock();

    if ( _queue_is_full( queue ) ) {
        p_queue->spinLock.release();
        return R_FAIL;
    }

    _queue_push_back( p_queue, msg );
    p_queue->notified = true;
    p_queue->notification.notify_one();
    p_queue->spinLock.release();

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


template<class TYPE>
result Queue<TYPE>::receive( obj_queue_t queue, TYPE* p_msg, size_t msg_size, unsigned int timeout_ms )
{
    assert( p_msg );
    assert( msg_size && msg_size == sizeof( TYPE ) );

    if ( !_valid( queue ) ) {
        return R_FAIL;
    }

    result                    rval    = R_FAIL;
    _obj_queue_t*             p_queue = &s_queues[ queue ];
    std::chrono::milliseconds ms( timeout_ms );

    p_queue->spinLock.lock();

    while ( _queue_is_empty( queue ) && !p_queue->notified ) {
        p_queue->spinLock.release();
        std::unique_lock<std::mutex> queue_lock( p_queue->mutex );
        std::cv_status               status = p_queue->notification.wait_for( queue_lock, ms );
        p_queue->spinLock.lock();

        if ( status == std::cv_status::timeout ) {
            rval = R_TIMEOUT;
            goto Exit;
        }

        break;
    }

    if ( _queue_is_empty( queue ) ) {
        // Can happen if thread was notified (e.g. to terminate),
        // but no message was pushed to queue
        rval = R_FAIL;
        goto Exit;
    }

    if ( _queue_pop_front( p_queue, p_msg ) )
        rval = R_OK;

Exit:
    p_queue->notified = false;
    p_queue->spinLock.release();

    return rval;
}


template<class TYPE>
result Queue<TYPE>::notifySender( obj_queue_t queue, result rval )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    SpinLockGuard lock( p_queue->spinLock );
    p_queue->response_rval = rval;
    p_queue->response.notify_one();

    return R_OK;
}


template<class TYPE>
result Queue<TYPE>::notify( obj_queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    SpinLockGuard lock( p_queue->spinLock );
    p_queue->notified = true;
    p_queue->notification.notify_one();

    return R_OK;
}


template<class TYPE>
result Queue<TYPE>::notifyAll( obj_queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    SpinLockGuard lock( p_queue->spinLock );
    p_queue->notification.notify_all();

    return R_OK;
}


template<class TYPE>
size_t Queue<TYPE>::size( obj_queue_t queue )
{
    if ( !_valid( queue ) )
        return R_FAIL;

    _obj_queue_t* p_queue = &s_queues[ queue ];

    SpinLockGuard lock( p_queue->spinLock );
    return p_queue->used;
}


//
// Private methods; these assume the queue lock is already held
//

template<class TYPE>
bool Queue<TYPE>::_queue_push_back( _obj_queue_t* p_queue, const TYPE* p_msg )
{
    if ( p_queue->used >= p_queue->length )
        return false;

    *p_queue->tail = *p_msg;
    p_queue->tail++;
    p_queue->used++;

    if ( p_queue->tail >= &p_queue->msgs[ p_queue->length - 1 ] ) {
        p_queue->tail = p_queue->msgs;
    }

    return true;
}


template<class TYPE>
bool Queue<TYPE>::_queue_pop_front( _obj_queue_t* p_queue, TYPE* p_msg )
{
    if ( p_queue->used == 0 )
        false;

    *p_msg = *p_queue->head;
    p_queue->head++;
    p_queue->used--;

    if ( p_queue->head >= &p_queue->msgs[ p_queue->length - 1 ] ) {
        p_queue->head = p_queue->msgs;
    }

    return true;
}


template<class TYPE>
bool Queue<TYPE>::_valid( obj_queue_t queue )
{
    if ( queue == INVALID_QUEUE || queue >= ARRAY_SIZE( s_queues ) ) {
        return false;
    }

    return true;
}


template<class TYPE>
bool Queue<TYPE>::_queue_is_full( obj_queue_t queue )
{
    return s_queues[ queue ].used >= s_queues[ queue ].length;
}


template<class TYPE>
bool Queue<TYPE>::_queue_is_empty( obj_queue_t queue )
{
    return s_queues[ queue ].used == 0;
}


void testObjectQueues();

} // namespace pk
