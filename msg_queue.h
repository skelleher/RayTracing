#pragma once

#include "result.h"

#include <limits>
#include <stdint.h>


//
// A simple thread-safe message queue for sending messages from a producer thread to a consumer thread.
// Copy-on-send, copy-on-receive.
// Messages must be Plain-Old-Data types (structs, pointers, or primitive types).
// Does not support construction / destruction of object elements.
//

namespace pk
{

typedef uint32_t queue_t;
const queue_t    INVALID_QUEUE = ( queue_t )( -1 );

queue_t queue_create( size_t msg_size, uint32_t queue_length, void* queue_mem );
result  queue_destroy( queue_t queue );
result  queue_send( queue_t queue, const void* msg );
result  queue_send_blocking( queue_t queue, const void* msg, uint32_t timeout_ms = ( std::numeric_limits<uint32_t>::max )() );
result  queue_send_and_wait_for_response( queue_t queue, const void* msg );
result  queue_receive( queue_t queue, void* p_msg, size_t msg_size, unsigned int timeout_ms );
result  queue_notify_sender( queue_t queue, result rval );
result  queue_notify( queue_t queue );
result  queue_notify_all( queue_t queue );
size_t  queue_size( queue_t queue );

} // namespace pk
