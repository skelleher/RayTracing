#pragma once

#include <condition_variable>
#include <mutex>


namespace pk
{

class Event final {
public:
    Event() :
        signalled( false )
    {
    }

    void set()
    {
        std::unique_lock<std::mutex> lock( mutex );
        signalled = true;
        notification.notify_one();
    }

    result wait( uint32_t timeoutMS = ( uint32_t )( -1 ) )
    {
        std::chrono::milliseconds    ms( timeoutMS );
        std::unique_lock<std::mutex> lock( mutex );
        while ( !signalled ) {
            std::cv_status status = notification.wait_for( lock, ms );
            if ( status == std::cv_status::timeout ) {
                signalled = false;
                return R_TIMEOUT;
            }
        }
        signalled = false;

        return R_OK;
    }

    void reset()
    {
        std::unique_lock<std::mutex> lock( mutex );
        signalled = false;
        notification.notify_one();
    }

private:
    std::mutex              mutex;
    std::condition_variable notification;
    bool                    signalled;
};

} // namespace pk
