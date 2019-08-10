#pragma once

#include <atomic>

namespace pk
{

//
// The simplest, laziest spin lock imaginable (thanks Stack Overflow)
//

class SpinLock final {
    std::atomic_flag _lock = ATOMIC_FLAG_INIT;

public:
    void lock()
    {
        while ( _lock.test_and_set( std::memory_order_acquire ) ) {
            ; // spin
        }
    }

    void release()
    {
        //assert(_lock);
        _lock.clear( std::memory_order_release );
    }

    SpinLock() = default;
    SpinLock(const SpinLock& rhs) = delete;
    SpinLock& operator=(const SpinLock& rhs) = delete;
};


class SpinLockGuard final {
public:
    SpinLockGuard( SpinLock& lock ) :
        m_lock( lock )
    {
        m_lock.lock();
    }

    ~SpinLockGuard()
    {
        m_lock.release();
    }

    SpinLockGuard(const SpinLockGuard& rhs) = delete;
    SpinLockGuard& operator=(const SpinLock& rhs) = delete;

private:
    SpinLock& m_lock;
};


} // namespace pk
