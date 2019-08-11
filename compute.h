#pragma once

#include "result.h"
#include "spin_lock.h"
#include "thread_pool.h"

#include <atomic>
#include <memory>
#include <stdint.h>

namespace pk
{

typedef uint32_t compute_t;
#define INVALID_COMPUTE_INSTANCE ( compute_t )( -1 )
#define DEFAULT_COMPUTE_INSTANCE ( compute_t( 0 ) )

typedef uint32_t compute_job_t;
#define INVALID_COMPUTE_JOB ( compute_job_t )( -1 )

//
// Submit ComputeJobs to a ComputeInstance:
//

class IComputeJob;

result        computeInit( bool enableValidation = false );
compute_t     computeAcquire( uint32_t device = 0 );
result        computeRelease( compute_t instance );
uint32_t      computeGetMaxJobs( compute_t instance = DEFAULT_COMPUTE_INSTANCE );
compute_job_t computeSubmitJob( IComputeJob& job, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeWaitForJob( compute_job_t job, uint32_t timeoutMS = (uint32_t)-1, compute_t instance = DEFAULT_COMPUTE_INSTANCE );


//
// ComputeJobs must implement this interface
//

class IComputeJob {
public:
    virtual void init()                           = 0; // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit()                      = 0; // update share inputs / uniforms
    virtual void submit()                         = 0; // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ) = 0; // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job
    virtual void destroy()                        = 0; // clean up resources

    // Mandatory members used by ComputeInstance
    SpinLock      spinLock;
    compute_job_t handle;
    job_t         cpuThreadHandle;
    compute_t     hCompute;

protected:
    IComputeJob() = delete;
    IComputeJob( const IComputeJob& ) = delete;
    IComputeJob& operator=( const IComputeJob& ) = delete;

    // All instances of IComputeJob must call the base class ctor / dtor
    // to ensure the ComputeInstance is acquired / released
    IComputeJob( compute_t handle ) :
        handle( IComputeJob::nextHandle++ ),
        cpuThreadHandle( INVALID_JOB )
    {
        hCompute = handle;
        computeAcquire( hCompute );
    }

    virtual ~IComputeJob()
    {
        handle = INVALID_COMPUTE_JOB;
        computeRelease( hCompute );
    }

    static std::atomic<compute_job_t> nextHandle;
};


} // namespace pk
