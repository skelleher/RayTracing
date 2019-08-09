#pragma once

#include "result.h"

#include <atomic>
#include <stdint.h>

namespace pk
{

typedef uint32_t compute_job_t;
#define INVALID_COMPUTE_JOB ( compute_job_t )( -1 )

class IComputeJob {
public:
    virtual void create()                         = 0; // allocate resources: load shader; allocate buffers, bind descriptors
    virtual void presubmit()                      = 0; // update share inputs / uniforms
    virtual void submit()                         = 0; // submit command buffer to queue; DO NOT BLOCK in this function
    virtual void postsubmit( uint32_t timeoutMS ) = 0; // block until shader complete; do something with output, e.g. copy to CPU or pass to next compute job
    virtual void destroy()                        = 0; // clean up resources

    static std::atomic<compute_job_t> nextHandle;
};


typedef uint32_t compute_t;
#define INVALID_COMPUTE_INSTANCE ( compute_t )( -1 )
#define DEFAULT_COMPUTE_INSTANCE ( compute_t( 0 ) )

compute_t     computeCreate( uint32_t preferredDevice = 0, bool enableValidation = false );
compute_job_t computeSubmitJob( IComputeJob& job, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeExecuteJobs( uint32_t timeoutMS = (uint32_t)-1, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeWaitForJob( compute_job_t job, uint32_t timeoutMS = (uint32_t)-1, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeDestroy( compute_t instance = DEFAULT_COMPUTE_INSTANCE );

} // namespace pk
