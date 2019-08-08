#pragma once

#include "result.h"

#include <stdint.h>

namespace pk
{

typedef uint32_t compute_t;
typedef uint32_t compute_job_t;
#define INVALID_COMPUTE_INSTANCE ( compute_t )( -1 )
#define DEFAULT_COMPUTE_INSTANCE ( compute_t( 0 ) )
#define INVALID_COMPUTE_JOB ( compute_job_t )( -1 )

// TODO: each job should have a function for:
// create (load shader; define, allocate, and bind buffers and descriptor sets)
// pre-submit (update uniforms)
// post-submit (do something with output, e.g. copy to CPU or pass to next compute job)
// destroy (clean up resources)
typedef struct
{
    const char* shaderPath;
    void* context;
    size_t contextSize;
} compute_job_desc_t;

compute_t     computeCreate( uint32_t preferredDevice = 0, bool enableValidation = false );
compute_job_t computeCreateJob( compute_job_desc_t& job, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeSubmitJob( compute_job_t job );
result        computeExecuteJobs( uint32_t timeoutMS = -1, compute_t instance = DEFAULT_COMPUTE_INSTANCE );
result        computeDestroyJob( compute_job_t job );
result        computeDestroy( compute_t instance = DEFAULT_COMPUTE_INSTANCE );

} // namespace pk
