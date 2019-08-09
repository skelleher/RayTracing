#include "compute.h"
#include "compute_job_vulkan.h"
#include "utils.h"

#include <assert.h>


namespace pk
{

void testCompute( uint32_t preferredDevice, bool enableValidation )
{
    result    rval;
    compute_t instances[ 2 ] = {};
    uint32_t  timeoutMS      = -1;

    // Test creation / destruction of multiple instances
    for ( int i = 0; i < ARRAY_SIZE( instances ); i++ ) {
        instances[ i ] = computeCreate( enableValidation );
        assert( instances[ i ] != INVALID_COMPUTE_INSTANCE );
    }

    if (preferredDevice > ARRAY_SIZE(instances) - 1)
        preferredDevice = 0;
    printf( "Using compute instance %d\n", preferredDevice );
    compute_t cp = instances[ preferredDevice ];

    // Create and submit jobs
    uint32_t maxJobs = computeGetMaxJobs(cp);
    ComputeJobVulkan* jobs = new ComputeJobVulkan[ maxJobs ];
    for ( uint32_t i = 0; i < maxJobs; i++ ) {
        jobs[ i ].enableGammaCorrection = i % 2;
        jobs[ i ].maxIterations         = uint32_t(random() * 512);
        jobs[i].outputWidth = 320;
        jobs[i].outputHeight = 240;

        //printf( "job[%d]: %d %d\n", i, jobs[ i ].enableGammaCorrection, jobs[ i ].maxIterations );

        jobs[ i ].handle = computeSubmitJob( jobs[ i ], cp );
        assert( jobs[ i ].handle != INVALID_COMPUTE_JOB );
    }

    
    // Wait for jobs to complete
    for ( int i = 0; i < ARRAY_SIZE( jobs ); i++ ) {
        rval = computeWaitForJob( jobs[ i ].handle, timeoutMS, cp );
        assert( rval == R_OK );
    }

    jobs[ 0 ].save( "job1.ppm" );

    printf( "testCompute(): PASS\n" );

    delete[] jobs;

    for ( int i = 0; i < ARRAY_SIZE( instances ); i++ ) {
        rval = computeDestroy( instances[ i ] );
        assert( rval == R_OK );
    }
}

} // namespace pk
