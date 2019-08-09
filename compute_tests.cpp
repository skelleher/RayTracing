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
        instances[ i ] = computeCreate( preferredDevice, enableValidation );
        assert( instances[ i ] != INVALID_COMPUTE_INSTANCE );
    }

    unsigned int i = (unsigned int)( random() * ARRAY_SIZE( instances ) ) % ARRAY_SIZE( instances );
    printf( "Using compute instance %d\n", i );
    compute_t cp = instances[ i ];

    // Create and submit jobs
    ComputeJobVulkan job1;
    job1.outputPath            = "job1.ppm";
    job1.enableGammaCorrection = true;

    ComputeJobVulkan job2;
    job2.outputPath            = "job2.ppm";
    job2.enableGammaCorrection = false;

    ComputeJobVulkan job3;
    job3.outputPath    = "job3.ppm";
    job3.maxIterations = 10;

    compute_job_t handle1 = computeSubmitJob( job1, cp );
    compute_job_t handle2 = computeSubmitJob( job2, cp );
    compute_job_t handle3 = computeSubmitJob( job3, cp );
    assert( handle1 != INVALID_COMPUTE_JOB );
    assert( handle2 != INVALID_COMPUTE_JOB );
    assert( handle3 != INVALID_COMPUTE_JOB );

    rval = computeExecuteJobs( timeoutMS, cp );
    assert( rval == R_OK );

    // Wait for jobs to complete
    rval = computeWaitForJob( handle1, timeoutMS, cp );
    assert( rval == R_OK );

    rval = computeWaitForJob( handle2, timeoutMS, cp );
    assert( rval == R_OK );

    rval = computeWaitForJob( handle3, timeoutMS, cp );
    assert( rval == R_OK );

    printf( "testCompute(): PASS\n" );

    for ( int i = 0; i < ARRAY_SIZE( instances ); i++ ) {
        rval = computeDestroy( instances[ i ] );
        assert( rval == R_OK );
    }
}

} // namespace pk
