#include "compute.h"
#include "utils.h"

#include <assert.h>


namespace pk
{

void testCompute( uint32_t preferredDevice, bool enableValidation )
{
    result    rval;
    compute_t instances[ 1 ] = {};

    // Test creation / destruction of multiple instances
    for ( int i = 0; i < ARRAY_SIZE( instances ); i++ ) {
        instances[ i ] = computeCreate( preferredDevice, enableValidation );
        assert(instances[i] != INVALID_COMPUTE_INSTANCE);
    }

    unsigned int i = (unsigned int)( random() * ARRAY_SIZE( instances ) ) % ARRAY_SIZE( instances );
    printf( "Using compute instance %d\n", i );
    compute_t cp = instances[ i ];

    // Create and submit a job
    const uint32_t COMPUTE_OUTPUT_WIDTH  = 3200;
    const uint32_t COMPUTE_OUTPUT_HEIGHT = 2400;
    const size_t OUTPUT_SIZE = COMPUTE_OUTPUT_WIDTH * COMPUTE_OUTPUT_HEIGHT * 4 * sizeof(float);
    uint8_t *output = new uint8_t[OUTPUT_SIZE];
    compute_job_desc_t job_desc =
    {
        "shaders\test_vulkan.spv",
        output,
        OUTPUT_SIZE,
    };

    compute_job_t job = computeCreateJob( job_desc, cp );
    assert( job != INVALID_COMPUTE_JOB );

    rval = computeSubmitJob( job );
    //assert( rval == R_OK );

    // Wait for job(s) to complete
    uint32_t timeoutMS = -1;
    rval = computeExecuteJobs( timeoutMS, cp );
    assert( rval == R_OK );

    rval = computeDestroyJob( job );
    //assert( rval == R_OK );

    for ( int i = 0; i < ARRAY_SIZE( instances ); i++ ) {
        rval = computeDestroy( instances[ i ] );
        assert( rval == R_OK );
    }
}

} // namespace pk
