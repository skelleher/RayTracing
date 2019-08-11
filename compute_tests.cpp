#include "compute.h"
#include "compute_job.h"
#include "utils.h"

#include <assert.h>
#include <vector>


namespace pk
{

void testCompute( uint32_t preferredDevice, bool enableValidation )
{
    result    rval;
    compute_t instances[ 2 ] = {};
    uint32_t  timeoutMS      = -1;

    rval = computeCreate( enableValidation );
    ASSERT( rval == R_OK );

    if ( preferredDevice > ARRAY_SIZE( instances ) - 1 )
        preferredDevice = 0;
    printf( "Using compute instance %d\n", preferredDevice );
    compute_t hCompute = computeAcquire( preferredDevice );
    ASSERT( hCompute != INVALID_COMPUTE_INSTANCE );

    // Create and submit jobs
    uint32_t maxJobs = 200; // computeGetMaxJobs(cp);
    std::vector<std::unique_ptr<ComputeJob>> jobs;
    jobs.resize( maxJobs );
    for ( unsigned i = 0; i < maxJobs; i++ ) {
        jobs[ i ] = ComputeJob::create( hCompute );
    }

    // Try to saturate the GPU
    uint32_t maxIterations = 30;
    for ( uint32_t iter = 0; iter < maxIterations; iter++ ) {
        printf( "Submitting %d jobs\n", maxJobs );

        for ( uint32_t i = 0; i < maxJobs; i++ ) {
            jobs[ i ]->enableGammaCorrection = i % 2;
            jobs[ i ]->maxIterations         = uint32_t( random() * 512 );
            jobs[ i ]->outputWidth           = 1000;
            jobs[ i ]->outputHeight          = 1000;

            //printf( "job[%d]: %d %d\n", i, jobs[ i ]->enableGammaCorrection, jobs[ i ]->maxIterations );

            jobs[ i ]->handle = computeSubmitJob( *jobs[ i ], hCompute );
            ASSERT( jobs[ i ]->handle != INVALID_COMPUTE_JOB );
        }

        // Wait for jobs to complete
        printf( "Waiting for jobs to complete...\n" );
        for ( unsigned i = 0; i < maxJobs; i++ ) {
            rval = computeWaitForJob( jobs[ i ]->handle, timeoutMS, hCompute );
            ASSERT( rval == R_OK );
        }
    }


    jobs[ 0 ]->save( "job1.ppm" );

    printf( "testCompute(): PASS\n" );

    //for ( int i = 0; i < maxJobs; i++ ) {
    //    jobs[i]->destroy();
    //}

    rval = computeRelease( hCompute );
    ASSERT( rval == R_OK );
}

} // namespace pk
