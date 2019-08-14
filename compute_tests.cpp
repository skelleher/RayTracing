#include "compute.h"
#include "example_compute_job.h"
#include "log.h"
#include "mandelbrot_compute_job.h"
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

    //rval = computeInit( enableValidation );
    //ASSERT( rval == R_OK );

    if ( preferredDevice == -1 || preferredDevice > ARRAY_SIZE( instances ) - 1 )
        preferredDevice = 0;
    printf( "Using compute instance %d\n", preferredDevice );
    compute_t hCompute = computeAcquire( preferredDevice );
    ASSERT( hCompute != INVALID_COMPUTE_INSTANCE );

    //
    // Create and submit vanilla compute jobs
    //
    uint32_t maxJobs = 200;
    std::vector<std::unique_ptr<ExampleComputeJob>> jobs;
    jobs.resize( maxJobs );
    for ( unsigned i = 0; i < maxJobs; i++ ) {
        uint32_t inputWidth = 0;
        uint32_t inputHeight = 0;
        uint32_t outputWidth = 1000;
        uint32_t outputHeight = 1000;
        jobs[ i ] = ExampleComputeJob::create( hCompute, inputWidth, inputHeight, outputWidth, outputHeight );
    }

    for ( uint32_t i = 0; i < maxJobs; i++ ) {
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
    jobs[ 0 ]->save( "job1.ppm" );

    // HACK: Manually free resources so we don't run out of descriptors or command buffers for the next test
    for (unsigned i = 0; i < maxJobs; i++) {
        jobs[i].reset();
    }

    //
    // Create and submit custom compute jobs
    //
    std::vector<std::unique_ptr<MandelbrotComputeJob>> mandelbrotJobs;
    mandelbrotJobs.resize( maxJobs );
    for ( unsigned i = 0; i < maxJobs; i++ ) {
        uint32_t outputWidth = 1000;
        uint32_t outputHeight = 1000;
        mandelbrotJobs[ i ] = MandelbrotComputeJob::create( hCompute, outputWidth, outputHeight );
    }
    // Try to saturate the GPU
    uint32_t maxIterations = 30;
    for ( uint32_t iter = 0; iter < maxIterations; iter++ ) {
        printf( "Submitting %d jobs\n", maxJobs );

        for ( uint32_t i = 0; i < maxJobs; i++ ) {
            mandelbrotJobs[ i ]->enableGammaCorrection = i % 2;
            mandelbrotJobs[ i ]->maxIterations         = uint32_t( random() * 512 );

            //printf( "job[%d]: %d %d\n", i, jobs[ i ]->enableGammaCorrection, jobs[ i ]->maxIterations );

            mandelbrotJobs[ i ]->handle = computeSubmitJob( *mandelbrotJobs[ i ], hCompute );
            ASSERT( mandelbrotJobs[ i ]->handle != INVALID_COMPUTE_JOB );
        }

        // Wait for jobs to complete
        printf( "Waiting for jobs to complete...\n" );
        for ( unsigned i = 0; i < maxJobs; i++ ) {
            rval = computeWaitForJob( mandelbrotJobs[ i ]->handle, timeoutMS, hCompute );
            ASSERT( rval == R_OK );
        }
    }

    mandelbrotJobs[ 0 ]->save( "mandelbrot1.ppm" );

    printf( "testCompute(): PASS\n" );

    rval = computeRelease( hCompute );
    ASSERT( rval == R_OK );
}

} // namespace pk
