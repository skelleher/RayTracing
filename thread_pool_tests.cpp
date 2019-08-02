#include "perf_timer.h"
#include "thread_pool.h"
#include "utils.h"

#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <thread>


namespace pk
{

//
// Test thread pool: submit raw functions and object methods as job; jobs submitting other jobs, etc.
//

struct TestContext {
    int*         array1;
    int*         array2;
    unsigned int offset;
    unsigned int blockSize;
    job_t        handle;
};


class TestObject {
public:
    static bool static_method( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "_method1[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }


    bool method1( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method1[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }


    bool method2( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method2[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }

    bool method3( void* context, uint32_t tid )
    {
        TestContext* ctx = (TestContext*)context;

        //printf( "method3[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

        for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
            ctx->array2[ i ] = ctx->array1[ i ] * 2;
        }

        return true;
    }
};


bool _job( void* context, uint32_t tid )
{
    TestContext* ctx = (TestContext*)context;

    //printf( "_job[%d]: %d .. %d\n", tid, ctx->offset, ctx->offset + ctx->blockSize );

    for ( unsigned i = ctx->offset; i < ctx->offset + ctx->blockSize; i++ ) {
        ctx->array2[ i ] = ctx->array1[ i ] * 2;
    }

    return true;
}


void testThreadPool()
{
    std::cout << "test thread: " << std::this_thread::get_id() << std::endl;

    enum test_case_t : uint8_t {
        TEST_FUNCTION,
        TEST_STATIC_METHOD,
        TEST_METHOD_1,
        TEST_METHOD_2,
        TEST_METHOD_3,

        TEST_MAX
    };

    int        numThreads  = std::thread::hardware_concurrency() - 1;
    int        numElements = 1 << 20;
    int        blockSize   = 128;
    int        numBlocks   = numElements / blockSize;
    TestObject obj;

    thread_pool_t tp = threadPoolInit( numThreads );

    int* array1 = new int[ numElements ];
    int* array2 = new int[ numElements ];

    for ( int i = 0; i < ARRAY_SIZE( array1 ); i++ ) {
        array1[ i ] = i;
    }

    for ( int i = 0; i < ARRAY_SIZE( array2 ); i++ ) {
        array2[ i ] = -1;
    }

    TestContext* jobs = (TestContext*)new uint8_t[ sizeof( TestContext ) * numBlocks ];

    for ( uint8_t test = TEST_FUNCTION; test < TEST_MAX; test++ ) {
        printf( "[%d] Submitting %d jobs\n", test, numBlocks );

        PerfTimer timer;

        for ( int i = 0; i < numBlocks; i++ ) {
            jobs[ i ].array1    = array1;
            jobs[ i ].array2    = array2;
            jobs[ i ].offset    = i * blockSize;
            jobs[ i ].blockSize = blockSize;

            switch ( test ) {
                case TEST_FUNCTION:
                    jobs[ i ].handle = threadPoolSubmitJob( Function( _job, &jobs[ i ] ) );
                    break;

                case TEST_STATIC_METHOD:
                    jobs[ i ].handle = threadPoolSubmitJob( Function( TestObject::static_method, &jobs[ i ] ) );
                    break;

                case TEST_METHOD_1:
                    jobs[ i ].handle = threadPoolSubmitJob( Method( &obj, &TestObject::method1, &jobs[ i ]) );
                    break;

                case TEST_METHOD_2:
                    jobs[ i ].handle = threadPoolSubmitJob( Method( &obj, &TestObject::method2, &jobs[ i ]) );
                    break;

                case TEST_METHOD_3:
                    jobs[ i ].handle = threadPoolSubmitJob( Method( &obj, &TestObject::method3, &jobs[ i ]) );
                    break;

                default:
                    assert( 0 );
                    break;
            }

            printf( "." );
        }

        printf( "\nSubmitted %d jobs in %f msec\n", numBlocks, timer.ElapsedMilliseconds() );

        printf( "[%d] Waiting for %d jobs\n", test, numBlocks );
        timer.Reset();
        for ( int i = 0; i < numBlocks; i++ ) {
            threadPoolWaitForJob( jobs[ i ].handle, 5000, tp );
            printf( "." );
        }
        printf( " %f msec\n", timer.ElapsedMilliseconds() );

        int error = 0;
        for ( int i = 0; i < ARRAY_SIZE( array2 ); i++ ) {
            error += array2[ i ] - ( array1[ i ] * 2 );
        }
        assert( error == 0 );
    }

    threadPoolDeinit( tp );

    delete[] jobs;
    delete[] array1;
    delete[] array2;
}


} // namespace pk
