#pragma once

namespace pk
{

typedef enum {
    R_OK   = 0,
    R_FAIL = 1,

    R_INVALID_ARG       = 2,
    R_NOTIMPL           = 3,
    R_NULL_POINTER      = 4,
    R_INVALID_VERSION   = 5,
    R_TIMEOUT           = 6,
    R_QUEUE_FULL        = 7,
    R_INVALID_OPERATION = 8,
} result;


#define SUCCESS( x ) ( ar::R_OK == ( x ) ? true : false )
#define FAIL( x ) ( ar::R_OK != ( x ) ? true : false )

// Check boolean for failure
#define CBR( x )               \
    {                          \
        if ( ( x ) != true ) { \
            rval = R_FAIL;     \
            goto Exit;         \
        }                      \
    }

// Check pointer for NULL
#define CPR( x )                       \
    {                                  \
        if ( NULL == ( x ) ) {         \
            rval = ar::R_NULL_POINTER; \
            goto Exit;                 \
        }                              \
    }

// Check RESULT for failure
#define CHR( x )                         \
    {                                    \
        rval = x;                        \
        if ( SUCCESS( rval ) != true ) { \
            goto Exit;                   \
        }                                \
    }

} // END namespace pk
