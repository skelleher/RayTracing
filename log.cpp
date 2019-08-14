#ifdef _WIN32
#include <windows.h>
#endif

#define KEEP_PRINTF
#include "log.h"

#include "result.h"

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <time.h>


namespace pk
{

// Keep N copies of the log file so we can investigate crashes
const int NUM_LOGS_TO_BACKUP = 3;


//
// Static Data
//
std::recursive_mutex Log::s_mutex;
bool                 Log::s_initialized = false;
FILE *               Log::s_pFile       = NULL;
std::string          Log::s_filename( "" );
Log::ZoneMask        Log::s_zoneMask         = ( Log::ZoneMask )( ZONE_ERROR | ZONE_WARN | ZONE_INFO );
bool                 Log::s_enableTimestamps = false;


result Log::OpenWithFilename( const std::string &filename, bool timestamps )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    result rval = R_OK;

    s_enableTimestamps = timestamps;

    if ( filename.length() == 0 ) {
        Print( ZONE_ERROR, "Log:OpenWithFilename: empty filename\n" );
        return R_FAIL;
    }

    if ( s_initialized ) {
        Print( ZONE_WARN, "Log::OpenWithFilename(%s) called more than once; ignoring.\n", filename.c_str() );
        return R_FAIL;
    }


    BackupFileIfItExists( filename );

    rval = SetFilename( filename );

    s_initialized = true;

    return rval;
}


void Log::BackupFileIfItExists( const std::string &logfilename )
{
}


result Log::Close()
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    result rval = R_OK;

    {
        char      timestring[ 128 ];
        time_t    now = time( NULL );
        struct tm result;
        localtime_s( &result, &now );
        strftime( timestring, sizeof( timestring ), "%c", &result );

        Print( ZONE_INFO, "Log Closed %s\n", timestring );
    }

    fflush( s_pFile );
    fclose( s_pFile );

    s_pFile = NULL;

    return rval;
}


result Log::SetFilename( const std::string &logfilename )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    result rval = R_OK;

    if ( 0 == logfilename.size() || logfilename == "" ) {
        CHR( R_INVALID_ARG );
    }


    if ( s_filename.length() != 0 ) {
        Print( ZONE_INFO, "Log::SetFilename(%s): closing previous log [%s]\n", logfilename.c_str(), s_filename.c_str() );
        fflush( s_pFile );
        fclose( s_pFile );
    }

    fopen_s( &s_pFile, logfilename.c_str(), "wc" ); // 'c' = commit to disk and is a non-portable Windows extension to make fflush actually flush

    if ( NULL == s_pFile ) {
        Print( ZONE_INFO, "Error: Log: fail to open file [%s] errno %d\n", logfilename.c_str(), errno );
        assert( 0 );
    } else {
        s_filename = logfilename;
    }


    {
        char      timestring[ 128 ];
        time_t    now = time( NULL );
        struct tm result;
        localtime_s( &result, &now );
        strftime( timestring, sizeof( timestring ), "%c", &result );

        Print( ZONE_INFO, "Log [%s]\n", logfilename.c_str() );
        Print( ZONE_INFO, "Created %s\n", timestring );
    }

Exit:
    return rval;
}


const std::string &Log::Filename()
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );
    return s_filename;
}


void Log::Print( ZoneMask zone, const char *format, ... )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    if ( zone == ( s_zoneMask & zone ) ) {
        char    buf[ 1024 ];
        va_list vargs;
        va_start( vargs, format );
        vsprintf_s( buf, sizeof( buf ), format, vargs );

        if ( s_pFile ) {
            if ( s_enableTimestamps ) {
                char      timestring[ 128 ];

#ifdef _WIN32
                SYSTEMTIME systime;
                GetLocalTime(&systime);
                sprintf_s(timestring, sizeof(timestring), "%d.%d.%d - %2d:%02d:%02d.%04d",
                    systime.wDay, systime.wMonth, systime.wYear,
                    systime.wHour, systime.wMinute, systime.wSecond, systime.wMilliseconds
                    );

                fprintf( s_pFile, "[%s] %s", timestring, buf );
                printf( "[%s] %s", timestring, buf );
#else
                time_t    now = time( NULL );
                struct tm result;
                localtime_s( &result, &now );
                strftime( timestring, sizeof( timestring ), "%c", &result );
                fprintf( s_pFile, "[%s] %s", timestring, buf );
                printf( "[%s] %s", timestring, buf );
#endif

            } else {
                fprintf( s_pFile, "%s", buf );
                printf( "%s", buf );
            }

            fflush( s_pFile );
        }

        va_end( vargs );
    }
}


void Log::SetZoneMask( ZoneMask zoneMask )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    RETAIL( ZONE_VERBOSE, "Log::SetZoneMask( 0x%x )", zoneMask );
    s_zoneMask = zoneMask;
}


Log::ZoneMask Log::GetZoneMask()
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    return s_zoneMask;
}


bool Log::IsZoneEnabled( ZoneMask zone )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    return ( s_zoneMask & zone ) == zone ? true : false;
}


void Log::EnableZone( ZoneMask zone )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    s_zoneMask |= zone;
}


void Log::DisableZone( ZoneMask zone )
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    s_zoneMask &= ~zone;
}


void Log::EnableTimestamps(bool enable)
{
    std::unique_lock<std::recursive_mutex> lock( s_mutex );

    s_enableTimestamps = enable;
}


} // namespace pk
