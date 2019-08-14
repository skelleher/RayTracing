#pragma once

#include "result.h"

#include <mutex>
#include <stdint.h>
#include <string>

// Atomic option: remap printf() to Log in any file that includes this header
#ifdef __CUDA_ARCH__
#define KEEP_PRINTF
#endif

#ifndef KEEP_PRINTF
#define printf(...) RETAIL( ZONE_INFO, ##__VA_ARGS__ )
#endif

#define RETAIL( zone, format, ... ) \
    pk::Log::Print( zone, format, ##__VA_ARGS__ );


#ifdef _DEBUG
#define DBG( zone, format, ... ) \
    pk::Log::Print( zone, format, ##__VA_ARGS__ );
#else
#define DBG( zone, format, ... )
#endif

#define TRACE() DBG( ZONE_INFO, "%s", __FUNCTION__ )

#define ENUM_AS_STRING( enumValue ) \
    case enumValue:                 \
        return #enumValue;


namespace pk
{
class Log {
public:
    static const uint32_t ZONE_ERROR   = ( 1 << 0 );
    static const uint32_t ZONE_WARN    = ( 1 << 1 );
    static const uint32_t ZONE_INFO    = ( 1 << 2 );
    static const uint32_t ZONE_VERBOSE = ( 1 << 30 ); 
    typedef uint32_t      ZoneMask;

public:
    static result             OpenWithFilename( const std::string &filename, bool timestamps = false );
    static result             Close();
    static const std::string &Filename();

    static void     SetZoneMask( ZoneMask zoneMask );
    static ZoneMask GetZoneMask();
    static bool     IsZoneEnabled( ZoneMask zoneMask );
    static void     EnableZone( ZoneMask zoneMask );
    static void     DisableZone( ZoneMask zoneMask );
    static void     EnableTimestamps( bool enable );


    static void Print( ZoneMask zone, const char *format, ... );
    static void Print( const char *format, ... );

protected:
    // Singleton
    Log();
    Log( const Log &rhs );
    Log &operator=( const Log &rhs );
    virtual ~Log();

    static result SetFilename( const std::string &filename );
    static void   BackupFileIfItExists( const std::string &filename );

protected:
    static std::recursive_mutex s_mutex;
    static bool                 s_initialized;
    static std::string          s_filename;
    static FILE *               s_pFile;
    static ZoneMask             s_zoneMask;
    static bool                 s_enableTimestamps;
};
} // namespace pk

#define ZONE_ERROR pk::Log::ZONE_ERROR
#define ZONE_WARN pk::Log::ZONE_WARN
#define ZONE_INFO pk::Log::ZONE_INFO
#define ZONE_VERBOSE pk::Log::ZONE_VERBOSE
