#include "compute.h"
#include "compute_job_vulkan.h"
#include "event_object.h"
#include "object_queue.h"
#include "perf_timer.h"
#include "spin_lock.h"
#include "thread_pool.h"
#include "utils.h"

#include <assert.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>


namespace pk
{

static const int MAX_COMPUTE_INSTANCES       = 2;         // Max active compute instances (i.e. number of GPUs)
static const int MAX_JOBS                    = 1024;      // Max active jobs per compute instance; strictly speaking, this is maxUniformBufferRange / average-size-of-uniforms-per-shader
static const int MAX_UNIFORM_BUFFERS_PER_JOB = 1;         // Assumes each compute shader has at most one uniform buffer
static const int MAX_STORAGE_BUFFERS_PER_JOB = 2;         // Assumes each compute shader has at most one storage buffer for input and one for output
static const int MAX_COMPUTE_JOB_TIMEOUT_MS  = 60 * 1000; // Don't allow compute job to execute for too long (NOT IMPLEMENTED YET)

std::atomic<compute_job_t> IComputeJob::nextHandle = 0;

struct ComputeInstance {
    SpinLock    spinLock;
    compute_t   handle;
    std::string deviceName;

    bool                     enableValidationLayers;
    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;

    VkDebugReportCallbackEXT debugReportCallback;
    VkInstance               instance;
    VkPhysicalDevice         physicalDevice;
    VkDevice                 device;
    VkQueue                  queue;
    uint32_t                 queueFamilyIndex;
    VkDescriptorPool         descriptorPool;
    VkCommandPool            commandPool;

    uint32_t                                             maxJobs;
    std::unordered_map<compute_job_t, ComputeJobVulkan*> activeJobs;
    std::unordered_map<compute_job_t, Event>             activeJobEvents;
    std::unordered_map<compute_job_t, ComputeJobVulkan*> finishedJobs;

    ComputeInstance() :
        handle( INVALID_COMPUTE_INSTANCE ),
        enableValidationLayers( false )
    {
    }
};

static std::mutex      s_compute_instances_mutex;
static ComputeInstance s_compute_instances[ MAX_COMPUTE_INSTANCES ];


static bool     _valid( compute_t pool );
static bool     _findPhysicalDevice( ComputeInstance& cp, uint32_t preferredDevice );
static uint32_t _findComputeQueueFamilyIndex( ComputeInstance& cp );
static bool     _initComputeInstance( ComputeInstance& cp, uint32_t preferredDevice, bool enableValidation );
static bool     _destroyComputeInstance( ComputeInstance& cp );
static bool     _createInstance( ComputeInstance& cp, bool enableValidation );
static bool     _enableValidationLayers( ComputeInstance& cp );
static bool     _createLogicalDevice( ComputeInstance& cp );
static bool     _createDescriptorPool( ComputeInstance& cp );
static bool     _createCommandPool( ComputeInstance& cp );
static bool     _executeComputeJob( void* context, uint32_t tid );
static void     _computeJobMarkFinished( ComputeJobVulkan* job );


//
//  Public
//

compute_t computeCreate( bool enableValidation, uint32_t preferredDevice )
{
    ComputeInstance* cp     = nullptr;
    compute_t        handle = INVALID_COMPUTE_INSTANCE;

    std::lock_guard<std::mutex> lock( s_compute_instances_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_compute_instances ); i++ ) {
        SpinLockGuard lock( s_compute_instances[ i ].spinLock );

        if ( s_compute_instances[ i ].handle == INVALID_COMPUTE_INSTANCE ) {
            cp         = &s_compute_instances[ i ];
            handle     = (compute_t)i;
            cp->handle = handle;
            break;
        }
    }

    if ( !cp ) {
        printf( "ERROR: Compute: max instances created\n" );
        return INVALID_COMPUTE_INSTANCE;
    }

    SpinLockGuard spinlock( cp->spinLock );

    if ( !_initComputeInstance( *cp, preferredDevice, enableValidation ) ) {
        printf( "ERROR: Compute[%d]: create FAIL\n", handle );
        s_compute_instances[ handle ].handle = INVALID_COMPUTE_INSTANCE;

        return INVALID_COMPUTE_INSTANCE;
    }

    printf( "\n" );

    return handle;
}


uint32_t computeGetMaxJobs( compute_t handle )
{
    if ( !_valid( handle ) )
        return R_FAIL;

    ComputeInstance& cp = s_compute_instances[ handle ];
    SpinLockGuard    compute_lock( cp.spinLock );

    return cp.maxJobs;
}


compute_job_t computeSubmitJob( IComputeJob& job, compute_t handle )
{
    if ( !_valid( handle ) )
        return R_FAIL;

    ComputeInstance& cp = s_compute_instances[ handle ];
    SpinLockGuard    compute_lock( cp.spinLock );

    ComputeJobVulkan* _job = dynamic_cast<ComputeJobVulkan*>( &job );
    SpinLockGuard     job_lock( _job->spinLock );

    // Bind the job to this compute instance
    _job->device         = cp.device;
    _job->physicalDevice = cp.physicalDevice;
    _job->descriptorPool = cp.descriptorPool;
    _job->commandPool    = cp.commandPool;
    _job->queue          = cp.queue;
    _job->instance       = cp.handle;

    _job->create();
    assert( _job->handle != INVALID_COMPUTE_JOB );

    // Remove job from the finished list (it is normal to allocate a job once and re-submit it frequently)
    cp.finishedJobs.erase( _job->handle );
    cp.activeJobs[ _job->handle ] = _job;
    cp.activeJobEvents[ _job->handle ].reset();

    _job->cpu_thread_handle = threadPoolSubmitJob( Function( _executeComputeJob, _job ) );
    if ( _job->cpu_thread_handle == INVALID_JOB ) {
        printf( "ERROR: Compute[%d]: submitJob failed\n", cp.handle );
        _job->handle = INVALID_COMPUTE_JOB;
    }

    return _job->handle;
}


result computeWaitForJob( compute_job_t job, uint32_t timeoutMS, compute_t handle )
{
    if ( !_valid( handle ) )
        return R_FAIL;

    ComputeInstance& cp = s_compute_instances[ handle ];

    cp.spinLock.lock();
    if ( cp.activeJobEvents.find( job ) == cp.activeJobEvents.end() ) {
        printf( "ERROR: Compute[%d]: waitForJob: handle %d is not owned by this instance\n", cp.handle, job );
        cp.spinLock.release();

        return R_FAIL;
    }
    cp.spinLock.release();

    while ( true ) {

        //cp.spinLock.lock();
        //if ( cp.finishedJobs.find( job ) != cp.finishedJobs.end() && cp.finishedJobs[ job ]->handle == job ) {
        //    cp.finishedJobs.erase( job );
        //    cp.activeJobs.erase( job );
        //    cp.activeJobEvents.erase( job );
        //    cp.spinLock.release();
        //    return R_OK;
        //}
        //cp.spinLock.release();

        result rval = cp.activeJobEvents[ job ].wait( timeoutMS );
        cp.spinLock.lock();
        cp.activeJobEvents.erase( job );
        cp.spinLock.release();

        return rval;
    }

    return R_OK;
}


result computeDestroy( compute_t handle )
{
    if ( !_valid( handle ) )
        return R_FAIL;

    ComputeInstance& cp = s_compute_instances[ handle ];
    SpinLockGuard    lock( cp.spinLock );

    if ( _destroyComputeInstance( cp ) ) {
        return R_OK;
    } else {
        return R_FAIL;
    }
}


//
// Private implementation; this will fork if we ever support Metal
//

static bool _valid( compute_t handle )
{
    if ( handle == INVALID_COMPUTE_INSTANCE || handle >= ARRAY_SIZE( s_compute_instances ) ) {
        return false;
    }

    return true;
}


static VKAPI_ATTR VkBool32 VKAPI_CALL _debugReportCallbackFn(
    VkDebugReportFlagsEXT      flags,
    VkDebugReportObjectTypeEXT objectType,
    uint64_t                   object,
    size_t                     location,
    int32_t                    messageCode,
    const char*                pLayerPrefix,
    const char*                pMessage,
    void*                      pUserData )
{
    printf( "[VK]: %s: %s\n", pLayerPrefix, pMessage );

    return VK_FALSE;
}


static bool _initComputeInstance( ComputeInstance& cp, uint32_t preferredDevice, bool enableValidation )
{
    _createInstance( cp, enableValidation );
    _findPhysicalDevice( cp, preferredDevice );
    _createLogicalDevice( cp );
    _createDescriptorPool( cp );
    _createCommandPool( cp );

    return true;
}


static bool _destroyComputeInstance( ComputeInstance& cp )
{
    printf( "Compute[%d]: destroying...\n", cp.handle );

    if ( cp.enableValidationLayers ) {
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr( cp.instance, "vkDestroyDebugReportCallbackEXT" );
        if ( func == nullptr ) {
            throw std::runtime_error( "Could not load vkDestroyDebugReportCallbackEXT" );
        }
        func( cp.instance, cp.debugReportCallback, nullptr );
    }

    vkDestroyDescriptorPool( cp.device, cp.descriptorPool, nullptr );
    vkDestroyCommandPool( cp.device, cp.commandPool, nullptr );
    vkDestroyDevice( cp.device, nullptr );
    vkDestroyInstance( cp.instance, nullptr );

    cp.handle = INVALID_COMPUTE_INSTANCE;

    return true;
}


static bool _createInstance( ComputeInstance& cp, bool enableValidation )
{
    if ( enableValidation )
        _enableValidationLayers( cp );

    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "com.partikle.demo";
    appInfo.applicationVersion = 0;
    appInfo.pEngineName        = "partikle";
    appInfo.engineVersion      = 0;
    appInfo.apiVersion         = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo    = {};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags                   = 0;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledLayerCount       = (uint32_t)cp.enabledLayers.size();
    createInfo.ppEnabledLayerNames     = cp.enabledLayers.data();
    createInfo.enabledExtensionCount   = (uint32_t)cp.enabledExtensions.size();
    createInfo.ppEnabledExtensionNames = cp.enabledExtensions.data();

    CHECK_VK( vkCreateInstance( &createInfo, nullptr, &cp.instance ) );

    printf( "Compute[%d]: created Vulkan instance\n", cp.handle );

    if ( cp.enableValidationLayers ) {
        VkDebugReportCallbackCreateInfoEXT createInfo = {};
        createInfo.sType                              = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        createInfo.flags                              = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        createInfo.pfnCallback                        = &_debugReportCallbackFn;

        auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr( cp.instance, "vkCreateDebugReportCallbackEXT" );
        if ( vkCreateDebugReportCallbackEXT == nullptr ) {
            printf( "ERROR: Compute: fail to GetProcAddress for debug callback\n" );
        }

        CHECK_VK( vkCreateDebugReportCallbackEXT( cp.instance, &createInfo, nullptr, &cp.debugReportCallback ) );
    }

    cp.maxJobs = MAX_JOBS;

    return true;
}


static bool _enableValidationLayers( ComputeInstance& cp )
{
    uint32_t numLayers = 0;
    vkEnumerateInstanceLayerProperties( &numLayers, nullptr );

    std::vector<VkLayerProperties> layerProperties( numLayers );
    vkEnumerateInstanceLayerProperties( &numLayers, layerProperties.data() );

    bool foundLayer = false;
    for ( VkLayerProperties prop : layerProperties ) {
        if ( strcmp( "VK_LAYER_LUNARG_standard_validation", prop.layerName ) == 0 ) {
            foundLayer = true;
            break;
        }
    }

    if ( !foundLayer ) {
        printf( "ERROR: failed to enable VK_LAYER_LUNARG_standard_validation\n" );
    }

    cp.enabledLayers.push_back( "VK_LAYER_LUNARG_standard_validation" );

    uint32_t numExtensions = 0;
    vkEnumerateInstanceExtensionProperties( nullptr, &numExtensions, nullptr );
    std::vector<VkExtensionProperties> extensionProperties( numExtensions );
    vkEnumerateInstanceExtensionProperties( nullptr, &numExtensions, extensionProperties.data() );

    bool foundExtension = false;
    for ( VkExtensionProperties prop : extensionProperties ) {
        if ( strcmp( VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName ) == 0 ) {
            foundExtension = true;
            break;
        }
    }

    if ( !foundExtension ) {
        printf( "ERROR: failed to enable VK_EXT_DEBUG_REPORT_EXTENSION_NAME\n" );
    }

    cp.enabledExtensions.push_back( VK_EXT_DEBUG_REPORT_EXTENSION_NAME );

    printf( "Compute[%d]: enabled validation layers\n", cp.handle );

    return foundLayer && foundExtension;
}


static bool _findPhysicalDevice( ComputeInstance& cp, uint32_t preferredDevice )
{
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices( cp.instance, &numDevices, nullptr );
    if ( numDevices == 0 ) {
        printf( "ERROR: Compute: No vulkan device found\n" );
        return false;
    }

    std::vector<VkPhysicalDevice> devices( numDevices );
    vkEnumeratePhysicalDevices( cp.instance, &numDevices, devices.data() );

    bool                       found          = false;
    uint32_t                   deviceIndex    = 0;
    uint32_t                   deviceID       = 0;
    VkPhysicalDevice           selectedDevice = nullptr;
    VkPhysicalDeviceProperties props          = {};

    for ( VkPhysicalDevice device : devices ) {
        vkGetPhysicalDeviceProperties( device, &props );

        // Select the preferred device, if specified
        if ( preferredDevice != -1 && deviceIndex == preferredDevice ) {
            cp.deviceName  = std::string( props.deviceName );
            deviceID       = props.deviceID;
            selectedDevice = device;
            found          = true;
            break;
        }
        // Else, map ComputeIndex[N] to device[n]
        if ( deviceIndex == cp.handle && !deviceID ) {
            cp.deviceName  = std::string( props.deviceName );
            deviceID       = props.deviceID;
            selectedDevice = device;
            found          = true;
            break;
        }
        deviceIndex++;
    }

    cp.physicalDevice = selectedDevice;

    if ( !found ) {
        printf( "ERROR: Compute[%d]: found no physical device\n", cp.handle );
        return R_FAIL;
    }

    printf( "Compute[%d]: using physical device %d [%s]\n", cp.handle, deviceID, cp.deviceName.c_str() );

    printf( "\tdeviceName = %s\n", props.deviceName );
    printf( "\tdeviceID = %d\n", props.deviceID );
    printf( "\tdeviceType = %d\n", props.deviceType );
    printf( "\tapiVersion = 0x%x\n", props.apiVersion );
    printf( "\tdriverVersion = 0x%x\n", props.driverVersion );
    printf( "\tvendorID = 0x%x\n", props.vendorID );

    printf( "\ttimestampComputeAndGraphics = %d\n", props.limits.timestampComputeAndGraphics );
    printf( "\tmaxFramebufferWidth = %d\n", props.limits.maxFramebufferWidth );
    printf( "\tmaxFramebufferHeight = %d\n", props.limits.maxFramebufferHeight );
    printf( "\tmaxComputeSharedMemorySize = %d\n", props.limits.maxComputeSharedMemorySize );
    printf( "\tmaxComputeWorkGroupCount = %d x %x x %d\n",
        props.limits.maxComputeWorkGroupCount[ 0 ],
        props.limits.maxComputeWorkGroupCount[ 1 ],
        props.limits.maxComputeWorkGroupCount[ 2 ] );
    printf( "\tmaxComputeWorkGroupSize = %d x %d x %d\n",
        props.limits.maxComputeWorkGroupSize[ 0 ],
        props.limits.maxComputeWorkGroupSize[ 1 ],
        props.limits.maxComputeWorkGroupSize[ 2 ] );
    printf( "\tmaxComputeWorkGroupInvocations = %d\n", props.limits.maxComputeWorkGroupInvocations );
    printf( "\tmaxUniformBufferRange = %d\n", props.limits.maxUniformBufferRange );
    printf( "\tmaxPushConstantsSize = %d\n", props.limits.maxPushConstantsSize );
    printf( "\tmaxStorageBufferRange = %d\n", props.limits.maxStorageBufferRange );
    printf( "\tmaxMemoryAllocationCount = %d\n", props.limits.maxMemoryAllocationCount );
    printf( "\tmaxBoundDescriptorSets = %d\n", props.limits.maxBoundDescriptorSets );
    printf( "\tmaxPerStageResources = %d\n", props.limits.maxPerStageResources );
    printf( "\tmaxPerStageDescriptorStorageBuffers = %d\n", props.limits.maxPerStageDescriptorStorageBuffers );
    printf( "\tmaxDescriptorSetStorageBuffers = %d\n", props.limits.maxDescriptorSetStorageBuffers );

    printf( "\tmaxStorageBufferRange = %d\n", props.limits.maxStorageBufferRange );
    printf( "\tmaxStorageBufferRange = %d\n", props.limits.maxStorageBufferRange );

    return R_OK;
}


static uint32_t _findComputeQueueFamilyIndex( ComputeInstance& cp )
{
    uint32_t numQueueFamilies;
    vkGetPhysicalDeviceQueueFamilyProperties( cp.physicalDevice, &numQueueFamilies, nullptr );
    std::vector<VkQueueFamilyProperties> queueFamilies( numQueueFamilies );
    vkGetPhysicalDeviceQueueFamilyProperties( cp.physicalDevice, &numQueueFamilies, queueFamilies.data() );

    //for ( uint32_t i = 0; i < numQueueFamilies; i++ ) {
    //    VkQueueFamilyProperties props = queueFamilies[ i ];
    //    printf( "\tQueue.queueCount = %d\n", props.queueCount );
    //    printf( "\tQueue.flags = 0x%x\n", props.queueFlags );
    //}

    uint32_t idx = ( uint32_t )( -1 );
    for ( uint32_t i = 0; i < numQueueFamilies; i++ ) {
        VkQueueFamilyProperties props = queueFamilies[ i ];

        if ( props.queueCount > 0 && ( props.queueFlags & VK_QUEUE_COMPUTE_BIT ) ) {
            idx = i;
            break;
        }
    }

    if ( idx == -1 ) {
        printf( "ERROR: Compute[%d]: no compute queue found\n", cp.handle );
    }

    return idx;
}


static bool _createLogicalDevice( ComputeInstance& cp )
{
    // Find the first command queue that supports compute shaders, and bind to a logical device
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex        = _findComputeQueueFamilyIndex( cp );
    queueCreateInfo.queueCount              = 1;
    float queuePriorities                   = 1.0f;
    queueCreateInfo.pQueuePriorities        = &queuePriorities;

    VkDeviceCreateInfo       deviceCreateInfo = {};
    VkPhysicalDeviceFeatures deviceFeatures   = {};
    deviceCreateInfo.sType                    = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos        = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount     = 1;
    deviceCreateInfo.pEnabledFeatures         = &deviceFeatures;

    CHECK_VK( vkCreateDevice( cp.physicalDevice, &deviceCreateInfo, nullptr, &cp.device ) );

    vkGetDeviceQueue( cp.device, cp.queueFamilyIndex, 0, &cp.queue );
    printf( "Compute[%d]: created logical device on queue %d\n", cp.handle, queueCreateInfo.queueFamilyIndex );

    return true;
}


static bool _createDescriptorPool( ComputeInstance& cp )
{
    VkDescriptorPoolSize uniformBufferPoolSize = {};
    uniformBufferPoolSize.type                 = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferPoolSize.descriptorCount      = MAX_JOBS * MAX_UNIFORM_BUFFERS_PER_JOB;

    VkDescriptorPoolSize storageBufferPoolSize = {};
    storageBufferPoolSize.type                 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageBufferPoolSize.descriptorCount      = MAX_JOBS * MAX_STORAGE_BUFFERS_PER_JOB;

    VkDescriptorPoolSize poolSizes[] = {
        uniformBufferPoolSize,
        storageBufferPoolSize
    };

    VkDescriptorPoolCreateInfo createDescriptorPoolInfo = {};
    createDescriptorPoolInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createDescriptorPoolInfo.maxSets                    = MAX_JOBS;
    createDescriptorPoolInfo.poolSizeCount              = ARRAY_SIZE( poolSizes );
    createDescriptorPoolInfo.pPoolSizes                 = poolSizes;
    createDescriptorPoolInfo.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    CHECK_VK( vkCreateDescriptorPool( cp.device, &createDescriptorPoolInfo, nullptr, &cp.descriptorPool ) );

    //printf( "Compute[%d]: created %d descriptor pools:\n", cp.handle, createDescriptorPoolInfo.poolSizeCount );
    for ( int i = 0; i < ARRAY_SIZE( poolSizes ); i++ ) {
        switch ( poolSizes[ i ].type ) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                printf( "Compute[%d]: Uniform pool: %d descriptors\n", cp.handle, poolSizes[ i ].descriptorCount );
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                printf( "Compute[%d]: Storage pool: %d descriptors\n", cp.handle, poolSizes[ i ].descriptorCount );
                break;
            default:
                assert( 0 );
        }
    }

    return true;
}


static bool _createCommandPool( ComputeInstance& cp )
{
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex        = cp.queueFamilyIndex;
    CHECK_VK( vkCreateCommandPool( cp.device, &commandPoolCreateInfo, nullptr, &cp.commandPool ) );

    printf( "Compute[%d]: created command pool\n", cp.handle );

    return true;
}


static bool _executeComputeJob( void* context, uint32_t tid )
{
    ComputeJobVulkan* job = (ComputeJobVulkan*)context;
    ComputeInstance&  cp  = s_compute_instances[ job->instance ];
    assert( job->handle != INVALID_COMPUTE_JOB );

    //printf( "Compute[%d]: execute job %d\n", job->instance, job->handle );

    job->presubmit();

    // It's safe to create resources on many threads, but vkQueueSubmit() must be synchronized:
    cp.spinLock.lock();
    job->submit();
    cp.spinLock.release();

    job->postsubmit( MAX_COMPUTE_JOB_TIMEOUT_MS );

    _computeJobMarkFinished( job );

    return true;
}


static void _computeJobMarkFinished( ComputeJobVulkan* job )
{
    assert( job->instance < ARRAY_SIZE( s_compute_instances ) );

    ComputeInstance& cp = s_compute_instances[ job->instance ];

    cp.spinLock.lock();
    cp.finishedJobs[ job->handle ] = job;
    cp.activeJobEvents[ job->handle ].set();
    cp.spinLock.release();
}

} // namespace pk
