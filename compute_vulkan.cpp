#include "compute.h"
#include "perf_timer.h"
#include "spin_lock.h"
#include "utils.h"

#include <assert.h>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>

namespace pk
{

static const int MAX_COMPUTE_INSTANCES = 2;

// TEST:
static const uint32_t WORKGROUP_SIZE             = 32;
static const uint32_t COMPUTE_OUTPUT_WIDTH       = 3200;
static const uint32_t COMPUTE_OUTPUT_HEIGHT      = 2400;
static const uint32_t COMPUTE_OUTPUT_DEPTH       = 1;
static const size_t   COMPUTE_OUTPUT_BUFFER_SIZE = COMPUTE_OUTPUT_WIDTH * COMPUTE_OUTPUT_HEIGHT * 4 * sizeof( float );
static const char*    DEFAULT_SHADER_PATH        = "shaders\\test_vulkan.spv";


//
// Based on https://github.com/Erkaman/vulkan_minimal_compute
//

// The MIT License (MIT)
//
// Copyright (c) 2017 Eric Arnebäck
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


struct _computeInstance {
    SpinLock spinLock;

    // Handle returned to user
    compute_t hCompute;

    bool                     enableValidationLayers;
    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;

    // In order to use Vulkan, you must create an instance.
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;

    // The physical device is some device on the system that supports usage of Vulkan.
    // Often, it is simply a graphics card that supports Vulkan.
    VkPhysicalDevice physicalDevice;

    // Then we have the logical device VkDevice, which basically allows
    // us to interact with the physical device.
    VkDevice device;

    // In order to execute commands on a device(GPU), the commands must be submitted
    // to a queue. The commands are stored in a command buffer, and this command buffer
    // is given to the queue.
    // There will be different kinds of queues on the device. Not all queues support
    // graphics operations, for instance. For this application, we at least want a queue
    // that supports compute operations.
    VkQueue queue;

    // Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
    // are grouped into queue families.
    //
    // When submitting a command buffer, you must specify to which queue in the family you are submitting to.
    // This variable keeps track of the index of that queue in its family.
    uint32_t queueFamilyIndex;

    // The command buffer is used to record commands, that will be submitted to a queue.
    // To allocate such command buffers, we use a command pool.
    VkCommandPool   commandPool;
    VkCommandBuffer commandBuffer;

    // *******************************************************************
    // TODO: fields below are shader-specific; move to _computeJob
    // *******************************************************************

    uint32_t workgroupWidth;
    uint32_t workgroupHeight;
    uint32_t workgroupDepth;

    // The compute shader output will be written to this buffer.
    // There is both a buffer handle and buffer backign store.
    VkBuffer       buffer;
    VkDeviceMemory bufferMemory;
    uint32_t       bufferSize;

    // Descriptors represent resources in shaders. They allow us to use things like
    // uniform buffers, storage buffers and images in GLSL.
    // A single descriptor represents a single resource, and several descriptors are organized
    // into descriptor sets, which are basically just collections of descriptors.
    VkDescriptorPool      descriptorPool;
    VkDescriptorSet       descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    // The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.
    // We will be creating a simple compute pipeline in this application.
    VkPipeline       pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule   computeShaderModule;

    // TODO: one per job
    VkFence fence;

    _computeInstance() :
        hCompute( INVALID_COMPUTE_INSTANCE ),
        enableValidationLayers( false ),
        workgroupWidth( (uint32_t)ceil( COMPUTE_OUTPUT_WIDTH / (float)WORKGROUP_SIZE ) ),
        workgroupHeight( (uint32_t)ceil( COMPUTE_OUTPUT_HEIGHT / (float)WORKGROUP_SIZE ) ),
        workgroupDepth( 1 )
    {
    }
};

static std::mutex       s_compute_instances_mutex;
static _computeInstance s_compute_instances[ MAX_COMPUTE_INSTANCES ];

static bool _valid( compute_t pool );
static bool _initComputeInstance( _computeInstance* instance, uint32_t preferredDevice, bool enableValidation );
static bool _destroyComputeInstance( _computeInstance* instance );

static bool     _findPhysicalDevice( _computeInstance* cp, uint32_t preferredDevice );
static uint32_t _findComputeQueueFamilyIndex( _computeInstance* cp );
static uint32_t _findMemoryType( _computeInstance* cp, uint32_t type, VkMemoryPropertyFlags properties );

static bool _createInstance( _computeInstance* cp, bool enableValidation );
static bool _enableValidationLayers( _computeInstance* cp );
static bool _createLogicalDevice( _computeInstance* cp );
static bool _createCommandBuffer( _computeInstance* cp );
static bool _executeJobs( _computeInstance* cp, uint32_t timeoutMS );

// TODO: these are specific to the job, not the device
static bool _createBuffer( size_t bufferSize, _computeInstance* cp );
static bool _createDescriptorSetLayout( _computeInstance* cp );
static bool _createDescriptorSet( _computeInstance* cp );
static bool _createComputePipeline( const std::string& shaderPath, _computeInstance* cp );

// Caller must delete[] the returned buffer
static uint32_t* _loadShader( const std::string& shaderPath, size_t* pShaderLength, _computeInstance* cp );


//
//  Public
//

compute_t computeCreate( uint32_t preferredDevice, bool enableValidation )
{
    _computeInstance* cp     = nullptr;
    compute_t         handle = INVALID_COMPUTE_INSTANCE;

    std::lock_guard<std::mutex> lock( s_compute_instances_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_compute_instances ); i++ ) {
        SpinLockGuard lock( s_compute_instances[ i ].spinLock );

        if ( s_compute_instances[ i ].hCompute == INVALID_COMPUTE_INSTANCE ) {
            cp           = &s_compute_instances[ i ];
            handle       = (compute_t)i;
            cp->hCompute = handle;
            break;
        }
    }

    if ( !cp ) {
        printf( "ERROR: Compute: max instances created\n" );
        return INVALID_COMPUTE_INSTANCE;
    }

    SpinLockGuard spinlock( cp->spinLock );
    if ( _initComputeInstance( cp, preferredDevice, enableValidation ) ) {
        printf( "Compute[%d]: created OK\n", handle );
        return handle;
    } else {
        printf( "ERROR: Compute[-1]: create FAIL\n" );
        return INVALID_COMPUTE_INSTANCE;
    }
}


compute_job_t computeCreateJob( compute_job_desc_t& jobDesc, compute_t instance )
{
    // TODO: create shader pipeline and command buffer
    return R_NOTIMPL;
}


result computeSubmitJob( compute_job_t job )
{
    // TODO: submit a command buffer to the compute queue
    return R_NOTIMPL;
}


result computeExecuteJobs( uint32_t timeoutMS, compute_t handle )
{
    if ( !_valid( handle ) )
        return R_FAIL;

    _computeInstance* cp = &s_compute_instances[ handle ];
    SpinLockGuard     lock( cp->spinLock );

    if ( _executeJobs( cp, timeoutMS ) )
        return R_OK;
    else
        return R_FAIL;
}


result computeDestroyJob( compute_job_t job )
{
    return R_NOTIMPL;
}


result computeDestroy( compute_t instance )
{
    if ( !_valid( instance ) )
        return R_FAIL;

    _computeInstance* cp = &s_compute_instances[ instance ];

    if ( _destroyComputeInstance( cp ) ) {
        return R_OK;
    } else {
        return R_FAIL;
    }
}


//
// Private implementation
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
    printf( "Vulkan Debug Report: %s: %s\n", pLayerPrefix, pMessage );

    return VK_FALSE;
}


static bool _initComputeInstance( _computeInstance* cp, uint32_t preferredDevice, bool enableValidation )
{
    _createInstance( cp, enableValidation );
    _findPhysicalDevice( cp, preferredDevice );
    _createLogicalDevice( cp );

    // TODO: these are specific to the job, not the device
    _createBuffer( COMPUTE_OUTPUT_BUFFER_SIZE, cp );
    _createDescriptorSetLayout( cp );
    _createDescriptorSet( cp );
    _createComputePipeline( DEFAULT_SHADER_PATH, cp );
    _createCommandBuffer( cp );

    return true;
}


static bool _destroyComputeInstance( _computeInstance* cp )
{
    SpinLockGuard lock( cp->spinLock );

    if ( cp->enableValidationLayers ) {
        // destroy callback.
        auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr( cp->instance, "vkDestroyDebugReportCallbackEXT" );
        if ( func == nullptr ) {
            throw std::runtime_error( "Could not load vkDestroyDebugReportCallbackEXT" );
        }
        func( cp->instance, cp->debugReportCallback, nullptr );
    }

    vkFreeMemory( cp->device, cp->bufferMemory, nullptr );
    vkDestroyBuffer( cp->device, cp->buffer, nullptr );
    vkDestroyShaderModule( cp->device, cp->computeShaderModule, nullptr );
    vkDestroyDescriptorPool( cp->device, cp->descriptorPool, nullptr );
    vkDestroyDescriptorSetLayout( cp->device, cp->descriptorSetLayout, nullptr );
    vkDestroyPipelineLayout( cp->device, cp->pipelineLayout, nullptr );
    vkDestroyPipeline( cp->device, cp->pipeline, nullptr );
    vkDestroyCommandPool( cp->device, cp->commandPool, nullptr );
    vkDestroyFence( cp->device, cp->fence, nullptr );
    vkDestroyDevice( cp->device, nullptr );
    vkDestroyInstance( cp->instance, nullptr );

    printf( "Compute[%d] destroyed\n", cp->hCompute );
    cp->hCompute = INVALID_COMPUTE_INSTANCE;

    return true;
}


static bool _createInstance( _computeInstance* cp, bool enableValidation )
{
    if ( enableValidation )
        _enableValidationLayers( cp );

    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "Ray Tracer";
    appInfo.applicationVersion = 0;
    appInfo.pEngineName        = "partikle";
    appInfo.engineVersion      = 0;
    appInfo.apiVersion         = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo    = {};
    createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags                   = 0;
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledLayerCount       = (uint32_t)cp->enabledLayers.size();
    createInfo.ppEnabledLayerNames     = cp->enabledLayers.data();
    createInfo.enabledExtensionCount   = (uint32_t)cp->enabledExtensions.size();
    createInfo.ppEnabledExtensionNames = cp->enabledExtensions.data();

    CHECK_VK( vkCreateInstance( &createInfo, nullptr, &cp->instance ) );

    printf( "Compute[%d]: created Vulkan instance\n", cp->hCompute );

    if ( cp->enableValidationLayers ) {
        VkDebugReportCallbackCreateInfoEXT createInfo = {};
        createInfo.sType                              = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        createInfo.flags                              = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        createInfo.pfnCallback                        = &_debugReportCallbackFn;

        auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr( cp->instance, "vkCreateDebugReportCallbackEXT" );
        if ( vkCreateDebugReportCallbackEXT == nullptr ) {
            printf( "ERROR: Compute: fail to GetProcAddress for debug callback\n" );
        }

        CHECK_VK( vkCreateDebugReportCallbackEXT( cp->instance, &createInfo, nullptr, &cp->debugReportCallback ) );
    }

    return true;
}


static bool _enableValidationLayers( _computeInstance* cp )
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

    cp->enabledLayers.push_back( "VK_LAYER_LUNARG_standard_validation" );

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

    cp->enabledExtensions.push_back( VK_EXT_DEBUG_REPORT_EXTENSION_NAME );

    printf( "Compute[%d]: enabled validation layers\n", cp->hCompute );

    return foundLayer && foundExtension;
}


static bool _findPhysicalDevice( _computeInstance* cp, uint32_t preferredDevice )
{
    bool rval = false;

    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices( cp->instance, &numDevices, nullptr );
    if ( numDevices == 0 ) {
        printf( "ERROR: Compute: No vulkan device found\n" );
        return false;
    }

    std::vector<VkPhysicalDevice> devices( numDevices );
    vkEnumeratePhysicalDevices( cp->instance, &numDevices, devices.data() );

    uint32_t         deviceIndex = 0;
    uint32_t         deviceID    = 0;
    char             deviceName[ 32 ];
    VkPhysicalDevice selectedDevice = nullptr;
    for ( VkPhysicalDevice device : devices ) {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties( device, &props );

        // Select the preferred device, if specified
        if ( preferredDevice != -1 && deviceIndex == preferredDevice ) {
            strncpy_s( deviceName, props.deviceName, ARRAY_SIZE( deviceName ) );
            deviceID       = props.deviceID;
            selectedDevice = device;
            rval           = true;
        }
        // Else, select the first device
        if ( !deviceID ) {
            strncpy_s( deviceName, props.deviceName, ARRAY_SIZE( deviceName ) );
            deviceID       = props.deviceID;
            selectedDevice = device;
            rval           = true;
        }
        deviceIndex++;

        printf( "deviceID = %d\n", props.deviceID );
        printf( "\tdeviceType = %d\n", props.deviceType );
        printf( "\tdeviceName = [%s]\n", props.deviceName );
        printf( "\tapiVersion = 0x%x\n", props.apiVersion );
        printf( "\tdriverVersion = 0x%x\n", props.driverVersion );
        printf( "\tvendorID = 0x%x\n", props.vendorID );

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
        printf( "\tmaxStorageBufferRange = %d\n", props.limits.maxStorageBufferRange );

        printf( "\n" );
    }

    cp->physicalDevice = selectedDevice;

    printf( "Compute[%d]: using physical device %d [%s]\n", cp->hCompute, deviceID, deviceName );

    return rval;
}


static uint32_t _findComputeQueueFamilyIndex( _computeInstance* cp )
{
    uint32_t numQueueFamilies;
    vkGetPhysicalDeviceQueueFamilyProperties( cp->physicalDevice, &numQueueFamilies, nullptr );
    std::vector<VkQueueFamilyProperties> queueFamilies( numQueueFamilies );
    vkGetPhysicalDeviceQueueFamilyProperties( cp->physicalDevice, &numQueueFamilies, queueFamilies.data() );

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
        printf( "ERROR: Compute[%d]: no compute queue found\n", cp->hCompute );
    }

    printf( "Compute[%d]: using queue %d\n", cp->hCompute, idx );

    return idx;
}


static uint32_t _findMemoryType( _computeInstance* cp, uint32_t type, VkMemoryPropertyFlags properties )
{
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    vkGetPhysicalDeviceMemoryProperties( cp->physicalDevice, &memoryProperties );

    for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ ) {
        printf( "Compute[%d]: memory type[%d]: 0x%x\n", cp->hCompute, i, memoryProperties.memoryTypes[ i ].propertyFlags );
    }

    for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ ) {
        if ( type & ( 1 << i ) && ( memoryProperties.memoryTypes[ i ].propertyFlags & properties ) == properties ) {
            return i;
        }
    }

    return (uint32_t)-1;
}


static bool _createLogicalDevice( _computeInstance* cp )
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
    deviceCreateInfo.enabledLayerCount        = (uint32_t)cp->enabledLayers.size();
    deviceCreateInfo.ppEnabledLayerNames      = cp->enabledLayers.data();
    // BUG: creation fails because "VK_EXT_debug_report" is not supported by selected device, even though
    // we enabled it earlier
    //deviceCreateInfo.enabledExtensionCount    = (uint32_t)cp->enabledExtensions.size();
    //deviceCreateInfo.ppEnabledExtensionNames  = cp->enabledExtensions.data();
    deviceCreateInfo.pQueueCreateInfos    = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures     = &deviceFeatures;

    CHECK_VK( vkCreateDevice( cp->physicalDevice, &deviceCreateInfo, nullptr, &cp->device ) );

    vkGetDeviceQueue( cp->device, cp->queueFamilyIndex, 0, &cp->queue );
    printf( "Compute[%d]: created logical device, queue 0x%llx\n", cp->hCompute, (uintptr_t)cp->queue );

    return true;
}

// TODO: this is shader-specific
static bool _createBuffer( size_t bufferSize, _computeInstance* cp ) // TODO: compute_job_t job
{
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size               = bufferSize;
    bufferCreateInfo.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // TODO: parameterize
    bufferCreateInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;          // TODO: parameterize

    CHECK_VK( vkCreateBuffer( cp->device, &bufferCreateInfo, nullptr, &cp->buffer ) );

    VkMemoryRequirements memoryRequirements = {};
    vkGetBufferMemoryRequirements( cp->device, cp->buffer, &memoryRequirements );

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize       = memoryRequirements.size;
    allocateInfo.memoryTypeIndex      = _findMemoryType( cp, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );

    CHECK_VK( vkAllocateMemory( cp->device, &allocateInfo, nullptr, &cp->bufferMemory ) );
    CHECK_VK( vkBindBufferMemory( cp->device, cp->buffer, cp->bufferMemory, 0 ) );
    cp->bufferSize = (uint32_t)bufferSize;

    printf( "Compute[%d]: allocated %zd bytes of storage buffer\n", cp->hCompute, bufferSize );

    return true;
}


// TODO: this is shader-specific
static bool _createDescriptorSetLayout( _computeInstance* cp ) // TODO: compute_job_t job
{
    // Define descriptors for shader resources
    // (in this case, a single handle to a storage buffer)

    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
    descriptorSetLayoutBinding.binding                      = 0;
    descriptorSetLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // or UNIFORM or SAMPLER or ...
    descriptorSetLayoutBinding.descriptorCount              = 1;
    descriptorSetLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount                    = 1;
    createInfo.pBindings                       = &descriptorSetLayoutBinding;

    CHECK_VK( vkCreateDescriptorSetLayout( cp->device, &createInfo, nullptr, &cp->descriptorSetLayout ) );
    printf( "Compute[%d]: defined %d descriptors\n", cp->hCompute, createInfo.bindingCount );

    return true;
}


// TODO: this is shader-specific
static bool _createDescriptorSet( _computeInstance* cp ) // TODO: compute_job_t job
{
    bool rval = false;

    // Bind descriptors to shader resources
    // (in this case, a single handle to a storage buffer)


    unsigned int numDescriptors = 1;

    VkDescriptorPoolSize poolSize = {};
    poolSize.type                 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount      = 1;

    VkDescriptorPoolCreateInfo createInfo = {};
    createInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.maxSets                    = 1;
    createInfo.poolSizeCount              = 1;
    createInfo.pPoolSizes                 = &poolSize;

    CHECK_VK( vkCreateDescriptorPool( cp->device, &createInfo, nullptr, &cp->descriptorPool ) );

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool              = cp->descriptorPool;
    allocInfo.descriptorSetCount          = numDescriptors;
    allocInfo.pSetLayouts                 = &cp->descriptorSetLayout;

    CHECK_VK( vkAllocateDescriptorSets( cp->device, &allocInfo, &cp->descriptorSet ) );

    VkDescriptorBufferInfo descriptorBufferInfo = {};
    descriptorBufferInfo.buffer                 = cp->buffer;
    descriptorBufferInfo.offset                 = 0;
    descriptorBufferInfo.range                  = cp->bufferSize;

    VkWriteDescriptorSet writeSet = {};
    writeSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.dstSet               = cp->descriptorSet;
    writeSet.dstBinding           = 0;
    writeSet.descriptorCount      = 1;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeSet.pBufferInfo          = &descriptorBufferInfo;

    vkUpdateDescriptorSets( cp->device, 1, &writeSet, 0, nullptr );
    printf( "Compute[%d]: bound %d descriptors\n", cp->hCompute, numDescriptors );

    return true;
}


uint32_t* _loadShader( const std::string& shaderPath, size_t* pShaderLength, _computeInstance* cp )
{
    FILE*   fp  = nullptr;
    errno_t err = fopen_s( &fp, shaderPath.c_str(), "rb" );
    if ( fp == nullptr || err == EINVAL ) {
        printf( "ERROR: Compute: failed to load shader [%s]\n", shaderPath.c_str() );
        return nullptr;
    }

    fseek( fp, 0, SEEK_END );
    size_t filesize = ftell( fp );
    fseek( fp, 0, SEEK_SET );

    // Vulkan / SPIR-V requires shader buffer to be an array of uint32_t padded with 0
    size_t    padded = size_t( ceil( filesize / 4.0f ) * 4 );
    uint32_t* buffer = new uint32_t[ padded ];
    fread( buffer, filesize, sizeof( uint8_t ), fp );
    for ( size_t i = filesize; i < padded; i++ ) {
        buffer[ i ] = 0;
    }

    fclose( fp );

    if ( pShaderLength ) {
        *pShaderLength = padded;
    }

    printf( "Compute[%d]: loaded %zd bytes of shader (padded to %zd)\n", cp->hCompute, filesize, padded );

    return buffer;
}


// TODO: this is shader-specific
static bool _createComputePipeline( const std::string& shaderPath, _computeInstance* cp ) // TODO: compute_job_t job
{
    size_t    shaderLength = 0;
    uint32_t* shaderBinary = _loadShader( shaderPath, &shaderLength, cp );

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = shaderBinary;
    shaderModuleCreateInfo.codeSize                 = shaderLength;
    CHECK_VK( vkCreateShaderModule( cp->device, &shaderModuleCreateInfo, nullptr, &cp->computeShaderModule ) );
    delete[] shaderBinary;

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module                          = cp->computeShaderModule;
    shaderStageCreateInfo.pName                           = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount             = 1;
    pipelineLayoutCreateInfo.pSetLayouts                = &cp->descriptorSetLayout;
    CHECK_VK( vkCreatePipelineLayout( cp->device, &pipelineLayoutCreateInfo, nullptr, &cp->pipelineLayout ) );

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage                       = shaderStageCreateInfo;
    pipelineCreateInfo.layout                      = cp->pipelineLayout;

    CHECK_VK( vkCreateComputePipelines( cp->device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &cp->pipeline ) );

    printf( "Compute[%d]: created shader pipeline for [%s]\n", cp->hCompute, shaderPath.c_str() );

    return true;
}


static bool _createCommandBuffer( _computeInstance* cp )
{
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags                   = 0;
    commandPoolCreateInfo.queueFamilyIndex        = cp->queueFamilyIndex;
    CHECK_VK( vkCreateCommandPool( cp->device, &commandPoolCreateInfo, nullptr, &cp->commandPool ) );

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool                 = cp->commandPool;
    allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount          = 1; // TODO: one per job?
    CHECK_VK( vkAllocateCommandBuffers( cp->device, &allocInfo, &cp->commandBuffer ) );

    // TODO: this part is job-specific
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // or VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT. Buffer is only submitted and used once
    CHECK_VK( vkBeginCommandBuffer( cp->commandBuffer, &beginInfo ) );

    vkCmdBindPipeline( cp->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipeline );
    vkCmdBindDescriptorSets( cp->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipelineLayout, 0, 1, &cp->descriptorSet, 0, nullptr );
    vkCmdDispatch( cp->commandBuffer, cp->workgroupWidth, cp->workgroupHeight, cp->workgroupDepth );
    CHECK_VK( vkEndCommandBuffer( cp->commandBuffer ) );

    printf( "Compute[%d]: dispatch command buffer, workgroup[%d x %d x %d]\n", cp->hCompute, cp->workgroupWidth, cp->workgroupHeight, cp->workgroupDepth );

    return true;
}


static bool _executeJobs( _computeInstance* cp, uint32_t timeoutMS )
{
    PerfTimer timer;
    printf( "Compute[%d]: execute...\n", cp->hCompute );

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cp->commandBuffer;

    // TODO: do this once per job
    //VkFence           fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    CHECK_VK( vkCreateFence( cp->device, &fenceCreateInfo, nullptr, &cp->fence ) );

    CHECK_VK( vkQueueSubmit( cp->queue, 1, &submitInfo, cp->fence ) );
    VkResult rval = vkWaitForFences( cp->device, 1, &cp->fence, VK_TRUE, timeoutMS );

    if ( rval != VK_SUCCESS ) {
        printf( "ERROR: Compute[%d]: execute timeout\n", cp->hCompute );
        return false;
    }

    printf( "Compute[%d]: executeJobs %d msec\n", cp->hCompute, (uint32_t)timer.ElapsedMilliseconds() );

    // TEST: save output of mandelbrot
    {
        void* mappedMemory = nullptr;
        vkMapMemory( cp->device, cp->bufferMemory, 0, cp->bufferSize, 0, &mappedMemory );

        struct Pixel {
            float r;
            float g;
            float b;
            float a;
        };

        Pixel* pixels = (Pixel*)mappedMemory;

        // Save image
        std::string filename = "mandelbrot.ppm";
        FILE*       file     = nullptr;
        errno_t     err      = fopen_s( &file, filename.c_str(), "w" );
        if ( !file || err != 0 ) {
            printf( "Error: failed to open [%s] for writing errno %d.\n", filename.c_str(), err );
            return false;
        }

        fprintf( file, "P3\n" );
        fprintf( file, "%d %d\n", COMPUTE_OUTPUT_WIDTH, COMPUTE_OUTPUT_HEIGHT );
        fprintf( file, "255\n" );

        for ( uint32_t y = 0; y < COMPUTE_OUTPUT_HEIGHT; y++ ) {
            for ( uint32_t x = 0; x < COMPUTE_OUTPUT_WIDTH; x++ ) {
                Pixel&  rgb = pixels[ y * COMPUTE_OUTPUT_WIDTH + x ];
                uint8_t _r  = ( uint8_t )( rgb.r * 255 );
                uint8_t _g  = ( uint8_t )( rgb.g * 255 );
                uint8_t _b  = ( uint8_t )( rgb.b * 255 );

                fprintf( file, "%d %d %d\n", _r, _g, _b );
            }
        }

        fflush( file );
        fclose( file );

        vkUnmapMemory( cp->device, cp->bufferMemory );
    }

    return true;
}


} // namespace pk
