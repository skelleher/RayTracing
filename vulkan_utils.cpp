// *****************************************************************************
// Common utility methods
// *****************************************************************************

#include "vulkan_utils.h"

#include "utils.h"

#include <stdint.h>

namespace pk
{

static uint32_t  _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties );
static uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength );


result VulkanUtils::createComputeShader( const VkDevice &device, const std::string& shaderPath, VkShaderModule* pShader )
{
    uint32_t  shaderLength = 0;
    uint32_t* shaderBinary = nullptr;

    shaderBinary = _loadShader( shaderPath, &shaderLength );

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pCode                    = shaderBinary;
    shaderModuleCreateInfo.codeSize                 = shaderLength;
    CHECK_VK( vkCreateShaderModule( device, &shaderModuleCreateInfo, nullptr, pShader ) );
    delete[] shaderBinary;

    return R_OK;
}


bool VulkanUtils::createBuffer( const VkDevice& device, const VkPhysicalDevice& physicalDevice, size_t bufferSize, VkBuffer* pBuffer, VkDeviceMemory* pBufferMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties )
{
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size               = bufferSize;
    bufferCreateInfo.usage              = usage;
    bufferCreateInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

    CHECK_VK( vkCreateBuffer( device, &bufferCreateInfo, nullptr, pBuffer ) );

    VkMemoryRequirements memoryRequirements = {};
    vkGetBufferMemoryRequirements( device, *pBuffer, &memoryRequirements );

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize       = memoryRequirements.size;
    allocateInfo.memoryTypeIndex      = _findMemoryType( physicalDevice, memoryRequirements.memoryTypeBits, properties );

    CHECK_VK( vkAllocateMemory( device, &allocateInfo, nullptr, pBufferMemory ) );
    CHECK_VK( vkBindBufferMemory( device, *pBuffer, *pBufferMemory, 0 ) );

    //printf( "ComputeJob[%d:%d]: allocated %zd bytes of buffer usage 0x%x props 0x%x\n", hCompute, handle, bufferSize, usage, properties );

    return true;
}


//
// Private Implementation
//

static uint32_t _findMemoryType( VkPhysicalDevice physicalDevice, uint32_t type, VkMemoryPropertyFlags properties )
{
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    vkGetPhysicalDeviceMemoryProperties( physicalDevice, &memoryProperties );

    for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ ) {
        if ( type & ( 1 << i ) && ( memoryProperties.memoryTypes[ i ].propertyFlags & properties ) == properties ) {
            return i;
        }
    }

    return (uint32_t)-1;
}


static uint32_t* _loadShader( const std::string& shaderPath, uint32_t* pShaderLength )
{
    FILE*   fp  = nullptr;
    errno_t err = fopen_s( &fp, shaderPath.c_str(), "rb" );
    if ( fp == nullptr || err == EINVAL ) {
        printf( "ERROR: ComputeJob: failed to load shader [%s]\n", shaderPath.c_str() );
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
        *pShaderLength = (uint32_t)padded;
    }

    printf( "ComputeJob: loaded %zd bytes of shader (padded to %zd)\n", filesize, padded );

    return buffer;
}


} // namespace pk
