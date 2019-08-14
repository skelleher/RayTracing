#pragma once

#include <memory>
#include <stdint.h>


namespace pk
{
enum ComputeBufferType {
    COMPUTE_BUFFER_TYPE_UNKNOWN = 0,
    COMPUTE_BUFFER_UNIFORM      = 1,
    COMPUTE_BUFFER_STORAGE      = 2,
};

enum ComputeBufferVisibility {
    COMPUTE_BUFFER_VISIBILITY_UNKNOWN = 0,
    COMPUTE_BUFFER_SHARED             = 1,
    COMPUTE_BUFFER_DEVICE             = 2,
};

struct ComputeBufferDims {
    size_t width;
    size_t height;
    size_t elementSize;

    ComputeBufferDims() = default;
};

class IComputeBuffer {
public:
    // Creation and initialization of an IComputeBuffer is specific to each subclass
    // NOTE: never call bind() or resize() while the buffer is in use by the GPU
    virtual bool   bind( void* shader )                    = 0;
    virtual bool   resize( const ComputeBufferDims& dims ) = 0;
    virtual size_t size() const                            = 0;
    virtual void   map()                                   = 0;
    virtual void   unmap()                                 = 0;
    virtual void   free()                                  = 0;

    // must be public for smart pointers
    virtual ~IComputeBuffer() = default;

    // TODO: make these properties lest someone modify them incorrectly
    uint32_t                binding; // Shader descriptor slot to bind to
    ComputeBufferType       type;
    ComputeBufferVisibility visibility;
    ComputeBufferDims       dims;
    void*                   mapped;
    bool                    sizeHasChanged; // Indicates shader's command buffer / pipeline may need to be regenerated

protected:
    IComputeBuffer( uint32_t binding, ComputeBufferType type, ComputeBufferVisibility visibility, const ComputeBufferDims& dims, void* mapped ) :
        binding( binding ),
        type( type ),
        visibility( visibility ),
        dims( dims ),
        mapped( mapped ),
        sizeHasChanged( false )
    {
    }

    IComputeBuffer()                            = default;
    IComputeBuffer( const IComputeBuffer& rhs ) = delete;
    IComputeBuffer& operator=( const IComputeBuffer& ) = delete;
};

typedef std::shared_ptr<IComputeBuffer> IComputeBufferPtr;

} // namespace pk
