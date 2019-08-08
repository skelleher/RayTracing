#pragma once

namespace pk
{

void testCPU();
void testCPUThreaded();
void testCUDA();
void testCompute( uint32_t preferredDevice = 0 );

} // namespace pk
