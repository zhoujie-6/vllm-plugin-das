// SPDX-License-Identifier: Apache-2.0
#include <c10/hip/HIPException.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime_api.h>
#include <sys/types.h>

extern "C" void* dsv41_hip_segment_alloc(ssize_t bytes, int device, hipStream_t stream) {
    const c10::DeviceGuard guard(c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
    void* ptr = nullptr;
    auto status = hipMallocAsync(&ptr, bytes, stream);
    if (status == hipErrorOutOfMemory) {
        (void)hipGetLastError();
        return nullptr;
    }
    C10_HIP_CHECK(status);
    return ptr;
}

extern "C" void dsv41_hip_segment_free(void* ptr, ssize_t, int device, hipStream_t stream) {
    const c10::DeviceGuard guard(c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device)));
    C10_HIP_CHECK(hipFreeAsync(ptr, stream));
}
