#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void field_summary_kernel_ocl_
(double* vol, double* mass, double* ie, double* temp)
{
    chunk.field_summary_kernel(vol, mass, ie, temp);
}

void CloverChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    ENQUEUE_OFFSET(field_summary_device);

    *vol = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
    *mass = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
    *ie = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
    *temp = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

