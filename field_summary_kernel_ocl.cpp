#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void field_summary_kernel_ocl_
(double* vol, double* mass, double* ie, double* ke, double* press, double* temp)
{
    chunk.field_summary_kernel(vol, mass, ie, ke, press, temp);
}

void CloverChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* ke, double* press, double* temp)
{
    ENQUEUE(field_summary_device);
    *vol = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
    *mass = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
    *ie = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
    *ke = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
    *press = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
    *temp = reduceValue<double>(sum_red_kernels_double, reduce_buf_6);
}
