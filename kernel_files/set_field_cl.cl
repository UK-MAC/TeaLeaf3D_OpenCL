#include <kernel_files/macros_cl.cl>

__kernel void set_field
(__global const double* __restrict const energy0,
 __global       double* __restrict const energy1)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy1[THARR3D(0, 0, 0, 0, 0)]  = energy0[THARR3D(0, 0, 0, 0, 0)];
    }
}

