#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_init_diag
(__global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 __global       double * __restrict const Kz,
 double rx, double ry, double rz)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 1
    && /*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 && */column <= (x_max + 1) + 1)
    {
        Kx[THARR3D(0, 0, 0, 0, 0)] *= rx;
        Ky[THARR3D(0, 0, 0, 0, 0)] *= ry;
        Kz[THARR3D(0, 0, 0, 0, 0)] *= rz;
    }
}

__kernel void tea_leaf_finalise
(__global const double * __restrict const density1,
 __global const double * __restrict const u1,
 __global       double * __restrict const energy1)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        energy1[THARR3D(0, 0, 0, 0, 0)] = u1[THARR3D(0, 0, 0, 0, 0)]/density1[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_calc_residual
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const r,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        const double smvp = (1.0
            + (Kx[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)])
            + (Ky[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)])
            + (Kz[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]))*u[THARR3D(0, 0, 0, 0, 0)]
            - (Kx[THARR3D(1, 0, 0, 0, 0)]*u[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(-1, 0, 0, 0, 0)])
            - (Ky[THARR3D(0, 1, 0, 0, 0)]*u[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, -1, 0, 0, 0)])
            - (Kz[THARR3D(0, 0, 1, 0, 0)]*u[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, 0, -1, 0, 0)]);

        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - smvp;
    }
}
