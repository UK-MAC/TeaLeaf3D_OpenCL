#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_cheby_solve_init_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global const double * __restrict const Mi,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 double theta, double rx, double ry, double rz)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        w[THARR3D(0, 0, 0, 0, 0)] = (1.0
            + (Kx[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)])
            + (Ky[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)])
            + (Kz[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]))*u[THARR3D(0, 0, 0, 0, 0)]
            - (Kx[THARR3D(1, 0, 0, 0, 0)]*u[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(-1, 0, 0, 0, 0)])
            - (Ky[THARR3D(0, 1, 0, 0, 0)]*u[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, -1, 0, 0, 0)])
            - (Kz[THARR3D(0, 0, 1, 0, 0)]*u[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, 0, -1, 0, 0)]);

        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - w[THARR3D(0, 0, 0, 0, 0)];
        p[THARR3D(0, 0, 0, 0, 0)] = (Mi[THARR3D(0, 0, 0, 0, 0)]*r[THARR3D(0, 0, 0, 0, 0)])/theta;
    }
}

__kernel void tea_leaf_cheby_solve_calc_u
(__global       double * __restrict const u,
 __global const double * __restrict const p)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 && */row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        u[THARR3D(0, 0, 0, 0, 0)] += p[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const Mi,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 double rx, double ry, double rz, int step)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        w[THARR3D(0, 0, 0, 0, 0)] = (1.0
            + (Kx[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)])
            + (Ky[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)])
            + (Kz[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]))*u[THARR3D(0, 0, 0, 0, 0)]
            - (Kx[THARR3D(1, 0, 0, 0, 0)]*u[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(-1, 0, 0, 0, 0)])
            - (Ky[THARR3D(0, 1, 0, 0, 0)]*u[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, -1, 0, 0, 0)])
            - (Kz[THARR3D(0, 0, 1, 0, 0)]*u[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*u[THARR3D(0, 0, -1, 0, 0)]);

        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - w[THARR3D(0, 0, 0, 0, 0)];
        p[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*p[THARR3D(0, 0, 0, 0, 0)]
                                  + beta[step]*Mi[THARR3D(0, 0, 0, 0, 0)]*r[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_calc_2norm
(__global const double * __restrict const r,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 0
    && /*row >= (y_min + 1) - 0 &&*/ row <= (y_max + 1) + 0
    && /*column >= (x_min + 1) - 0 &&*/ column <= (x_max + 1) + 0)
    {
        rro_shared[lid] = r[THARR3D(0, 0, 0, 0, 0)]*r[THARR3D(0, 0, 0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}


