#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_cheby_solve_init_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const w,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 double theta, double rx, double ry, double rz)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        w[THARR3D(0, 0, 0, 0, 0)] = SMVP(u);

        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - w[THARR3D(0, 0, 0, 0, 0)];
    }

    #if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __local double r_l[BLOCK_SZ];
        __local double z_l[BLOCK_SZ];

        r_l[lid] = 0;
        z_l[lid] = 0;

        if (WITHIN_BOUNDS)
        {
            r_l[lid] = r[THARR3D(0, 0, 0, 0, 0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(r_l, z_l, cp, bfp, Kx, Ky, Kz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = z_l[lid]/theta;
        }
    }
    #else
    if (WITHIN_BOUNDS)
    {
        #if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = (Mi[THARR3D(0, 0, 0, 0, 0)]*r[THARR3D(0, 0, 0, 0, 0)])/theta;
        }
        #else
        {
            p[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]/theta;
        }
        #endif
    }
    #endif
}

__kernel void tea_leaf_cheby_solve_calc_u
(__global       double * __restrict const u,
 __global const double * __restrict const p)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0, 0)] += p[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_cheby_solve_calc_p
(__global const double * __restrict const u,
 __global const double * __restrict const u0,
 __global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const w,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 double rx, double ry, double rz, int step)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        w[THARR3D(0, 0, 0, 0, 0)] = SMVP(u);

        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - w[THARR3D(0, 0, 0, 0, 0)];
    }

    #if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __local double r_l[BLOCK_SZ];
        __local double z_l[BLOCK_SZ];

        r_l[lid] = 0;
        z_l[lid] = 0;

        if (WITHIN_BOUNDS)
        {
            r_l[lid] = r[THARR3D(0, 0, 0, 0, 0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(r_l, z_l, cp, bfp, Kx, Ky, Kz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*p[THARR3D(0, 0, 0, 0, 0)]
                            + beta[step]*z_l[lid];
        }
    }
    #else
    if (WITHIN_BOUNDS)
    {
        #if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*p[THARR3D(0, 0, 0, 0, 0)]
                                + beta[step]*Mi[THARR3D(0, 0, 0, 0, 0)]*r[THARR3D(0, 0, 0, 0, 0)];
        }
        #else
        {
            p[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*p[THARR3D(0, 0, 0, 0, 0)]
                            + beta[step]*r[THARR3D(0, 0, 0, 0, 0)];
        }
        #endif
    }
    #endif
}

