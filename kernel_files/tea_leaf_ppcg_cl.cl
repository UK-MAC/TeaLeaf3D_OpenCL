#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_ppcg_solve_init_sd
(__global       double * __restrict const r,
 __global       double * __restrict const sd,

 __global       double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global       double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,

 __global const double * __restrict const u,
 __global const double * __restrict const u0,

 double theta)
{
    __kernel_indexes;

    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
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
            sd[THARR3D(0, 0, 0, 0, 0)] = z_l[lid]/theta;
        }
    }
    else if (WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            //z[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)];
            sd[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)]/theta;
        }
        else if (PRECONDITIONER == TL_PREC_NONE)
        {
            sd[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]/theta;
        }
    }
}

__kernel void tea_leaf_ppcg_solve_update_r
(__global       double * __restrict const u,
 __global       double * __restrict const r,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 __global const double * __restrict const sd)
{
    __kernel_indexes;

    // either matrix powers is enabled, or block jacobi is
    if (HALO_DEPTH >= 2 || WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0, 0)] += sd[THARR3D(0, 0, 0, 0, 0)];

        r[THARR3D(0, 0, 0, 0, 0)] -= SMVP(sd);
    }
}

__kernel void tea_leaf_ppcg_solve_calc_sd
(__global const double * __restrict const r,
 __global       double * __restrict const sd,

 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,

 __constant const double * __restrict const alpha,
 __constant const double * __restrict const beta,
 int step)
{
    __kernel_indexes;

    if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
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
            if (WITHIN_BOUNDS)
            {
                block_solve_func(r_l, z_l, cp, bfp, Kx, Ky, Kz);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            sd[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*sd[THARR3D(0, 0, 0, 0, 0)]
                                + beta[step]*z_l[lid];
        }
    }
    else if (HALO_DEPTH >= 2 || WITHIN_BOUNDS)
    {
        if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            //z[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)];
            sd[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*sd[THARR3D(0, 0, 0, 0, 0)]
                                + beta[step]*r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)];
        }
        else if (PRECONDITIONER == TL_PREC_NONE)
        {
            sd[THARR3D(0, 0, 0, 0, 0)] = alpha[step]*sd[THARR3D(0, 0, 0, 0, 0)]
                                + beta[step]*r[THARR3D(0, 0, 0, 0, 0)];
        }
    }
}

