#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

/*
 *  Kernels used for conjugate gradient method
 */

__kernel void tea_leaf_cg_solve_init_p
(__global       double * __restrict const p,
 __global       double * __restrict const r,
 __global       double * __restrict const z,
 __global       double * __restrict const Mi,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        #if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
        {
            // z initialised when block_solve kernel is called before this
            p[THARR3D(0, 0, 0, 0, 0)] = z[THARR3D(0, 0, 0, 0, 0)];
        }
        #elif (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            z[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)];
            p[THARR3D(0, 0, 0, 0, 0)] = z[THARR3D(0, 0, 0, 0, 0)];
        }
        #elif (PRECONDITIONER == TL_PREC_NONE)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)];
        }
        #endif

        rro_shared[lid] = r[THARR3D(0, 0, 0, 0, 0)]*p[THARR3D(0, 0, 0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

/* reduce rro */

__kernel void tea_leaf_cg_solve_calc_w
(__global       double * __restrict const pw,
 __global const double * __restrict const p,
 __global       double * __restrict const w,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    __kernel_indexes;

    __local double pw_shared[BLOCK_SZ];
    pw_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        w[THARR3D(0, 0, 0, 0, 0)] = SMVP(p);
        
        pw_shared[lid] = p[THARR3D(0, 0, 0, 0, 0)]*w[THARR3D(0, 0, 0, 0, 0)];
    }

    REDUCTION(pw_shared, pw, SUM);
}

/* reduce pw */

__kernel void tea_leaf_cg_solve_calc_ur
(double alpha,
 __global       double * __restrict const u,
 __global const double * __restrict const p,
 __global       double * __restrict const r,
 __global const double * __restrict const w,

 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,

 __global       double * __restrict const rrn)
{
    __kernel_indexes;

    __local double rrn_shared[BLOCK_SZ];
    rrn_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0, 0)] += alpha*p[THARR3D(0, 0, 0, 0, 0)];
        rrn_shared[lid] = r[THARR3D(0, 0, 0, 0, 0)] -= alpha*w[THARR3D(0, 0, 0, 0, 0)];
    }

    #if (PRECONDITIONER == TL_PREC_JAC_BLOCK)
    {
        __local double z_l[BLOCK_SZ];

        //if (WITHIN_BOUNDS)
        //{
        //    rrn_shared[lid] = r[THARR3D(0, 0, 0, 0, 0)];
        //}

        barrier(CLK_LOCAL_MEM_FENCE);
        if (loc_row == 0)
        {
            block_solve_func(rrn_shared, z_l, cp, bfp, Kx, Ky, Kz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (WITHIN_BOUNDS)
        {
            z[THARR3D(0, 0, 0, 0, 0)] = z_l[lid];

            rrn_shared[lid] *= z_l[lid];
        }
    }
    #else
    if (WITHIN_BOUNDS)
    {
        #if (PRECONDITIONER == TL_PREC_JAC_DIAG)
        {
            z[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)]*Mi[THARR3D(0, 0, 0, 0, 0)];
            rrn_shared[lid] *= z[THARR3D(0, 0, 0, 0, 0)];
        }
        #elif (PRECONDITIONER == TL_PREC_NONE)
        {
            rrn_shared[lid] *= rrn_shared[lid];
        }
        #endif
    }
    #endif

    REDUCTION(rrn_shared, rrn, SUM);
}

/* reduce rrn */

__kernel void tea_leaf_cg_solve_calc_p
(double beta,
 __global       double * __restrict const p,
 __global const double * __restrict const r,
 __global const double * __restrict const z)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        #if (PRECONDITIONER != TL_PREC_NONE)
        {
            p[THARR3D(0, 0, 0, 0, 0)] = z[THARR3D(0, 0, 0, 0, 0)] + beta*p[THARR3D(0, 0, 0, 0, 0)];
        }
        #else
        {
            p[THARR3D(0, 0, 0, 0, 0)] = r[THARR3D(0, 0, 0, 0, 0)] + beta*p[THARR3D(0, 0, 0, 0, 0)];
        }
        #endif
    }
}

