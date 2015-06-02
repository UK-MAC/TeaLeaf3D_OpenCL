#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_jacobi_copy_u
(__global const double * __restrict const u,
 __global       double * __restrict const u_old)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u_old[THARR3D(0, 0, 0, 0, 0)] = u[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_jacobi_solve
(__global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 __global const double * __restrict const u0,
 __global       double * __restrict const u,
 __global const double * __restrict const u_old,
 __global       double * __restrict const error)
{
    __kernel_indexes;

    __local double error_local[BLOCK_SZ];
    error_local[lid] = 0;

    if (WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0, 0)] = (u0[THARR3D(0, 0, 0, 0, 0)]
            + Kx[THARR3D(1, 0, 0, 0, 0)]*u_old[THARR3D( 1,  0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*u_old[THARR3D(-1,  0, 0, 0, 0)]
            + Ky[THARR3D(0, 1, 0, 0, 0)]*u_old[THARR3D( 0,  1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*u_old[THARR3D( 0, -1, 0, 0, 0)]
            + Kz[THARR3D(0, 0, 1, 0, 0)]*u_old[THARR3D( 0,  0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*u_old[THARR3D( 0, 0, -1, 0, 0)])
            /(1.0 + (Kx[THARR3D(0, 0, 0, 0, 0)] + Kx[THARR3D(1, 0, 0, 0, 0)])
                  + (Ky[THARR3D(0, 0, 0, 0, 0)] + Ky[THARR3D(0, 1, 0, 0, 0)])
                  + (Kz[THARR3D(0, 0, 0, 0, 0)] + Kz[THARR3D(0, 0, 1, 0, 0)]));
        
        error_local[lid] = fabs(u[THARR3D(0, 0, 0, 0, 0)] - u_old[THARR3D(0, 0, 0, 0, 0)]);
    }

    REDUCTION(error_local, error, SUM);
}

