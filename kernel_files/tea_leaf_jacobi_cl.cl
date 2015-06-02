#include <kernel_files/macros_cl.cl>

__kernel void tea_leaf_jacobi_copy_u
(__global const double * __restrict const u1,
 __global       double * __restrict const un)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        un[THARR3D(0, 0, 0, 0)] = u1[THARR3D(0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_jacobi_solve
(__global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz,
 __global const double * __restrict const u0,
 __global       double * __restrict const u1,
 __global const double * __restrict const un,
 __global       double * __restrict const error)
{
    __kernel_indexes;

    __local double error_local[BLOCK_SZ];
    error_local[lid] = 0;

    if (WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0, 0)] = (u0[THARR3D(0, 0, 0, 0, 0)]
            + Kx[THARR3D(1, 0, 0, 0, 0)]*rx*u_old[THARR3D( 1,  0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*rx*u_old[THARR3D(-1,  0, 0, 0, 0)]
            + Ky[THARR3D(0, 1, 0, 0, 0)]*ry*u_old[THARR3D( 0,  1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*ry*u_old[THARR3D( 0, -1, 0, 0, 0)]
            + Kz[THARR3D(0, 0, 1, 0, 0)]*rz*u_old[THARR3D( 0,  0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*rz*u_old[THARR3D( 0, 0, -1, 0, 0)])
            /(1.0 + (Kx[THARR3D(0, 0, 0, 0, 0)] + Kx[THARR3D(1, 0, 0, 0, 0)])*rx
                  + (Ky[THARR3D(0, 0, 0, 0, 0)] + Ky[THARR3D(0, 1, 0, 0, 0)])*ry
                  + (Kz[THARR3D(0, 0, 0, 0, 0)] + Kz[THARR3D(0, 0, 1, 0, 0)])*rz);
        
        error_local[lid] = fabs(u1[THARR3D(0, 0, 0, 0)] - un[THARR3D(0, 0, 0, 0)]);
    }

    REDUCTION(error_local, error, SUM);
}

