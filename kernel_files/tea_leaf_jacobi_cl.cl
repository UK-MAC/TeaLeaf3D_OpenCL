#include <kernel_files/macros_cl.cl>

/*
 *  Kernels used for jacobi method
 */
#define COEF_CONDUCTIVITY 1
#define COEF_RECIP_CONDUCTIVITY 2
__kernel void tea_leaf_jacobi_init
(__global const double * __restrict const density1,
 __global const double * __restrict const energy1,
 __global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 __global       double * __restrict const Kz,
 __global       double * __restrict const u0,
 __global       double * __restrict const u1,
 const int coef)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 1
    && /*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 1)
    {
        u0[THARR3D(0, 0, 0, 0, 0)] = energy1[THARR3D(0, 0, 0, 0, 0)]*density1[THARR3D(0, 0, 0, 0, 0)];
        u1[THARR3D(0, 0, 0, 0, 0)] = energy1[THARR3D(0, 0, 0, 0, 0)]*density1[THARR3D(0, 0, 0, 0, 0)];

        // don't do this bit in second row/column
        if (row >= (y_min + 1)
        && column >= (x_min + 1))
        {
            double dens_centre, dens_left, dens_up, dens_front;

            if (COEF_CONDUCTIVITY == coef)
            {
                dens_centre = density1[THARR3D(0, 0, 0, 0, 0)];
                dens_left = density1[THARR3D(-1, 0, 0, 0, 0)];
                dens_up = density1[THARR3D(0, -1, 0, 0, 0)];
                dens_front = density1[THARR3D(0, 0, -1, 0, 0)];
            }
            else if (COEF_RECIP_CONDUCTIVITY == coef)
            {
                dens_centre = 1.0/density1[THARR3D(0, 0, 0, 0, 0)];
                dens_left = 1.0/density1[THARR3D(-1, 0, 0, 0, 0)];
                dens_up = 1.0/density1[THARR3D(0, -1, 0,0, 0)];
                dens_front = 1.0/density1[THARR3D(0, 0, -1, 0, 0)];
            }

            Kx[THARR3D(0, 0, 0, 0, 0)] = (dens_left + dens_centre)/(2.0*dens_left*dens_centre);
            Ky[THARR3D(0, 0, 0, 0, 0)] = (dens_up + dens_centre)/(2.0*dens_up*dens_centre);
            Kz[THARR3D(0, 0, 0, 0, 0)] = (dens_front + dens_centre)/(2.0*dens_front*dens_centre);
        }
    }
}

__kernel void tea_leaf_jacobi_copy_u
(__global const double * __restrict const u1,
 __global       double * __restrict const un)
{
    __kernel_indexes;

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 1
    && /*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 1)
    {
        un[THARR3D(0, 0, 0, 0, 0)] = u1[THARR3D(0, 0, 0, 0, 0)];
    }
}

__kernel void tea_leaf_jacobi_solve
(double rx, double ry, double rz,
 __global const double * __restrict const Kx,
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

    if (/*slice >= (y_min + 1) - 1 &&*/ slice <= (z_max + 1) + 1
    && /*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))
    {
        u1[THARR3D(0, 0, 0, 0, 0)] = (u0[THARR3D(0, 0, 0, 0, 0)]
            + Kx[THARR3D(1, 0, 0, 0, 0)]*rx*un[THARR3D( 1,  0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)]*rx*un[THARR3D(-1,  0, 0, 0, 0)]
            + Ky[THARR3D(0, 1, 0, 0, 0)]*ry*un[THARR3D( 0,  1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)]*ry*un[THARR3D( 0, -1, 0, 0, 0)]
            + Kz[THARR3D(0, 0, 1, 0, 0)]*rz*un[THARR3D( 0,  0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]*rz*un[THARR3D( 0, 0, -1, 0, 0)])
            /(1.0 + (Kx[THARR3D(0, 0, 0, 0, 0)] + Kx[THARR3D(1, 0, 0, 0, 0)])*rx
                  + (Ky[THARR3D(0, 0, 0, 0, 0)] + Ky[THARR3D(0, 1, 0, 0, 0)])*ry
                  + (Kz[THARR3D(0, 0, 0, 0, 0)] + Kz[THARR3D(0, 0, 1, 0, 0)])*rz);
        
        error_local[lid] = fabs(u1[THARR3D(0, 0, 0, 0, 0)] - un[THARR3D(0, 0, 0, 0, 0)]);
    }

    REDUCTION(error_local, error, MAX);
}

