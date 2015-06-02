#include <kernel_files/macros_cl.cl>
#include <kernel_files/tea_block_jacobi.cl>

__kernel void tea_leaf_finalise
(__global const double * __restrict const density,
 __global const double * __restrict const u,
 __global       double * __restrict const energy)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy[THARR3D(0, 0, 0, 0, 0)] = u[THARR3D(0, 0, 0, 0, 0)]/density[THARR3D(0, 0, 0, 0, 0)];
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

    if (WITHIN_BOUNDS)
    {
        r[THARR3D(0, 0, 0, 0, 0)] = u0[THARR3D(0, 0, 0, 0, 0)] - SMVP(u);
    }
}

__kernel void tea_leaf_calc_2norm
(__global const double * const r1,
 __global const double * const r2,
 __global       double * __restrict const rro)
{
    __kernel_indexes;

    __local double rro_shared[BLOCK_SZ];
    rro_shared[lid] = 0.0;

    if (WITHIN_BOUNDS)
    {
        rro_shared[lid] = r1[THARR3D(0, 0, 0, 0, 0)]*r2[THARR3D(0, 0, 0, 0, 0)];
    }

    REDUCTION(rro_shared, rro, SUM)
}

__kernel void tea_leaf_init_common
(__global const double * __restrict const density,
 __global const double * __restrict const energy,
 __global       double * __restrict const Kx,
 __global       double * __restrict const Ky,
 __global       double * __restrict const Kz,
 __global       double * __restrict const u0,
 __global       double * __restrict const u,
 double rx, double ry, double rz,
 const int coef)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        double dens_centre, dens_left, dens_down, dens_back;

        if (COEF_CONDUCTIVITY == coef)
        {
            dens_centre = density[THARR3D(0, 0, 0, 0, 0)];
            dens_left = density[THARR3D(-1, 0, 0, 0, 0)];
            dens_down = density[THARR3D(0, -1, 0, 0, 0)];
            dens_back = density[THARR3D(0, 0, -1, 0, 0)];
        }
        else if (COEF_RECIP_CONDUCTIVITY == coef)
        {
            dens_centre = 1.0/density[THARR3D(0, 0, 0, 0, 0)];
            dens_left = 1.0/density[THARR3D(-1, 0, 0, 0, 0)];
            dens_down = 1.0/density[THARR3D(0, -1, 0, 0, 0)];
            dens_back = 1.0/density[THARR3D(0, 0, -1, 0, 0)];
        }

        Kx[THARR3D(0, 0, 0, 0, 0)] = (dens_left + dens_centre)/(2.0*dens_left*dens_centre);
        Kx[THARR3D(0, 0, 0, 0, 0)] *= rx;
        Ky[THARR3D(0, 0, 0, 0, 0)] = (dens_down + dens_centre)/(2.0*dens_down*dens_centre);
        Ky[THARR3D(0, 0, 0, 0, 0)] *= ry;
        Kz[THARR3D(0, 0, 0, 0, 0)] = (dens_back + dens_centre)/(2.0*dens_back*dens_centre);
        Kz[THARR3D(0, 0, 0, 0, 0)] *= rz;
    }
}

__kernel void tea_leaf_init_jac_diag
(__global       double * __restrict const Mi,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double diag = (1.0
            + (Kx[THARR3D(1, 0, 0, 0, 0)] + Kx[THARR3D(0, 0, 0, 0, 0)])
            + (Ky[THARR3D(0, 1, 0, 0, 0)] + Ky[THARR3D(0, 0, 0, 0, 0)])
            + (Kz[THARR3D(0, 0, 1, 0, 0)] + Kz[THARR3D(0, 0, 0, 0, 0)]));

        Mi[THARR3D(0, 0, 0, 0, 0)] = 1.0/diag;
    }
}
