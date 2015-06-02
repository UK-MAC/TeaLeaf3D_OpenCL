
#define COEF_A (-Kz[THARR3D(0, 0, k+0, 0, 0)])

#define COEF_B (1.0                                             \
    + (Kx[THARR3D(1, 0, k+0, 0, 0)] + Kx[THARR3D(0, 0, k+0, 0, 0)])       \
    + (Ky[THARR3D(0, 1, k+0, 0, 0)] + Ky[THARR3D(0, 0, k+0, 0, 0)])       \
    + (Kz[THARR3D(0, 0, k+1, 0, 0)] + Kz[THARR3D(0, 0, k+0, 0, 0)]))

#define COEF_C (-Kz[THARR3D(0, 0, k+1, 0, 0)])

void block_solve_func
(__local const double r_local[BLOCK_SZ],
 __local       double z_local[BLOCK_SZ],
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    const size_t column = get_global_id(0);
    const size_t row = get_global_id(1);
    const size_t slice = get_global_id(2);

    const size_t loc_column = get_local_id(0);
    const size_t loc_row_size = LOCAL_X;

    const size_t upper_limit = BLOCK_TOP;

    int k = 0;
#define LOC_K (loc_column + k*loc_row_size)

    __private double dp_priv[JACOBI_BLOCK_SIZE];

    dp_priv[k] = r_local[LOC_K]/COEF_B;

    for (k = 1; k < upper_limit; k++)
    {
        dp_priv[k] = (r_local[LOC_K] - COEF_A*dp_priv[k - 1])*bfp[THARR3D(0, 0, 0, k, 0)];
    }

    k = upper_limit - 1;

    z_local[LOC_K] = dp_priv[k];

    for (k = upper_limit - 2; k >= 0; k--)
    {
        z_local[LOC_K] = dp_priv[k] - cp[THARR3D(0, 0, 0, k, 0)]*z_local[LOC_K + LOCAL_X];
    }
}

__kernel void tea_leaf_block_solve
(__global const double * __restrict const r,
 __global       double * __restrict const z,
 __global const double * __restrict const cp,
 __global const double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    __kernel_indexes;

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
        z[THARR3D(0, 0, 0, 0, 0)] = z_l[lid];
    }
}

__kernel void tea_leaf_block_init
(__global const double * __restrict const r,
 __global const double * __restrict const z,
 __global       double * __restrict const cp,
 __global       double * __restrict const bfp,
 __global const double * __restrict const Kx,
 __global const double * __restrict const Ky,
 __global const double * __restrict const Kz)
{
    __kernel_indexes;

    const size_t upper_limit = BLOCK_TOP;

    if (WITHIN_BOUNDS)
    {
        if (loc_row == 0)
        {
            int k = 0;

            cp[THARR3D(0, 0, k, 0, 0)] = COEF_C/COEF_B;

            for (k = 1; k < upper_limit; k++)
            {
                bfp[THARR3D(0, 0, k, 0, 0)] = 1.0/(COEF_B - COEF_A*cp[THARR3D(0, 0, k - 1, 0, 0)]);
                cp[THARR3D(0, 0, k, 0, 0)] = COEF_C*bfp[THARR3D(0, 0, k, 0, 0)];
            }
        }
    }
}

