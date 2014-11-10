#include "./kernel_files/macros_cl.cl"
__kernel void field_summary
(__global const double * __restrict const volume,
 __global const double * __restrict const density0,
 __global const double * __restrict const energy0,
 __global const double * __restrict const u,

 __global double * __restrict const vol,
 __global double * __restrict const mass,
 __global double * __restrict const ie,
 __global double * __restrict const temp)
{
    __kernel_indexes;

    __local double vol_shared[BLOCK_SZ];
    __local double mass_shared[BLOCK_SZ];
    __local double ie_shared[BLOCK_SZ];
    __local double temp_shared[BLOCK_SZ];
    vol_shared[lid] = 0.0;
    mass_shared[lid] = 0.0;
    ie_shared[lid] = 0.0;
    temp_shared[lid] = 0.0;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1)
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1))
    {
        const double cell_vol = volume[THARR3D(0, 0, 0,0,0)];
        const double cell_mass = cell_vol * density0[THARR3D(0, 0, 0,0,0)];

        vol_shared[lid] = cell_vol;
        mass_shared[lid] = cell_mass;
        ie_shared[lid] = cell_mass * energy0[THARR3D(0, 0, 0,0,0)];
        temp_shared[lid] = cell_mass*u[THARR3D(0, 0, 0, 0, 0)];
    }

    REDUCTION(vol_shared, vol, SUM)
    REDUCTION(mass_shared, mass, SUM)
    REDUCTION(ie_shared, ie, SUM)
    REDUCTION(temp_shared, temp, SUM)
}
