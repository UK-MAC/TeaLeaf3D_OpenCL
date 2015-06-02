#include "./kernel_files/macros_cl.cl"
__kernel void generate_chunk
(__global const double * __restrict const vertexx,
 __global const double * __restrict const vertexy,
 __global const double * __restrict const vertexz,
 __global const double * __restrict const cellx,
 __global const double * __restrict const celly,
 __global const double * __restrict const cellz,
 __global       double * __restrict const density0,
 __global       double * __restrict const energy0,
 __global       double * __restrict const u,

 __global const double * __restrict const state_density,
 __global const double * __restrict const state_energy,
 __global const double * __restrict const state_xmin,
 __global const double * __restrict const state_xmax,
 __global const double * __restrict const state_ymin,
 __global const double * __restrict const state_ymax,
 __global const double * __restrict const state_zmin,
 __global const double * __restrict const state_zmax,
 __global const double * __restrict const state_radius,
 __global const int    * __restrict const state_geometry,

 const int g_rect,
 const int g_circ,
 const int g_point,

 const int state)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        const double x_cent = state_xmin[state];
        const double y_cent = state_ymin[state];
        const double z_cent = state_zmin[state];

        if (g_rect == state_geometry[state])
        {
            if (vertexx[1 + column] >= state_xmin[state]
            && vertexx[column] < state_xmax[state]
            && vertexy[1 + row] >= state_ymin[state]
            && vertexy[row] < state_ymax[state]
            && vertexz[1 + slice] >= state_zmin[state]
            && vertexz[slice] < state_zmax[state])
            {
                energy0[THARR3D(0, 0, 0,0,0)] = state_energy[state];
                density0[THARR3D(0, 0, 0,0,0)] = state_density[state];
            }
        }
        else if (state_geometry[state] == g_circ)
        {
            double x_pos = cellx[column]-x_cent;
            double y_pos = celly[row]-y_cent;
            double z_pos = cellz[slice]-z_cent;
            double radius = SQRT(x_pos*x_pos + y_pos*y_pos + z_pos*z_pos);

            if (radius <= state_radius[state])
            {
                energy0[THARR3D(0, 0, 0,0,0)] = state_energy[state];
                density0[THARR3D(0, 0, 0,0,0)] = state_density[state];
            }
        }
        else if (state_geometry[state] == g_point)
        {
            if (vertexx[column] == x_cent && vertexy[row] == y_cent && vertexz[slice] == z_cent)
            {
                energy0[THARR3D(0, 0, 0,0,0)] = state_energy[state];
                density0[THARR3D(0, 0, 0,0,0)] = state_density[state];
            }
        }

    }
}

__kernel void generate_chunk_init_u
(__global const double * density,
 __global const double * energy,
 __global       double * u,
 __global       double * u0)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        u[THARR3D(0, 0, 0, 0)] = energy[THARR3D(0, 0, 0, 0)]*density[THARR3D(0, 0, 0, 0)];
        u0[THARR3D(0, 0, 0, 0)] = energy[THARR3D(0, 0, 0, 0)]*density[THARR3D(0, 0, 0, 0)];
    }
}

__kernel void generate_chunk_init
(__global       double * density0,
 __global       double * energy0,
 __global const double * state_density,
 __global const double * state_energy)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        energy0[THARR3D(0, 0, 0,0,0)] = state_energy[0];
        density0[THARR3D(0, 0, 0,0,0)] = state_density[0];
    }
}

