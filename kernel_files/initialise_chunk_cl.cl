#include "./kernel_files/macros_cl.cl"

__kernel void initialise_chunk_first
(const double d_xmin,
 const double d_ymin,
 const double d_zmin,
 const double d_dx,
 const double d_dy,
 const double d_dz,
 __global double * __restrict const vertexx,
 __global double * __restrict const vertexdx,
 __global double * __restrict const vertexy,
 __global double * __restrict const vertexdy,
 __global double * __restrict const vertexz,
 __global double * __restrict const vertexdz,
 __global       double * __restrict const cellx,
 __global       double * __restrict const celldx,
 __global       double * __restrict const celly,
 __global       double * __restrict const celldy,
 __global       double * __restrict const cellz,
 __global       double * __restrict const celldz)
{
    __kernel_indexes;

    // fill out x arrays
    if (row == 0 && slice == 0 && column <= (x_max + 2*HALO_DEPTH))
    {
        vertexx[column] = d_xmin + d_dx*(double)((((int)column) - 1) - HALO_DEPTH + 1);
        vertexdx[column] = d_dx;
    }

    // fill out y arrays
    if (column == 0 && slice == 0 && row <= (y_max + 2*HALO_DEPTH))
    {
        vertexy[row] = d_ymin + d_dy*(double)((((int)row) - 1) - HALO_DEPTH + 1);
        vertexdy[row] = d_dy;
    }

    // fill out y arrays
    if (row == 0 && column == 0 && slice <= (z_max + 2*HALO_DEPTH))
    {
        vertexz[slice] = d_zmin + d_dz*(double)((((int)slice) - 1) - HALO_DEPTH + 1);
        vertexdz[slice] = d_dz;
    }

    const double vertexx_plusone = d_xmin + d_dx*(double)((((int)column)) - HALO_DEPTH + 1);
    const double vertexy_plusone = d_ymin + d_dy*(double)((((int)row)) - HALO_DEPTH + 1);
    const double vertexz_plusone = d_zmin + d_dz*(double)((((int)slice)) - HALO_DEPTH + 1);

    //fill x arrays
    if (row == 0 && slice == 0 && column <= (x_max + HALO_DEPTH))
    {
        cellx[column] = 0.5 * (vertexx[column] + vertexx_plusone);
        celldx[column] = d_dx;
    }

    //fill y arrays
    if (column == 0 && slice == 0 && row <= (y_max + HALO_DEPTH))
    {
        celly[row] = 0.5 * (vertexy[row] + vertexy_plusone);
        celldy[row] = d_dy;
    }

    //fill y arrays
    if (row == 0 && column == 0 && slice <= (z_max + HALO_DEPTH))
    {
        cellz[slice] = 0.5 * (vertexz[slice] + vertexz_plusone);
        celldz[slice] = d_dz;
    }
}

__kernel void initialise_chunk_second
(const double d_xmin,
 const double d_ymin,
 const double d_zmin,
 const double d_dx,
 const double d_dy,
 const double d_dz,
 __global       double * __restrict const volume, 
 __global       double * __restrict const xarea, 
 __global       double * __restrict const yarea,
 __global       double * __restrict const zarea)
{
    __kernel_indexes;

    if (WITHIN_BOUNDS)
    {
        volume[THARR3D(0, 0, 0, 0, 0)] = d_dx * d_dy * d_dz;
        xarea[THARR3D(0, 0, 0, 1, 0)] = d_dy * d_dz;
        yarea[THARR3D(0, 0, 0, 0, 1)] = d_dx * d_dz;
        zarea[THARR3D(0, 0, 0, 0, 0)] = d_dx * d_dy;
    }
}

