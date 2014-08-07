#include "ocl_common.hpp"

extern "C" void advec_cell_kernel_ocl_
(const int* dr, const int* swp_nmbr, int * advec_int)
{
    chunk.advec_cell_kernel(*dr, *swp_nmbr, * advec_int);
}

void CloverChunk::advec_cell_kernel
(int dr, int swp_nmbr,int advec_int)
{
    if (1 == dr)
    {
        advec_cell_pre_vol_x_device.setArg(0, swp_nmbr);
        advec_cell_ener_flux_x_device.setArg(0, swp_nmbr);
        advec_cell_x_device.setArg(0, swp_nmbr);

        ENQUEUE_OFFSET(advec_cell_pre_vol_x_device);

        ENQUEUE_OFFSET(advec_cell_ener_flux_x_device);

        ENQUEUE_OFFSET(advec_cell_x_device);
    }
    else if (2 == dr)
    {
        advec_cell_pre_vol_y_device.setArg(0, swp_nmbr);
        advec_cell_pre_vol_y_device.setArg(7, advec_int);
        advec_cell_ener_flux_y_device.setArg(0, swp_nmbr);
        advec_cell_y_device.setArg(0, swp_nmbr);

        ENQUEUE_OFFSET(advec_cell_pre_vol_y_device);

        ENQUEUE_OFFSET(advec_cell_ener_flux_y_device);

        ENQUEUE_OFFSET(advec_cell_y_device);
    }
    else
    {
        advec_cell_pre_vol_z_device.setArg(0, swp_nmbr);
        advec_cell_ener_flux_z_device.setArg(0, swp_nmbr);
        advec_cell_z_device.setArg(0, swp_nmbr);

        ENQUEUE_OFFSET(advec_cell_pre_vol_z_device);

        ENQUEUE_OFFSET(advec_cell_ener_flux_z_device);

        ENQUEUE_OFFSET(advec_cell_z_device);
    }
}
