#include "ocl_common.hpp"

void CloverChunk::initBuffers
(void)
{
    size_t total_cells = (z_max+2*halo_allocate_depth+1) * (x_max+2*halo_allocate_depth+1) * (y_max+2*halo_allocate_depth+1);
    const std::vector<double> zeros(total_cells, 0.0);

    #define BUF_ALLOC(name, buf_sz)                 \
        try                                         \
        {                                           \
            name = cl::Buffer(context,              \
                              CL_MEM_READ_WRITE,    \
                              (buf_sz));            \
            queue.enqueueWriteBuffer(name,          \
                                     CL_TRUE,       \
                                     0,             \
                                     (buf_sz),      \
                                     &zeros[0]);    \
        }                                           \
        catch (cl::Error e)                         \
        {                                           \
            DIE("Error in creating %s buffer %d\n", \
                    #name, e.err());                \
        }

    #define BUF1DX_ALLOC(name, x_e)     \
        BUF_ALLOC(name, (x_max+2*halo_allocate_depth+x_e) * sizeof(double))

    #define BUF1DY_ALLOC(name, y_e)     \
        BUF_ALLOC(name, (y_max+2*halo_allocate_depth+y_e) * sizeof(double))

    #define BUF1DZ_ALLOC(name, z_e)     \
        BUF_ALLOC(name, (z_max+2*halo_allocate_depth+z_e) * sizeof(double))

    #define BUF2D_ALLOC(name, x_e, y_e) \
        BUF_ALLOC(name, (x_max+2*halo_allocate_depth+x_e) * (y_max+2*halo_allocate_depth+y_e) * sizeof(double))
    #define BUF3D_ALLOC(name, x_e, y_e,z_e) \
        BUF_ALLOC(name, (x_max+2*halo_allocate_depth+x_e) * (y_max+2*halo_allocate_depth+y_e) *(z_max+2*halo_allocate_depth+z_e) * sizeof(double))

    BUF3D_ALLOC(density, 0, 0,0);
    BUF3D_ALLOC(energy0, 0, 0,0);
    BUF3D_ALLOC(energy1, 0, 0,0);

    BUF3D_ALLOC(volume, 0, 0, 0);
    BUF3D_ALLOC(xarea, 1, 0, 0);
    BUF3D_ALLOC(yarea, 0, 1, 0);
    BUF3D_ALLOC(zarea, 0, 0, 1);

    BUF1DX_ALLOC(cellx, 0);
    BUF1DX_ALLOC(celldx, 0);
    BUF1DX_ALLOC(vertexx, 1);
    BUF1DX_ALLOC(vertexdx, 1);

    BUF1DY_ALLOC(celly, 0);
    BUF1DY_ALLOC(celldy, 0);
    BUF1DY_ALLOC(vertexy, 1);
    BUF1DY_ALLOC(vertexdy, 1);

    BUF1DZ_ALLOC(cellz, 0);
    BUF1DZ_ALLOC(celldz, 0);
    BUF1DZ_ALLOC(vertexz, 1);
    BUF1DZ_ALLOC(vertexdz, 1);

    // work arrays used in various kernels (post_vol, pre_vol, mom_flux, etc)
    BUF3D_ALLOC(vector_p, 0, 0, 0);
    BUF3D_ALLOC(vector_r, 0, 0, 0);
    BUF3D_ALLOC(vector_w, 0, 0, 0);
    BUF3D_ALLOC(vector_Mi, 0, 0, 0);
    BUF3D_ALLOC(vector_Kx, 0, 0, 0);
    BUF3D_ALLOC(vector_Ky, 0, 0, 0);
    BUF3D_ALLOC(vector_Kz, 0, 0, 0);
    BUF3D_ALLOC(vector_sd, 0, 0, 0);

    // tealeaf
    BUF3D_ALLOC(u, 0, 0, 0);
    BUF3D_ALLOC(u0, 0, 0, 0);
    BUF3D_ALLOC(vector_z, 0, 0, 0);

    BUF3D_ALLOC(cp, 0, 0, 0);
    BUF3D_ALLOC(bfp, 0, 0, 0);

    // allocate enough for 1 item per work group, and then a bit extra for the reduction
    // 1.5 should work even if wg size is 2
    BUF_ALLOC(reduce_buf_1, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_2, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_3, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_4, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_5, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(reduce_buf_6, 1.5*((sizeof(double)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));
    BUF_ALLOC(PdV_reduce_buf, 1.5*((sizeof(int)*reduced_cells)/(LOCAL_X*LOCAL_Y*LOCAL_Z)));

    // set initial (ideal) size for buffers and increment untl it hits alignment
    size_t lr_mpi_buf_sz = sizeof(double)*(y_max + 2*halo_allocate_depth)*(z_max + 2*halo_allocate_depth)*halo_allocate_depth;
    size_t bt_mpi_buf_sz = sizeof(double)*(x_max + 2*halo_allocate_depth)*(z_max + 2*halo_allocate_depth)*halo_allocate_depth;
    size_t fb_mpi_buf_sz = sizeof(double)*(x_max + 2*halo_allocate_depth)*(y_max + 2*halo_allocate_depth)*halo_allocate_depth;

    // enough for 1 for each array - overkill, but not that much extra space
    BUF_ALLOC(left_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(right_buffer, NUM_BUFFERED_FIELDS*lr_mpi_buf_sz);
    BUF_ALLOC(bottom_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);
    BUF_ALLOC(top_buffer, NUM_BUFFERED_FIELDS*bt_mpi_buf_sz);
    BUF_ALLOC(back_buffer, NUM_BUFFERED_FIELDS*fb_mpi_buf_sz);
    BUF_ALLOC(front_buffer, NUM_BUFFERED_FIELDS*fb_mpi_buf_sz);

    fprintf(DBGOUT, "Buffers allocated\n");
}

