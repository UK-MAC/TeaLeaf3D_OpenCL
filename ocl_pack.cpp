#include "ocl_common.hpp"
#include <numeric>

extern "C" void ocl_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, host_buffer);
}

extern "C" void ocl_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, host_buffer);
}

void CloverChunk::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS],
 const int depth, const int face, const int pack,
 double * host_buffer)
{
    const int n_exchanged = std::accumulate(fields, fields + NUM_FIELDS, 0);

    if (n_exchanged < 1)
    {
        return;
    }

    // which buffer is being used for this operation
    cl::Buffer * device_buffer = NULL;

    switch (face)
    {
    case CHUNK_LEFT:
        device_buffer = &left_buffer;
        break;
    case CHUNK_RIGHT:
        device_buffer = &right_buffer;
        break;
    case CHUNK_BOTTOM:
        device_buffer = &bottom_buffer;
        break;
    case CHUNK_TOP:
        device_buffer = &top_buffer;
        break;
    case CHUNK_BACK:
        device_buffer = &back_buffer;
        break;
    case CHUNK_FRONT:
        device_buffer = &front_buffer;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    cl::Kernel * pack_kernel = NULL;

    // set which kernel to call
    if (pack)
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &pack_left_buffer_device;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &pack_right_buffer_device;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &pack_bottom_buffer_device;
            break;
        case CHUNK_TOP:
            pack_kernel = &pack_top_buffer_device;
            break;
        case CHUNK_BACK:
            pack_kernel = &pack_back_buffer_device;
            break;
        case CHUNK_FRONT:
            pack_kernel = &pack_front_buffer_device;
            break;
        default:
            DIE("Invalid face identifier %d passed to pack\n", face);
        }
    }
    else
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &unpack_left_buffer_device;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &unpack_right_buffer_device;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &unpack_bottom_buffer_device;
            break;
        case CHUNK_TOP:
            pack_kernel = &unpack_top_buffer_device;
            break;
        case CHUNK_BACK:
            pack_kernel = &unpack_back_buffer_device;
            break;
        case CHUNK_FRONT:
            pack_kernel = &unpack_front_buffer_device;
            break;
        default:
            DIE("Invalid face identifier %d passed to unpack\n", face);
        }
    }

    pack_kernel->setArg(3, *device_buffer);
    pack_kernel->setArg(4, depth);

    // size of this buffer
    size_t side_size = 0;
    // reuse the halo update kernels sizes to launch packing kernels
    cl::NDRange pack_global, pack_local, pack_offset;

    switch (face)
    {
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        side_size = lr_mpi_buf_sz;
        pack_global = update_lr_global_size[depth];
        pack_local = update_lr_local_size[depth];
        pack_offset = update_lr_offset[depth];
        break;
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        side_size = bt_mpi_buf_sz;
        pack_global = update_bt_global_size[depth];
        pack_local = update_bt_local_size[depth];
        pack_offset = update_bt_offset[depth];
        break;
    case CHUNK_BACK:
    case CHUNK_FRONT:
        side_size = fb_mpi_buf_sz;
        pack_global = update_fb_global_size[depth];
        pack_local = update_fb_local_size[depth];
        pack_offset = update_fb_offset[depth];
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    if (!pack)
    {
        queue.enqueueWriteBuffer(*device_buffer, CL_TRUE, 0,
            n_exchanged*side_size*sizeof(double),
            host_buffer);
    }

    for (int ii = 0; ii < NUM_FIELDS; ii++)
    {
        int which_field = ii+1;

        if (fields[ii])
        {
            if (offsets[ii] < 0 || offsets[ii] > NUM_FIELDS*side_size)
            {
                DIE("Tried to pack/unpack field %d but invalid offset %d given\n",
                    ii, offsets[ii]);
            }

            int x_inc = 0, y_inc = 0, z_inc = 0;

            // set x/y/z inc for array
            switch (which_field)
            {
            case FIELD_density:
            case FIELD_energy0:
            case FIELD_energy1:
            case FIELD_u:
            case FIELD_p:
            case FIELD_sd:
            case FIELD_r:
                break;
            default:
                DIE("Invalid field number %d in choosing _inc values\n", which_field);
            }

            #define CASE_BUF(which_array)   \
            case FIELD_##which_array:       \
            {                               \
                device_array = &which_array;\
            }

            cl::Buffer * device_array = NULL;

            switch (which_field)
            {
            CASE_BUF(density); break;
            CASE_BUF(energy0); break;
            CASE_BUF(energy1); break;
            CASE_BUF(u); break;
            CASE_BUF(vector_p); break;
            CASE_BUF(vector_sd); break;
            CASE_BUF(vector_r); break;
            default:
                DIE("Invalid face %d passed to left/right pack buffer\n", which_field);
            }

            #undef CASE_BUF

            // set args + launch kernel
            pack_kernel->setArg(0, x_inc);
            pack_kernel->setArg(1, y_inc);
            pack_kernel->setArg(2, *device_array);
            pack_kernel->setArg(5, offsets[ii]);

            enqueueKernel(*pack_kernel, __LINE__, __FILE__,
                          pack_offset,
                          pack_global,
                          pack_local);
        }
    }

    if (pack)
    {
        queue.finish();
        queue.enqueueReadBuffer(*device_buffer, CL_TRUE, 0,
            n_exchanged*side_size*sizeof(double),
            host_buffer);
    }
}

