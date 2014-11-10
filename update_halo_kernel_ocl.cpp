#include "ocl_common.hpp"

// types of array data
const static cell_info_t CELL(    0, 0, 0, 1, 1, 1, 0, 0,0, CELL_DATA);

extern "C" void update_halo_kernel_ocl_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void CloverChunk::update_array
(cl::Buffer& cur_array,
const cell_info_t& array_type,
const int* chunk_neighbours,
int depth)
{
    // could do clenqueuecopybufferrect, but it's blocking and would be slow

    // could do offset launch for updating bottom/right, but dont to keep parity with cuda
    #define CHECK_LAUNCH(face, dir) \
    if(chunk_neighbours[CHUNK_ ## face - 1] == EXTERNAL_FACE)\
    {\
        update_halo_##face##_device.setArg(0, array_type.x_extra); \
        update_halo_##face##_device.setArg(1, array_type.y_extra); \
        update_halo_##face##_device.setArg(2, array_type.z_extra); \
        update_halo_##face##_device.setArg(3, array_type.x_invert); \
        update_halo_##face##_device.setArg(4, array_type.y_invert); \
        update_halo_##face##_device.setArg(5, array_type.z_invert); \
        update_halo_##face##_device.setArg(6, array_type.x_face); \
        update_halo_##face##_device.setArg(7, array_type.y_face); \
        update_halo_##face##_device.setArg(8, array_type.z_face); \
        update_halo_##face##_device.setArg(9, array_type.grid_type); \
        update_halo_##face##_device.setArg(10, depth); \
        update_halo_##face##_device.setArg(11, cur_array); \
        CloverChunk::enqueueKernel(update_halo_##face##_device, \
            __LINE__, __FILE__, \
            cl::NullRange, \
            update_##dir##_global_size[depth-1], \
            update_##dir##_local_size[depth-1]); \
    }

    CHECK_LAUNCH(back, fb)
    CHECK_LAUNCH(front, fb)
    CHECK_LAUNCH(bottom, ud)
    CHECK_LAUNCH(top, ud)
    CHECK_LAUNCH(left, lr)
    CHECK_LAUNCH(right, lr)
}

void CloverChunk::update_halo_kernel
(const int* fields,
const int depth,
const int* chunk_neighbours)
{
    #define HALO_UPDATE_RESIDENT(arr, type)                 \
    if(fields[FIELD_ ## arr - 1] == 1)                      \
    {                                                       \
        update_array(arr, type, chunk_neighbours, depth);   \
    }

    HALO_UPDATE_RESIDENT(density, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);

    HALO_UPDATE_RESIDENT(u, CELL);
    HALO_UPDATE_RESIDENT(work_array_1, CELL);
    HALO_UPDATE_RESIDENT(work_array_8, CELL);
}
