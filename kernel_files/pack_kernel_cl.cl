#include "./kernel_files/macros_cl.cl"

#if 1
#define VERT_IDX                                                    \
    ((column - 0) +                                                 \
    ((row    - 1) + depth - 1)*depth +                              \
    ((slice  - 1) + depth - 1)*(y_max + y_extra + 2*depth)*depth)   \
    + offset
#else
#define VERT_IDX                                            \
    (slice  + depth - 1 +                                                           \
    (row    + depth - 1)* (x_max + x_extra + 2*depth) +                             \
    (column         - 1)*((z_max + z_extra + 2*depth)*(y_max + y_extra + 2*depth)))
#endif

// bottom/top
#define HORZ_IDX                                                                    \
    (column + depth - 1 +                                                           \
    (slice  + depth - 0)* (x_max + x_extra + 2*depth) +                             \
    (row            - 1)*((x_max + x_extra + 2*depth)*(z_max + z_extra + 2*depth))) \
    + offset

// back/front
#define DEPTH_IDX                                                                   \
    ((row    - 1) + depth +                                                           \
    ((column - 1) + depth)* (x_max + x_extra + 2*depth) +                             \
    ((slice  - 0)        )*((x_max + x_extra + 2*depth)*(y_max + y_extra + 2*depth))) \
    + offset

__kernel void pack_left_buffer
(int x_extra, int y_extra, int z_extra,
const  __global double * __restrict cur_array,
       __global double * __restrict left_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t src = 1 + (HALO_DEPTH - column - 1)*2;
        left_buffer[VERT_IDX] = cur_array[THARR3D(src, 0, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_left_buffer
(int x_extra, int y_extra, int z_extra,
       __global double * __restrict cur_array,
const  __global double * __restrict left_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t dst = 0;
        cur_array[THARR3D(dst, 0, 0, x_extra, y_extra)] = left_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_right_buffer
(int x_extra, int y_extra, int z_extra,
const  __global double * __restrict cur_array,
       __global double * __restrict right_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t src = x_max + x_extra;
        right_buffer[VERT_IDX] = cur_array[THARR3D(src, 0, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_right_buffer
(int x_extra, int y_extra, int z_extra,
       __global double * __restrict cur_array,
const  __global double * __restrict right_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    {
        const size_t dst = x_max + x_extra + (HALO_DEPTH - column - 1)*2 + 1;
        cur_array[THARR3D(dst, 0, 0, x_extra, y_extra)] = right_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_bottom_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 __global double * __restrict bottom_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = 1 + (HALO_DEPTH - row - 1)*2;
        bottom_buffer[HORZ_IDX] = cur_array[THARR3D(0, src, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_bottom_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 __global double * __restrict bottom_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t dst = 0;
        cur_array[THARR3D(0, dst, 0, x_extra, y_extra)] = bottom_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_top_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 __global double * __restrict top_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = y_max + y_extra;
        top_buffer[HORZ_IDX] = cur_array[THARR3D(0, src, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_top_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 __global double * __restrict top_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (slice >= HALO_DEPTH - depth && slice <= (z_max + HALO_DEPTH - 1) + z_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t dst = y_max + y_extra + (HALO_DEPTH - row - 1)*2 + 1;
        cur_array[THARR3D(0, dst, 0, x_extra, y_extra)] = top_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_back_buffer
(int x_extra, int y_extra, int z_extra,
 const __global double * __restrict cur_array,
 __global double * __restrict back_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = 1 + (HALO_DEPTH - row - 1)*2;
        back_buffer[DEPTH_IDX] = cur_array[THARR3D(0, 0, src, x_extra, y_extra)];
    }
}

__kernel void unpack_back_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 const __global double * __restrict back_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t dst = 0;
        cur_array[THARR3D(0, 0, dst, x_extra, y_extra)] = back_buffer[DEPTH_IDX];
    }
}

/************************************************************/

__kernel void pack_front_buffer
(int x_extra, int y_extra, int z_extra,
 const __global double * __restrict cur_array,
 __global double * __restrict front_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t src = z_max + z_extra;
        front_buffer[DEPTH_IDX] = cur_array[THARR3D(0, 0, src, x_extra, y_extra)];
    }
}

__kernel void unpack_front_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict cur_array,
 const __global double * __restrict front_buffer,
const int depth, int offset)
{
    __kernel_indexes;

    if (row >= HALO_DEPTH - depth && row <= (y_max + HALO_DEPTH - 1) + y_extra + depth)
    if (column >= HALO_DEPTH - depth && column <= (x_max + HALO_DEPTH - 1) + x_extra + depth)
    {
        const size_t dst = z_max + z_extra + (HALO_DEPTH - slice - 1)*2 + 1;
        cur_array[THARR3D(0, 0, dst, x_extra, y_extra)] = front_buffer[DEPTH_IDX];
    }
}
