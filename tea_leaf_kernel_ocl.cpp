#include "ocl_common.hpp"

extern CloverChunk chunk;

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// Chebyshev solver
extern "C" void tea_leaf_kernel_cheby_copy_u_ocl_
(void)
{
    chunk.tea_leaf_cheby_copy_u();
}

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_cheby_init_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * rz, const double * theta, double* error)
{
    chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *rz, *theta, error);
}

extern "C" void tea_leaf_kernel_cheby_iterate_ocl_
(const double * ch_alphas, const double * ch_betas, int *n_coefs,
 const double * rx, const double * ry, const double * rz, const int * cheby_calc_step)
{
    chunk.tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *rz, *cheby_calc_step);
}

void CloverChunk::tea_leaf_cheby_copy_u
(void)
{
    // copy into u0/work_array_3 for later residual check
    queue.finish();
    queue.enqueueCopyBuffer(u, u0, 0, 0, (x_max+4) * (y_max+4) * sizeof(double));
}

void CloverChunk::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        tea_leaf_cheby_solve_calc_resid_device.setArg(0, u0);
    }
    else if (norm_array == 1)
    {
        // norm of r
        tea_leaf_cheby_solve_calc_resid_device.setArg(0, work_array_2);
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be [1, 2]", norm_array);
    }

    //ENQUEUE(tea_leaf_cheby_solve_calc_resid_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_resid_device);
    *norm = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

void CloverChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double rz, const double theta, double* error)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);
    tea_leaf_cheby_solve_calc_p_device.setArg(9, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(10, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(11, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(12, ry);
    tea_leaf_cheby_solve_calc_p_device.setArg(13, rz);

    tea_leaf_cheby_solve_init_p_device.setArg(9, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(10, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(11, ry);
    tea_leaf_cheby_solve_init_p_device.setArg(12, rz);

    //ENQUEUE(tea_leaf_cheby_solve_init_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_init_p_device);

    //ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

void CloverChunk::tea_leaf_kernel_cheby_iterate
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double rz, const int cheby_calc_step)
{
    tea_leaf_cheby_solve_calc_p_device.setArg(14, cheby_calc_step-1);

    //ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
    //ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rz, double * rro)
{
    chunk.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rz, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_w_
(const double * rx, const double * ry, double * rz, double * pw)
{
    chunk.tea_leaf_kernel_cg_calc_w(*rx, *ry, *rz, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_ur_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_p_
(double * beta)
{
    chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

// copy back dx/dy and calculate rx/ry
void CloverChunk::calcrxryrz
(double dt, double * rx, double * ry, double * rz)
{
    static int initd = 0;
    if (!initd)
    {
        // make sure intialise chunk has finished
        queue.finish();
        // celldx doesnt change after that so check once
        initd = 1;
    }

    double dx, dy, dz;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*x_min, sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*y_min, sizeof(double), &dy);
        queue.enqueueReadBuffer(celldz, CL_TRUE,
            sizeof(double)*z_min, sizeof(double), &dz);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
    *rz = dt/(dz*dz);
}

/********************/
#include <cassert>

void CloverChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rz, double * rro)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV);

    calcrxryrz(dt, rx, ry, rz);

    // only needs to be set once
    tea_leaf_cg_solve_calc_w_device.setArg(6, *rx);
    tea_leaf_cg_solve_calc_w_device.setArg(7, *ry);
    tea_leaf_cg_solve_calc_w_device.setArg(8, *rz);
    tea_leaf_cg_init_others_device.setArg(9, *rx);
    tea_leaf_cg_init_others_device.setArg(10, *ry);
    tea_leaf_cg_init_others_device.setArg(11, *rz);
    tea_leaf_init_diag_device.setArg(3, *rx);
    tea_leaf_init_diag_device.setArg(4, *ry);
    tea_leaf_init_diag_device.setArg(5, *rz);

    // copy u, get density value modified by coefficient
    tea_leaf_cg_init_u_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_cg_init_u_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_u_device);

    // init Kx, Ky
    //ENQUEUE(tea_leaf_cg_init_directions_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_directions_device);

    // premultiply Kx/Ky
    //ENQUEUE(tea_leaf_init_diag_device);
    ENQUEUE_OFFSET(tea_leaf_init_diag_device);

    // get initial guess in w, r, etc
    //ENQUEUE(tea_leaf_cg_init_others_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_others_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double rz, double* pw)
{
    //ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    //ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);
    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

void CloverChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    //ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

/********************/

// jacobi solver functions
extern "C" void tea_leaf_kernel_init_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rz)
{
    chunk.tea_leaf_init_jacobi(*coefficient, *dt, rx, ry, rz);
}

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * rz, double * error)
{
    chunk.tea_leaf_kernel_jacobi(*rx, *ry, *rz, error);
}

// jacobi
void CloverChunk::tea_leaf_init_jacobi
(int coefficient, double dt, double * rx, double * ry, double * rz)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxryrz(dt, rx, ry, rz);

    tea_leaf_jacobi_init_device.setArg(7, coefficient);
    //ENQUEUE(tea_leaf_jacobi_init_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_init_device);

    tea_leaf_jacobi_solve_device.setArg(0, *rx);
    tea_leaf_jacobi_solve_device.setArg(1, *ry);
    tea_leaf_jacobi_solve_device.setArg(2, *rz);
}

void CloverChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double rz, double* error)
{
    //ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    //ENQUEUE(tea_leaf_jacobi_solve_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
}

/********************/

// used by both
extern "C" void tea_leaf_kernel_finalise_ocl_
(void)
{
    chunk.tea_leaf_finalise();
}

// both
void CloverChunk::tea_leaf_finalise
(void)
{
    //ENQUEUE(tea_leaf_finalise_device);
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}

