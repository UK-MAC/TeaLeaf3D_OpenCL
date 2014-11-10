#ifndef __CL_COMMON_HDR
#define __CL_COMMON_HDR

#include "CL/cl.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>

// 2 dimensional arrays - use a 2D tile for local group
const static size_t LOCAL_X = 64;
const static size_t LOCAL_Y = 2;
const static size_t LOCAL_Z = 1;
const static cl::NDRange local_group_size(LOCAL_X, LOCAL_Y,LOCAL_Z);


// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density       1
#define FIELD_energy0       2
#define FIELD_energy1       3
#define FIELD_u             4
#define FIELD_p             5
#define FIELD_sd            6
#define NUM_FIELDS          6
#define FIELD_work_array_1 FIELD_p
#define FIELD_work_array_8 FIELD_sd

#define NUM_BUFFERED_FIELDS 5

// which side to pack - keep the same as in fortran file
#define CHUNK_LEFT 1
#define CHUNK_left 1
#define CHUNK_RIGHT 2
#define CHUNK_right 2
#define CHUNK_BOTTOM 3
#define CHUNK_bottom 3
#define CHUNK_TOP 4
#define CHUNK_top 4
#define CHUNK_BACK 5
#define CHUNK_back 5
#define CHUNK_FRONT 6
#define CHUNK_front 6
#define EXTERNAL_FACE       (-1)

#define CELL_DATA   1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4
#define Z_FACE_DATA 5

typedef struct cell_info {
    const int x_extra;
    const int y_extra;
    const int z_extra;
    const int x_invert;
    const int y_invert;
    const int z_invert;
    const int x_face;
    const int y_face;
    const int z_face;
    const int grid_type;

    cell_info
    (int in_x_extra, int in_y_extra,int in_z_extra,
    int in_x_invert, int in_y_invert,int in_z_invert,
    int in_x_face, int in_y_face,int in_z_face,
    int in_grid_type)
    :x_extra(in_x_extra), y_extra(in_y_extra),z_extra(in_z_extra),
    x_invert(in_x_invert), y_invert(in_y_invert),z_invert(in_z_invert),
    x_face(in_x_face), y_face(in_y_face),z_face(in_z_face),
    grid_type(in_grid_type)
    {
        ;
    }

} cell_info_t;

// reductions
typedef struct red_t {
    cl::Kernel kernel;
    cl::NDRange global_size;
    cl::NDRange local_size;
} reduce_kernel_info_t;

// vectors of kernels and work group sizes for a specific reduction
typedef std::vector<reduce_kernel_info_t> reduce_info_vec_t;

class CloverChunk
{
private:
    // kernels
    cl::Kernel set_field_device;
    cl::Kernel field_summary_device;

    cl::Kernel generate_chunk_device;
    cl::Kernel generate_chunk_init_device;

    cl::Kernel initialise_chunk_first_device;
    cl::Kernel initialise_chunk_second_device;

    // halo updates
    cl::Kernel update_halo_top_device;
    cl::Kernel update_halo_bottom_device;
    cl::Kernel update_halo_left_device;
    cl::Kernel update_halo_right_device;
    cl::Kernel update_halo_back_device;
    cl::Kernel update_halo_front_device;
    // mpi packing
    cl::Kernel pack_left_buffer_device;
    cl::Kernel unpack_left_buffer_device;
    cl::Kernel pack_right_buffer_device;
    cl::Kernel unpack_right_buffer_device;
    cl::Kernel pack_bottom_buffer_device;
    cl::Kernel unpack_bottom_buffer_device;
    cl::Kernel pack_top_buffer_device;
    cl::Kernel unpack_top_buffer_device;
    cl::Kernel pack_back_buffer_device;
    cl::Kernel unpack_back_buffer_device;
    cl::Kernel pack_front_buffer_device;
    cl::Kernel unpack_front_buffer_device;

    // main buffers, with sub buffers for each offset
    cl::Buffer left_buffer;
    cl::Buffer right_buffer;
    cl::Buffer bottom_buffer;
    cl::Buffer top_buffer;
    cl::Buffer back_buffer;
    cl::Buffer front_buffer;

    #define TEA_ENUM_JACOBI     1
    #define TEA_ENUM_CG         2
    #define TEA_ENUM_CHEBYSHEV  3
    #define TEA_ENUM_PPCG       4
    int tea_solver;

    // tea leaf
    cl::Kernel tea_leaf_cg_init_u_device;
    cl::Kernel tea_leaf_cg_init_directions_device;
    cl::Kernel tea_leaf_cg_init_others_device;
    cl::Kernel tea_leaf_cg_solve_calc_w_device;
    cl::Kernel tea_leaf_cg_solve_calc_ur_device;
    cl::Kernel tea_leaf_cg_solve_calc_p_device;
    cl::Buffer z;

    // chebyshev solver
    cl::Kernel tea_leaf_cheby_solve_init_p_device;
    cl::Kernel tea_leaf_cheby_solve_calc_u_device;
    cl::Kernel tea_leaf_cheby_solve_calc_p_device;
    cl::Kernel tea_leaf_cheby_calc_2norm_device;

    cl::Kernel tea_leaf_ppcg_solve_init_sd_device;
    cl::Kernel tea_leaf_ppcg_solve_calc_sd_device;
    cl::Kernel tea_leaf_ppcg_solve_update_r_device;
    cl::Kernel tea_leaf_ppcg_solve_init_p_device;

    // used to hold the alphas/beta used in chebyshev solver - different from CG ones!
    cl::Buffer ch_alphas_device, ch_betas_device;

    // need more for the Kx/Ky arrays
    cl::Kernel tea_leaf_jacobi_init_device;
    cl::Kernel tea_leaf_jacobi_copy_u_device;
    cl::Kernel tea_leaf_jacobi_solve_device;

    cl::Buffer u, u0;
    cl::Kernel tea_leaf_finalise_device;
    // TODO could be used by all - precalculate diagonal + scale Kx/Ky
    cl::Kernel tea_leaf_init_diag_device;
    cl::Kernel tea_leaf_calc_residual_device;

    // tolerance specified in tea.in
    float tolerance;
    // whether preconditioner is enabled in input file
    bool preconditioner_on;

    // calculate rx/ry to pass back to fortran
    void calcrxryrz
    (double dt, double * rx, double * ry, double * rz);

    // specific sizes and launch offsets for different kernels
    typedef struct {
        cl::NDRange global;
        cl::NDRange offset;
    } launch_specs_t;
    std::map< std::string, launch_specs_t > launch_specs;

    // reduction kernels - need multiple levels
    reduce_info_vec_t min_red_kernels_double;
    reduce_info_vec_t max_red_kernels_double;
    reduce_info_vec_t sum_red_kernels_double;
    // for PdV
    reduce_info_vec_t max_red_kernels_int;

    // ocl things
    cl::CommandQueue queue;
    cl::Platform platform;
    cl::Device device;
    cl::Context context;

    // for passing into kernels for changing operation based on device type
    std::string device_type_prepro;

    // buffers
    cl::Buffer density;
    cl::Buffer energy0;
    cl::Buffer energy1;
    cl::Buffer volume;

    cl::Buffer cellx;
    cl::Buffer celly;
    cl::Buffer cellz;
    cl::Buffer celldx;
    cl::Buffer celldy;
    cl::Buffer celldz;
    cl::Buffer vertexx;
    cl::Buffer vertexy;
    cl::Buffer vertexz;
    cl::Buffer vertexdx;
    cl::Buffer vertexdy;
    cl::Buffer vertexdz;

    cl::Buffer xarea;
    cl::Buffer yarea;
    cl::Buffer zarea;

    // generic work arrays
    cl::Buffer work_array_1;
    cl::Buffer work_array_2;
    cl::Buffer work_array_3;
    cl::Buffer work_array_4;
    cl::Buffer work_array_5;
    cl::Buffer work_array_6;
    cl::Buffer work_array_7;
    cl::Buffer work_array_8;

    // for reduction in PdV
    cl::Buffer PdV_reduce_buf;

    // for reduction in field_summary
    cl::Buffer reduce_buf_1;
    cl::Buffer reduce_buf_2;
    cl::Buffer reduce_buf_3;
    cl::Buffer reduce_buf_4;
    cl::Buffer reduce_buf_5;
    cl::Buffer reduce_buf_6;

    // global size for kernels
    cl::NDRange global_size;
    // total number of cells
    size_t total_cells;
    // number of cells reduced
    size_t reduced_cells;

    // sizes for launching update halo kernels - l/r and u/d updates
    cl::NDRange update_lr_global_size[2];
    cl::NDRange update_ud_global_size[2];
    cl::NDRange update_fb_global_size[2];
    cl::NDRange update_lr_local_size[2];
    cl::NDRange update_ud_local_size[2];
    cl::NDRange update_fb_local_size[2];

    // values used to control operation
    size_t x_min;
    size_t x_max;
    size_t y_min;
    size_t y_max;
    size_t z_min;
    size_t z_max;
    // mpi rank
    int rank;

    // size of mpi buffers
    size_t lr_mpi_buf_sz, bt_mpi_buf_sz, fb_mpi_buf_sz;

    // desired type for opencl
    int desired_type;

    // if profiling
    int profiler_on;
    // for recording times if profiling is on
    std::map<std::string, double> kernel_times;
    // recording number of times each kernel was called
    std::map<std::string, int> kernel_calls;

    // Where to send debug output
    FILE* DBGOUT;

    // compile a file and the contained kernels, and check for errors
    void compileKernel
    (const std::string& options,
     const std::string& source_name,
     const char* kernel_name,
     cl::Kernel& kernel);
    cl::Program compileProgram
    (const std::string& source,
     const std::string& options);
    // keep track of built programs to avoid rebuilding them
    std::map<std::string, cl::Program> built_programs;
    std::vector<double> dumpArray
    (const std::string& arr_name, int x_extra, int y_extra, int z_extra);
    std::map<std::string, cl::Buffer> arr_names;

    /*
     *  initialisation subroutines
     */

    // initialise context, queue, etc
    void initOcl
    (void);
    // initialise all program stuff, kernels, etc
    void initProgram
    (void);
    // intialise local/global sizes
    void initSizes
    (void);
    // initialise buffers for device
    void initBuffers
    (void);
    // initialise all the arguments for each kernel
    void initArgs
    (void);
    // create reduction kernels
    void initReduction
    (void);

    // this function gets called when something goes wrong
    #define DIE(...) cloverDie(__LINE__, __FILE__, __VA_ARGS__)
    void cloverDie
    (int line, const char* filename, const char* format, ...);

public:
    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* temp);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_zmin, const double* state_zmax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,double d_zmin,
        double d_dx, double d_dy, double d_dz);

    void update_halo_kernel(const int* fields, int depth, const int* chunk_neighbours);
    void update_array
    (cl::Buffer& cur_array,
    const cell_info_t& array_type,
    const int* chunk_neighbours,
    int depth);

    void set_field_kernel();

    // Tea leaf
    void tea_leaf_init_jacobi(int, double, double*, double*, double*);
    void tea_leaf_kernel_jacobi(double, double, double, double*);

    void tea_leaf_init_cg(int, double, double*, double*, double*, double*);
    void tea_leaf_kernel_cg_calc_w(double rx, double ry, double rz, double* pw);
    void tea_leaf_kernel_cg_calc_ur(double alpha, double* rrn);
    void tea_leaf_kernel_cg_calc_p(double beta);

    void tea_leaf_cheby_copy_u
    (void);
    void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm);
    void tea_leaf_kernel_cheby_init
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double rz, const double theta, double* error);
    void tea_leaf_kernel_cheby_iterate
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double rz, const int cheby_calc_steps);

    void ppcg_init(const double * ch_alphas, const double * ch_betas,
        const double theta, const int n);
    void ppcg_init_sd();
    void ppcg_init_p(double * rro);
    void ppcg_inner(int);

    void tea_leaf_finalise();
    void tea_leaf_calc_residual(void);

    // ctor
    CloverChunk
    (void);
    CloverChunk
    (int* in_x_min, int* in_x_max,
     int* in_y_min, int* in_y_max,
     int* in_z_min, int* in_z_max,
     int* in_profiler_on);
    // dtor
    ~CloverChunk
    (void);

    // enqueue a kernel
    void enqueueKernel

    (cl::Kernel const& kernel,
     int line, const char* file,
     const cl::NDRange offset,
     const cl::NDRange global_range,
     const cl::NDRange local_range,
     const std::vector< cl::Event > * const events=NULL,
     cl::Event * const event=NULL) ;

    #define ENQUEUE_OFFSET(knl)                        \
        enqueueKernel(knl, __LINE__, __FILE__,         \
                      launch_specs.at(#knl).offset,    \
                      launch_specs.at(#knl).global,    \
                      local_group_size);

    // reduction
    template <typename T>
    T reduceValue
    (reduce_info_vec_t& red_kernels,
     const cl::Buffer& results_buf);

    void packUnpackAllBuffers
    (int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int depth,
     int face, int pack, double * buffer);
};

extern CloverChunk chunk;

class KernelCompileError : std::exception
{
private:
    const std::string _err;
public:
    KernelCompileError(const char* err):_err(err){}
    ~KernelCompileError() throw(){}
    const char* what() const throw() {return this->_err.c_str();}
};

#endif
