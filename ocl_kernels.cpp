#include "ocl_common.hpp"
#include <sstream>
#include <fstream>

void CloverChunk::initProgram
(void)
{
    // options
    std::stringstream options("");

#ifdef __arm__
    // on ARM, don't use built in functions as they don't exist
    options << "-DCLOVER_NO_BUILTINS ";
#endif

    if (preconditioner_on)
    {
        // use jacobi preconditioner when running CG solver
        options << "-DUSE_PRECONDITIONER ";
    }

    // pass in these values so you don't have to pass them in to every kernel
    options << "-Dx_min=" << x_min << " ";
    options << "-Dx_max=" << x_max << " ";
    options << "-Dy_min=" << y_min << " ";
    options << "-Dy_max=" << y_max << " ";
    options << "-Dz_min=" << z_min << " ";
    options << "-Dz_max=" << z_max << " ";

    // local sizes
    options << "-DBLOCK_SZ=" << LOCAL_X*LOCAL_Y*LOCAL_Z << " ";
    options << "-DLOCAL_X=" << LOCAL_X << " ";
    options << "-DLOCAL_Y=" << LOCAL_Y << " ";
    options << "-DLOCAL_Z=" << LOCAL_Z << " ";

    // for update halo
    options << "-DCELL_DATA=" << CELL_DATA << " ";
    options << "-DVERTEX_DATA=" << VERTEX_DATA << " ";
    options << "-DX_FACE_DATA=" << X_FACE_DATA << " ";
    options << "-DY_FACE_DATA=" << Y_FACE_DATA << " ";
    options << "-DZ_FACE_DATA=" << Z_FACE_DATA << " ";

    // include current directory
    options << "-I. ";

    // device type in the form "-D..."
    options << device_type_prepro;

    const std::string options_str = options.str();

    if (!rank)
    {
        fprintf(DBGOUT, "Compiling kernels with options:\n%s\n", options_str.c_str());
        fprintf(stdout, "Compiling kernels (may take some time)...");
        fflush(stdout);
    }

    compileKernel(options_str, "./kernel_files/initialise_chunk_cl.cl", "initialise_chunk_first", initialise_chunk_first_device);
    compileKernel(options_str, "./kernel_files/initialise_chunk_cl.cl", "initialise_chunk_second", initialise_chunk_second_device);
    compileKernel(options_str, "./kernel_files/generate_chunk_cl.cl", "generate_chunk_init", generate_chunk_init_device);
    compileKernel(options_str, "./kernel_files/generate_chunk_cl.cl", "generate_chunk", generate_chunk_device);

    compileKernel(options_str, "./kernel_files/set_field_cl.cl", "set_field", set_field_device);

    compileKernel(options_str, "./kernel_files/field_summary_cl.cl", "field_summary", field_summary_device);

    compileKernel(options_str, "./kernel_files/update_halo_cl.cl", "update_halo_top", update_halo_top_device);
    compileKernel(options_str, "./kernel_files/update_halo_cl.cl", "update_halo_bottom", update_halo_bottom_device);
    compileKernel(options_str, "./kernel_files/update_halo_cl.cl", "update_halo_left", update_halo_left_device);
    compileKernel(options_str, "./kernel_files/update_halo_cl.cl", "update_halo_right", update_halo_right_device);

    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_left_buffer", pack_left_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_left_buffer", unpack_left_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_right_buffer", pack_right_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_right_buffer", unpack_right_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_bottom_buffer", pack_bottom_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_bottom_buffer", unpack_bottom_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_top_buffer", pack_top_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_top_buffer", unpack_top_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_back_buffer", pack_back_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_back_buffer", unpack_back_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "pack_front_buffer", pack_front_buffer_device);
    compileKernel(options_str, "./kernel_files/pack_kernel_cl.cl", "unpack_front_buffer", unpack_front_buffer_device);

    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_init_u", tea_leaf_cg_init_u_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_init_directions", tea_leaf_cg_init_directions_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_init_others", tea_leaf_cg_init_others_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_w", tea_leaf_cg_solve_calc_w_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_ur", tea_leaf_cg_solve_calc_ur_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_cg_cl.cl", "tea_leaf_cg_solve_calc_p", tea_leaf_cg_solve_calc_p_device);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            compileKernel(options_str, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_init_p", tea_leaf_cheby_solve_init_p_device);
            compileKernel(options_str, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_calc_u", tea_leaf_cheby_solve_calc_u_device);
            compileKernel(options_str, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_solve_calc_p", tea_leaf_cheby_solve_calc_p_device);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            compileKernel(options_str, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_init_sd", tea_leaf_ppcg_solve_init_sd_device);
            compileKernel(options_str, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_calc_sd", tea_leaf_ppcg_solve_calc_sd_device);
            compileKernel(options_str, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_update_r", tea_leaf_ppcg_solve_update_r_device);
            compileKernel(options_str, "./kernel_files/tea_leaf_ppcg_cl.cl", "tea_leaf_ppcg_solve_init_p", tea_leaf_ppcg_solve_init_p_device);
        }
    }
    else
    {
        compileKernel(options_str, "./kernel_files/tea_leaf_jacobi_cl.cl", "tea_leaf_jacobi_init", tea_leaf_jacobi_init_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_jacobi_cl.cl", "tea_leaf_jacobi_copy_u", tea_leaf_jacobi_copy_u_device);
        compileKernel(options_str, "./kernel_files/tea_leaf_jacobi_cl.cl", "tea_leaf_jacobi_solve", tea_leaf_jacobi_solve_device);
    }

    compileKernel(options_str, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_init_diag", tea_leaf_init_diag_device);
    compileKernel(options_str, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_finalise", tea_leaf_finalise_device);
    compileKernel(options_str, "./kernel_files/tea_leaf_common_cl.cl", "tea_leaf_calc_residual", tea_leaf_calc_residual_device);
    compileKernel(options_str, "./kernel_files/tea_leaf_cheby_cl.cl", "tea_leaf_cheby_calc_2norm", tea_leaf_cheby_calc_2norm_device);

    if (!rank)
    {
        fprintf(stdout, "done.\n");
        fprintf(DBGOUT, "All kernels compiled\n");
    }
}

void CloverChunk::compileKernel
(const std::string& options_orig,
 const std::string& source_name,
 const char* kernel_name,
 cl::Kernel& kernel)
{
    std::string source_str;

    {
        std::ifstream ifile(source_name.c_str());
        source_str = std::string(
            (std::istreambuf_iterator<char>(ifile)),
            (std::istreambuf_iterator<char>()));
    }

    fprintf(DBGOUT, "Compiling %s...", kernel_name);
    cl::Program program;

#if defined(PHI_SOURCE_PROFILING)
    std::stringstream plusprof("");

    if (desired_type == CL_DEVICE_TYPE_ACCELERATOR)
    {
        plusprof << " -profiling ";
        plusprof << " -s \"" << source_name << "\"";
    }
    plusprof << options_orig;
    std::string options(plusprof.str());
#else
    std::string options(options_orig);
#endif

    if (built_programs.find(source_name + options) == built_programs.end())
    {
        try
        {
            program = compileProgram(source_str, options);
        }
        catch (KernelCompileError err)
        {
            DIE("Errors in compiling %s:\n%s\n", kernel_name, err.what());
        }

        built_programs[source_name + options] = program;
    }
    else
    {
        // + options to stop reduction kernels using the wrong types
        program = built_programs.at(source_name + options);
    }

    size_t max_wg_size;

    try
    {
        kernel = cl::Kernel(program, kernel_name);
    }
    catch (cl::Error e)
    {
        fprintf(DBGOUT, "Failed\n");
        DIE("Error %d (%s) in creating %s kernel\n",
            e.err(), e.what(), kernel_name);
    }
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(kernel(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_wg_size,
                                 NULL));
    if ((LOCAL_X*LOCAL_Y*LOCAL_Z) > max_wg_size)
    {
        DIE("Work group size %zux%zu is too big for kernel %s"
            " - maximum is %zu\n",
                LOCAL_X, LOCAL_Y, LOCAL_Z, kernel_name,
                max_wg_size);
    }

    fprintf(DBGOUT, "Done\n");
    fflush(DBGOUT);
}

cl::Program CloverChunk::compileProgram
(const std::string& source,
 const std::string& options)
{
    // catches any warnings/errors in the build
    std::stringstream errstream("");

    // very verbose
    //fprintf(stderr, "Making with source:\n%s\n", source.c_str());
    //fprintf(DBGOUT, "Making with options string:\n%s\n", options.c_str());
    fflush(DBGOUT);
    cl::Program program;

    cl::Program::Sources sources;
    sources = cl::Program::Sources(1, std::make_pair(source.c_str(), source.length()));

    try
    {
        program = cl::Program(context, sources);
        std::vector<cl::Device> dev_vec(1, device);
        program.build(dev_vec, options.c_str());
    }
    catch (cl::Error e)
    {
        fprintf(stderr, "Errors in creating program\n");

        try
        {
            errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        catch (cl::Error ie)
        {
            DIE("Error %d in retrieving build info\n", e.err());
        }

        std::string errs(errstream.str());
        //DIE("%s\n", errs.c_str());
        throw KernelCompileError(errs.c_str());
    }

    errstream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::string errs(errstream.str());

    // some will print out an empty warning log
    if (errs.size() > 10)
    {
        fprintf(DBGOUT, "Warnings:\n%s\n", errs.c_str());
    }

    return program;
}

void CloverChunk::initSizes
(void)
{
    fprintf(DBGOUT, "Local size = %zux%zux%zu\n", LOCAL_X, LOCAL_Y, LOCAL_Z);

    // pad the global size so the local size fits
    const size_t glob_x = x_max+5 +
        (((x_max+5)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max+5)%LOCAL_X)));
    const size_t glob_y = y_max+5 +
        (((y_max+5)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max+5)%LOCAL_Y)));
    const size_t glob_z = z_max+5 +
        (((z_max+5)%LOCAL_Z == 0) ? 0 : (LOCAL_Z - ((z_max+5)%LOCAL_Z)));
    total_cells = glob_x*glob_y*glob_z;

    fprintf(DBGOUT, "Global size = %zux%zux%zu\n", glob_x, glob_y, glob_z);
    global_size = cl::NDRange(glob_x, glob_y, glob_z);

    /*
     *  all the reductions only operate on the inner cells, because the halo
     *  cells aren't really part of the simulation. create a new global size
     *  that doesn't include these halo cells for the reduction which should
     *  speed it up a bit
     */
    const size_t red_x = x_max +
        (((x_max)%LOCAL_X == 0) ? 0 : (LOCAL_X - ((x_max)%LOCAL_X)));
    const size_t red_y = y_max +
        (((y_max)%LOCAL_Y == 0) ? 0 : (LOCAL_Y - ((y_max)%LOCAL_Y)));
    const size_t red_z = z_max +
        (((z_max)%LOCAL_Z == 0) ? 0 : (LOCAL_Z - ((z_max)%LOCAL_Z)));
    reduced_cells = red_x*red_y*red_z;

    /*
     *  update halo kernels need specific work group sizes - not doing a
     *  reduction, so can just fit it to the row/column even if its not a pwoer
     *  of 2
     */

    // get max local size for the update kernels
    // different directions are near identical, so should have the same limit
    size_t max_update_wg_sz;
    cl::detail::errHandler(
        clGetKernelWorkGroupInfo(update_halo_bottom_device(),
                                 device(),
                                 CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),
                                 &max_update_wg_sz,
                                 NULL));
    fprintf(DBGOUT, "Max work group size for update halo is %zu\n", max_update_wg_sz);

    // ideally multiple of 32 for nvidia, ideally multiple of 64 for amd
    size_t local_row_size = 16;
    size_t local_column_size = 16;
    size_t local_slice_size = 16;

    cl_device_type dtype;
    device.getInfo(CL_DEVICE_TYPE, &dtype);

    if (dtype == CL_DEVICE_TYPE_ACCELERATOR)
    {
        // want to run with work group size of 16 for phi to speed up l/r updates
        local_row_size = 4;
        local_column_size = 4;
        local_slice_size = 4;
    }

    // divide all local sizes if 16x16x1 is too big
    while (local_row_size * local_slice_size > max_update_wg_sz)
    {
        local_row_size /= 2;
        local_column_size /= 2;
        local_slice_size /= 2;
    }

    // divide one local dimension if 16x16x2 is still too big
    size_t local_divide = 1;
    while (local_row_size * local_slice_size * 2 / local_divide > max_update_wg_sz)
    {
        local_divide *= 2;
    }

    // create the local sizes, dividing the last possible dimension if needs be
    update_lr_local_size[0] = cl::NDRange(1, local_column_size, local_slice_size);
    update_lr_local_size[1] = cl::NDRange(2, local_column_size, local_slice_size/local_divide);
    update_ud_local_size[0] = cl::NDRange(local_row_size, 1, local_slice_size);
    update_ud_local_size[1] = cl::NDRange(local_row_size, 2, local_slice_size/local_divide);
    update_fb_local_size[0] = cl::NDRange(local_row_size, local_column_size, 1);
    update_fb_local_size[1] = cl::NDRange(local_row_size, local_column_size/local_divide, 2);

    // start off doing minimum amount of work
    size_t global_row_size = x_max + 5;
    size_t global_column_size = y_max + 5;
    size_t global_slice_size = z_max + 5;

    // increase just to fit in with local work group sizes
    while (global_row_size % local_row_size)
        global_row_size++;
    while (global_column_size % local_column_size)
        global_column_size++;
    while (global_slice_size % local_slice_size)
        global_slice_size++;

    // create ndranges
    update_lr_global_size[0] = cl::NDRange(1, global_column_size, global_slice_size);
    update_lr_global_size[1] = cl::NDRange(2, global_column_size, global_slice_size);
    update_ud_global_size[0] = cl::NDRange(global_row_size, 1, global_slice_size);
    update_ud_global_size[1] = cl::NDRange(global_row_size, 2, global_slice_size);
    update_fb_global_size[0] = cl::NDRange(global_row_size, global_column_size, 1);
    update_fb_global_size[1] = cl::NDRange(global_row_size, global_column_size, 2);

    for (int depth = 0; depth < 2; depth++)
    {
        fprintf(DBGOUT, "Depth %d:\n", depth + 1);
        fprintf(DBGOUT, "Left/right update halo size: [%zu %zu %zu] split by [%zu %zu %zu]\n",
            update_lr_global_size[depth][0], update_lr_global_size[depth][1], update_lr_global_size[depth][2],
            update_lr_local_size[depth][0], update_lr_local_size[depth][1], update_lr_local_size[depth][2]);
        fprintf(DBGOUT, "Bottom/top update halo size: [%zu %zu %zu] split by [%zu %zu %zu]\n",
            update_ud_global_size[depth][0], update_ud_global_size[depth][1], update_ud_global_size[depth][2],
            update_ud_local_size[depth][0], update_ud_local_size[depth][1], update_ud_local_size[depth][2]);
        fprintf(DBGOUT, "Front/back update halo size: [%zu %zu %zu] split by [%zu %zu %zu]\n",
            update_fb_global_size[depth][0], update_fb_global_size[depth][1], update_fb_global_size[depth][2],
            update_fb_local_size[depth][0], update_fb_local_size[depth][1], update_fb_local_size[depth][2]);
    }

    fprintf(DBGOUT, "Update halo parameters calculated\n");

    /*
     *  figure out offset launch sizes for the various kernels
     *  no 'smart' way to do this?
     */
    #define FIND_PADDING_SIZE(knl, smin, smax, vmin, vmax, hmin, hmax)  \
    {                                                                   \
        size_t global_horz_size = (-(hmin)) + (hmax) + x_max;           \
        while (global_horz_size % LOCAL_X) global_horz_size++;          \
        size_t global_vert_size = (-(vmin)) + (vmax) + y_max;           \
        while (global_vert_size % LOCAL_Y) global_vert_size++;          \
        size_t global_slic_size = (-(smin)) + (smax) + z_max;           \
        while (global_slic_size % LOCAL_Z) global_slic_size++;          \
        launch_specs_t cur_specs;                                       \
        cur_specs.global = cl::NDRange(global_horz_size,                \
            global_vert_size, global_slic_size);                        \
        cur_specs.offset = cl::NDRange((x_min + 1) + (hmin),            \
            (y_min + 1) + (vmin), (z_min + 1) + (smin));                \
        launch_specs[#knl"_device"] = cur_specs;                        \
    }

    FIND_PADDING_SIZE(set_field, 0, 0, 0, 0, 0, 0);
    FIND_PADDING_SIZE(field_summary, 0, 0, 0, 0, 0, 0);

    FIND_PADDING_SIZE(initialise_chunk_first, 0, 3, 0, 3, 0, 3);
    FIND_PADDING_SIZE(initialise_chunk_second, -2, 2, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk_init, -2, 2, -2, 2, -2, 2);
    FIND_PADDING_SIZE(generate_chunk, -2, 2, -2, 2, -2, 2);

    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        FIND_PADDING_SIZE(tea_leaf_cg_init_u, -2, 2, -2, 2, -2, 2);
        FIND_PADDING_SIZE(tea_leaf_cg_init_directions, 0, 1, 0, 1, 0, 1);
        FIND_PADDING_SIZE(tea_leaf_cg_init_others, 0, 0, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_w, 0, 0, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_ur, 0, 0, 0, 0, 0, 0);
        FIND_PADDING_SIZE(tea_leaf_cg_solve_calc_p, 0, 0, 0, 0, 0, 0);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            FIND_PADDING_SIZE(tea_leaf_cheby_solve_calc_u, 0, 0, 0, 0, 0, 0);
            FIND_PADDING_SIZE(tea_leaf_cheby_solve_calc_p, 0, 0, 0, 0, 0, 0);
            FIND_PADDING_SIZE(tea_leaf_cheby_solve_init_p, 0, 0, 0, 0, 0, 0);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            FIND_PADDING_SIZE(tea_leaf_ppcg_solve_init_sd, 0, 0, 0, 0, 0, 0);
            FIND_PADDING_SIZE(tea_leaf_ppcg_solve_calc_sd, 0, 0, 0, 0, 0, 0);
            FIND_PADDING_SIZE(tea_leaf_ppcg_solve_update_r, 0, 0, 0, 0, 0, 0);
            FIND_PADDING_SIZE(tea_leaf_ppcg_solve_init_p, 0, 0, 0, 0, 0, 0);
        }
    }
    else
    {
        FIND_PADDING_SIZE(tea_leaf_jacobi_init, -1, 2, -1, 2, -1, 2);
        FIND_PADDING_SIZE(tea_leaf_jacobi_copy_u, -1, 1, -1, 1, -1, 1);
        FIND_PADDING_SIZE(tea_leaf_jacobi_solve, 0, 0, 0, 0, 0, 0);
    }

    FIND_PADDING_SIZE(tea_leaf_init_diag, -1, 1, -1, 1, -1, 1);
    FIND_PADDING_SIZE(tea_leaf_finalise, 0, 0, 0, 0, 0, 0);
    FIND_PADDING_SIZE(tea_leaf_calc_residual, 0, 0, 0, 0, 0, 0);
    FIND_PADDING_SIZE(tea_leaf_cheby_calc_2norm, 0, 0, 0, 0, 0, 0);
}

void CloverChunk::initArgs
(void)
{
    #define SETARG_CHECK(knl, idx, buf) \
        try \
        { \
            knl.setArg(idx, buf); \
        } \
        catch (cl::Error e) \
        { \
            DIE("Error in setting argument index %d to %s for kernel %s (%s - %d)", \
                idx, #buf, #knl, \
                e.what(), e.err()); \
        }

    // initialise chunk
    initialise_chunk_first_device.setArg(6, vertexx);
    initialise_chunk_first_device.setArg(7, vertexdx);
    initialise_chunk_first_device.setArg(8, vertexy);
    initialise_chunk_first_device.setArg(9, vertexdy);
    initialise_chunk_first_device.setArg(10, vertexz);
    initialise_chunk_first_device.setArg(11, vertexdz);

    initialise_chunk_second_device.setArg(6, vertexx);
    initialise_chunk_second_device.setArg(7, vertexdx);
    initialise_chunk_second_device.setArg(8, vertexy);
    initialise_chunk_second_device.setArg(9, vertexdy);
    initialise_chunk_second_device.setArg(10, vertexz);
    initialise_chunk_second_device.setArg(11, vertexdz);
    initialise_chunk_second_device.setArg(12, cellx);
    initialise_chunk_second_device.setArg(13, celldx);
    initialise_chunk_second_device.setArg(14, celly);
    initialise_chunk_second_device.setArg(15, celldy);
    initialise_chunk_second_device.setArg(16, cellz);
    initialise_chunk_second_device.setArg(17, celldz);
    initialise_chunk_second_device.setArg(18, volume);
    initialise_chunk_second_device.setArg(19, xarea);
    initialise_chunk_second_device.setArg(20, yarea);
    initialise_chunk_second_device.setArg(21, zarea);



    // set field
    set_field_device.setArg(0, density);
    set_field_device.setArg(1, energy0);
    set_field_device.setArg(2, energy1);

    // generate chunk
    generate_chunk_init_device.setArg(0, density);
    generate_chunk_init_device.setArg(1, energy0);
    generate_chunk_init_device.setArg(2, xvel0);
    generate_chunk_init_device.setArg(3, yvel0);
    generate_chunk_init_device.setArg(4, zvel0);

    generate_chunk_device.setArg(0, vertexx);
    generate_chunk_device.setArg(1, vertexy);
    generate_chunk_device.setArg(2, vertexz);
    generate_chunk_device.setArg(3, cellx);
    generate_chunk_device.setArg(4, celly);
    generate_chunk_device.setArg(5, cellz);
    generate_chunk_device.setArg(6, density);
    generate_chunk_device.setArg(7, energy0);
    generate_chunk_device.setArg(8, xvel0);
    generate_chunk_device.setArg(9, yvel0);
    generate_chunk_device.setArg(10, zvel0);
    generate_chunk_device.setArg(11, u);


    // field summary
    field_summary_device.setArg(0, volume);
    field_summary_device.setArg(1, density);
    field_summary_device.setArg(2, energy0);
    field_summary_device.setArg(3, u);

    field_summary_device.setArg(4, reduce_buf_1);
    field_summary_device.setArg(5, reduce_buf_2);
    field_summary_device.setArg(6, reduce_buf_3);
    field_summary_device.setArg(7, reduce_buf_4);

    // no parameters set for update_halo here

    // tealeaf
    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        /*
         *  work_array_1 = p
         *  work_array_2 = r
         *  work_array_3 = w / d (just for initialisation)
         *  work_array_4 = Mi
         *
         *  work_array_5 = Kx
         *  work_array_6 = Ky
         *  work_array_7 = Kz
         *
         *  reduce_buf_1 = bb
         *  reduce_buf_2 = rro
         *  reduce_buf_3 = pw
         *  reduce_buf_4 = rrn
         */
        tea_leaf_cg_init_u_device.setArg(0, density);
        tea_leaf_cg_init_u_device.setArg(1, energy1);
        tea_leaf_cg_init_u_device.setArg(2, u);
        tea_leaf_cg_init_u_device.setArg(3, work_array_1);
        tea_leaf_cg_init_u_device.setArg(4, work_array_2);
        tea_leaf_cg_init_u_device.setArg(5, work_array_3);

        tea_leaf_cg_init_directions_device.setArg(0, work_array_3);
        tea_leaf_cg_init_directions_device.setArg(1, work_array_5);
        tea_leaf_cg_init_directions_device.setArg(2, work_array_6);
        tea_leaf_cg_init_directions_device.setArg(3, work_array_7);

        tea_leaf_cg_init_others_device.setArg(0, reduce_buf_2);
        tea_leaf_cg_init_others_device.setArg(1, u);
        tea_leaf_cg_init_others_device.setArg(2, work_array_1);
        tea_leaf_cg_init_others_device.setArg(3, work_array_2);
        tea_leaf_cg_init_others_device.setArg(4, work_array_3);
        tea_leaf_cg_init_others_device.setArg(5, work_array_4);
        tea_leaf_cg_init_others_device.setArg(6, work_array_5);
        tea_leaf_cg_init_others_device.setArg(7, work_array_6);
        tea_leaf_cg_init_others_device.setArg(8, work_array_7);
        // used when preconditioner is used
        tea_leaf_cg_init_others_device.setArg(12, z);

        tea_leaf_cg_solve_calc_w_device.setArg(0, reduce_buf_3);
        tea_leaf_cg_solve_calc_w_device.setArg(1, work_array_1);
        tea_leaf_cg_solve_calc_w_device.setArg(2, work_array_3);
        tea_leaf_cg_solve_calc_w_device.setArg(3, work_array_5);
        tea_leaf_cg_solve_calc_w_device.setArg(4, work_array_6);
        tea_leaf_cg_solve_calc_w_device.setArg(5, work_array_7);

        tea_leaf_cg_solve_calc_ur_device.setArg(1, reduce_buf_4);
        tea_leaf_cg_solve_calc_ur_device.setArg(2, u);
        tea_leaf_cg_solve_calc_ur_device.setArg(3, work_array_1);
        tea_leaf_cg_solve_calc_ur_device.setArg(4, work_array_2);
        tea_leaf_cg_solve_calc_ur_device.setArg(5, work_array_3);
        // used when preconditioner is used
        tea_leaf_cg_solve_calc_ur_device.setArg(6, z);
        tea_leaf_cg_solve_calc_ur_device.setArg(7, work_array_4);

        tea_leaf_cg_solve_calc_p_device.setArg(1, work_array_1);
        tea_leaf_cg_solve_calc_p_device.setArg(2, work_array_2);
        tea_leaf_cg_solve_calc_p_device.setArg(3, z);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            tea_leaf_cheby_solve_init_p_device.setArg(0, u);
            tea_leaf_cheby_solve_init_p_device.setArg(1, u0);
            tea_leaf_cheby_solve_init_p_device.setArg(2, work_array_1);
            tea_leaf_cheby_solve_init_p_device.setArg(3, work_array_2);
            tea_leaf_cheby_solve_init_p_device.setArg(4, work_array_4);
            tea_leaf_cheby_solve_init_p_device.setArg(5, work_array_3);
            tea_leaf_cheby_solve_init_p_device.setArg(6, work_array_5);
            tea_leaf_cheby_solve_init_p_device.setArg(7, work_array_6);
            tea_leaf_cheby_solve_init_p_device.setArg(8, work_array_7);

            tea_leaf_cheby_solve_calc_u_device.setArg(0, u);
            tea_leaf_cheby_solve_calc_u_device.setArg(1, work_array_1);

            tea_leaf_cheby_solve_calc_p_device.setArg(0, u);
            tea_leaf_cheby_solve_calc_p_device.setArg(1, u0);
            tea_leaf_cheby_solve_calc_p_device.setArg(2, work_array_1);
            tea_leaf_cheby_solve_calc_p_device.setArg(3, work_array_2);
            tea_leaf_cheby_solve_calc_p_device.setArg(4, work_array_4);
            tea_leaf_cheby_solve_calc_p_device.setArg(5, work_array_3);
            tea_leaf_cheby_solve_calc_p_device.setArg(6, work_array_5);
            tea_leaf_cheby_solve_calc_p_device.setArg(7, work_array_6);
            tea_leaf_cheby_solve_calc_p_device.setArg(8, work_array_7);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            tea_leaf_ppcg_solve_init_sd_device.setArg(0, work_array_2);
            tea_leaf_ppcg_solve_init_sd_device.setArg(1, work_array_4);
            tea_leaf_ppcg_solve_init_sd_device.setArg(2, work_array_8);

            tea_leaf_ppcg_solve_update_r_device.setArg(0, u);
            tea_leaf_ppcg_solve_update_r_device.setArg(1, work_array_2);
            tea_leaf_ppcg_solve_update_r_device.setArg(2, work_array_5);
            tea_leaf_ppcg_solve_update_r_device.setArg(3, work_array_6);
            tea_leaf_ppcg_solve_update_r_device.setArg(4, work_array_7);
            tea_leaf_ppcg_solve_update_r_device.setArg(5, work_array_8);

            tea_leaf_ppcg_solve_calc_sd_device.setArg(0, work_array_2);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(1, work_array_4);
            tea_leaf_ppcg_solve_calc_sd_device.setArg(2, work_array_8);

            tea_leaf_ppcg_solve_init_p_device.setArg(0, work_array_1);
            tea_leaf_ppcg_solve_init_p_device.setArg(1, work_array_2);
            tea_leaf_ppcg_solve_init_p_device.setArg(2, work_array_4);
            tea_leaf_ppcg_solve_init_p_device.setArg(3, reduce_buf_1);
        }
    }
    else
    {
        tea_leaf_jacobi_init_device.setArg(0, density);
        tea_leaf_jacobi_init_device.setArg(1, energy1);
        tea_leaf_jacobi_init_device.setArg(2, u);
        tea_leaf_jacobi_init_device.setArg(3, work_array_5);
        tea_leaf_jacobi_init_device.setArg(4, work_array_6);
        tea_leaf_jacobi_init_device.setArg(5, work_array_7);

        tea_leaf_jacobi_copy_u_device.setArg(0, u);
        tea_leaf_jacobi_copy_u_device.setArg(1, work_array_4);

        tea_leaf_jacobi_solve_device.setArg(3, u);
        tea_leaf_jacobi_solve_device.setArg(4, u0);
        tea_leaf_jacobi_solve_device.setArg(5, work_array_4);
        tea_leaf_jacobi_solve_device.setArg(6, work_array_5);
        tea_leaf_jacobi_solve_device.setArg(7, work_array_6);
        tea_leaf_jacobi_solve_device.setArg(8, work_array_7);
        tea_leaf_jacobi_solve_device.setArg(9, reduce_buf_1);
    }

    tea_leaf_init_diag_device.setArg(0, work_array_5);
    tea_leaf_init_diag_device.setArg(1, work_array_6);
    tea_leaf_init_diag_device.setArg(2, work_array_7);

    tea_leaf_calc_residual_device.setArg(0, u);
    tea_leaf_calc_residual_device.setArg(1, u0);
    tea_leaf_calc_residual_device.setArg(2, work_array_2);
    tea_leaf_calc_residual_device.setArg(3, work_array_5);
    tea_leaf_calc_residual_device.setArg(4, work_array_6);
    tea_leaf_calc_residual_device.setArg(5, work_array_7);

    tea_leaf_cheby_calc_2norm_device.setArg(1, reduce_buf_1);

    // both finalise the same
    tea_leaf_finalise_device.setArg(0, density);
    tea_leaf_finalise_device.setArg(1, u);
    tea_leaf_finalise_device.setArg(2, energy1);

    fprintf(DBGOUT, "Kernel arguments set\n");
}

