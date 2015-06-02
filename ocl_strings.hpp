#include <string>
#include <fstream>

#include "ocl_common.hpp"

enum {AMD_PLAT, INTEL_PLAT, NVIDIA_PLAT, NO_PLAT, ANY_PLAT, LIST_PLAT};

/*
 *  Reads the string assigned to a setting
 */
std::string readString
(std::ifstream& input, const char * setting);

/*
 *  Reads an integer assigned to a setting
 */
int readInt
(std::ifstream& input, const char * setting);

/*
 *  Takes string of type of context and returns enumerated value
 */
int typeMatch
(std::string& type_name);

/*
 *  Takes cl_device_type and returns string (merge into above/bit in ocl_init TODO)
 */
std::string strType
(cl_device_type dtype);

/*
 *  Returns stringified device type
 */
std::string errToString
(cl_int err);

/*
 *  Find if tl_use_cg is in the input file
 */
bool paramEnabled
(std::ifstream& input, const char* param);

/*
 *  Returns index of desired device, or -1 if some error occurs (none specified, invalid specification, etc)
 */
int preferredDevice
(std::ifstream& input);

/*
 *  Find out the value of a parameter
 */
std::string matchParam
(FILE * input,
 const char* param_name);

