/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <kernel_headers/random.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <map>
#include <iostream>
#include <dispatch.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        static const uint REPEAT  = 32;
        static const uint THREADS = 256;
        static uint random_seed[2];

        template<typename T, bool isRandu>
        struct random_name
        {
            const char *name()
            {
                return "randi";
            }
        };

        template<typename T>
        struct random_name<T, false>
        {
            const char *name()
            {
                return "randn";
            }
        };

        template<>
        struct random_name<float, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<>
        struct random_name<double, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<typename T, bool isRandu>
        void random(cl::Buffer out, dim_type elements)
        {
            static unsigned counter;

            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*> ranProgs;
            static std::map<int, Kernel*> ranKernels;

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                        Program::Sources setSrc;
                        setSrc.emplace_back(random_cl, random_cl_len);

                        ranProgs[device] = new Program(getContext(), setSrc);

                        std::ostringstream options;
                        options << " -D T=" << dtype_traits<T>::getName()
                                << " -D repeat="<< REPEAT
                                << " -D " << random_name<T, isRandu>().name();
                        if (std::is_same<T, double>::value ||
                            std::is_same<T, cdouble>::value) {
                            options << " -D USE_DOUBLE";
                        }
                        ranProgs[device]->build(options.str().c_str());

                        ranKernels[device] = new Kernel(*ranProgs[device], "random");
                    });

            auto randomOp = make_kernel<cl::Buffer, uint, uint, uint, uint>(*ranKernels[device]);

            uint groups = divup(elements, THREADS * REPEAT);
            counter += divup(elements, THREADS * groups);

            NDRange local(THREADS, 1);
            NDRange global(THREADS * groups, 1);

            randomOp(EnqueueArgs(getQueue(), global, local),
                     out, elements, counter, random_seed[0], random_seed[1]);
        }
    }
}
