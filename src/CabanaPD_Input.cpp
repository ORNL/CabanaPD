/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#include <iostream>

#include <Cabana_Core.hpp>

#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

// FIXME: hardcoded
Inputs::Inputs( const std::array<int, 3> nc, std::array<double, 3> lc,
                std::array<double, 3> hc, const double t_f, const double dt )
    : num_cells( nc )
    , low_corner( lc )
    , high_corner( hc )
    , final_time( t_f )
    , timestep( dt )
{
    num_steps = final_time / timestep;
}

Inputs::~Inputs() {}

void Inputs::read_args( int argc, char* argv[] )
{
    for ( int i = 1; i < argc; i++ )
    {
        // Help command
        if ( ( strcmp( argv[i], "-h" ) == 0 ) ||
             ( strcmp( argv[i], "--help" ) == 0 ) )
        {
            log( std::cout, "CabanaPD\n", "Options:" );
            log( std::cout,
                 "  -o [FILE] (OR)\n"
                 "  --output-file [FILE]:    Provide output file name\n" );
            log( std::cout,
                 "  -e [FILE] (OR)\n"
                 "  --error-file [FILE]:    Provide error file name\n" );
            log( std::cout,
                 "  --device-type [TYPE]:     Kokkos device type to run ",
                 "with\n",
                 "                                (SERIAL, PTHREAD, OPENMP, "
                 "CUDA, HIP)" );
        }
        // Output file names
        else if ( ( strcmp( argv[i], "-o" ) == 0 ) ||
                  ( strcmp( argv[i], "--output-file" ) == 0 ) )
        {
            output_file = argv[i + 1];
            ++i;
        }
        else if ( ( strcmp( argv[i], "-e" ) == 0 ) ||
                  ( strcmp( argv[i], "--error-file" ) == 0 ) )
        {
            error_file = argv[i + 1];
            ++i;
        }

        // Kokkos device type
        else if ( ( strcmp( argv[i], "--device-type" ) == 0 ) )
        {
            device_type = argv[i + 1];
            ++i;
        }

        else if ( ( strstr( argv[i], "--kokkos-" ) == NULL ) )
        {
            log_err( std::cout, "Unknown command line argument: ", argv[i] );
        }
    }
}

} // namespace CabanaPD
