/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef INPUTS_H
#define INPUTS_H

#include <string>

#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
template <int Dim = 3>
class Inputs
{
  public:
    std::string output_file = "cabanaPD.out";
    std::string error_file = "cabanaPD.err";
    std::string device_type = "SERIAL";

    std::array<int, Dim> num_cells;
    std::array<double, Dim> low_corner;
    std::array<double, Dim> high_corner;

    std::size_t num_steps;
    double final_time;
    double timestep;
    int output_frequency;
    bool output_reference;

    bool half_neigh = false;

    Inputs( const std::array<int, Dim> nc, std::array<double, Dim> lc,
            std::array<double, Dim> hc, const double t_f, const double dt,
            const int of, const bool output_ref )
        : num_cells( nc )
        , low_corner( lc )
        , high_corner( hc )
        , final_time( t_f )
        , timestep( dt )
        , output_frequency( of )
        , output_reference( output_ref )
    {
        num_steps = final_time / timestep;
    }
    ~Inputs(){};

    void read_args( int argc, char* argv[] )
    {
        for ( int i = 1; i < argc; i++ )
        {
            // Help command.
            if ( ( strcmp( argv[i], "-h" ) == 0 ) ||
                 ( strcmp( argv[i], "--help" ) == 0 ) )
            {
                if ( print_rank() )
                {
                    log( std::cout, "CabanaPD\n", "Options:" );
                    log( std::cout, "  -o [FILE] (OR)\n"
                                    "  --output-file [FILE]:    Provide output "
                                    "file name\n" );
                    log(
                        std::cout,
                        "  -e [FILE] (OR)\n"
                        "  --error-file [FILE]:    Provide error file name\n" );
                    /* Not yet enabled.
                    log(
                        std::cout,
                        "  --device-type [TYPE]:     Kokkos device type to run
                    ", "with\n", "                                (SERIAL,
                    PTHREAD, OPENMP, " "CUDA, HIP)" );
                    */
                }
            }
            // Output file names.
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

            // Kokkos device type.
            else if ( ( strcmp( argv[i], "--device-type" ) == 0 ) )
            {
                device_type = argv[i + 1];
                ++i;
            }

            else if ( ( strstr( argv[i], "--kokkos-" ) == NULL ) )
            {
                log_err( std::cout,
                         "Unknown command line argument: ", argv[i] );
            }
        }
    }
};

} // namespace CabanaPD

#endif
