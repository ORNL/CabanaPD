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

#ifndef INPUTS_H
#define INPUTS_H

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
class Inputs
{
  public:
    Inputs( const std::string filename )
    {
        // Get user inputs.
        inputs = parse( filename );

        // Add additional derived inputs to json. System size.
        setupSize();

        // Number of steps.
        double tf = inputs["final_time"]["value"];
        double dt = inputs["timestep"]["value"];

        // m
        // FIXME: this will be slightly different in y/z
        double dx = inputs["dx"]["value"][0];
        double delta = inputs["horizon"]["value"];
        int m = std::floor( delta / dx );
        inputs["m"]["value"] = m;

        // Set timestep safety factor if it not set by user
        if ( !inputs.contains( "timestep_safety_factor" ) )
            inputs["timestep_safety_factor"]["value"] = 0.85;

        // Set approximate bulk modulus if not set by user
        // Used in timestep estimation
        if ( !inputs.contains( "bulk_modulus" ) )
        {
            if ( !inputs.contains( "elastic_modulus" ) )
                throw std::runtime_error( "Must input either bulk_modulus or "
                                          "elastic_modulus." );
            double E = inputs["elastic_modulus"]["value"];
            double nu = 0.25;
            double K = E / ( 3 * ( 1 - 2 * nu ) );
            inputs["bulk_modulus"]["value"] = K;
        }

        // Check critical time step
        // This must be done after the values above are calculated
        computeCriticalTimeStep();

        int num_steps = tf / dt;
        inputs["num_steps"]["value"] = num_steps;

        // Output files.
        if ( !inputs.contains( "output_file" ) )
            inputs["output_file"]["value"] = "cabanaPD.out";
        if ( !inputs.contains( "error_file" ) )
            inputs["error_file"]["value"] = "cabanaPD.err";
        inputs["input_file"]["value"] = filename;

        // Save inputs (including derived) to new file.
        std::string input_file = "cabanaPD.in.json";
        if ( !inputs.contains( "exported_input_file" ) )
            inputs["exported_input_file"]["value"] = input_file;
        std::ofstream in( input_file );
        in << inputs;

        if ( !inputs.contains( "output_reference" ) )
            inputs["output_reference"]["value"] = true;

        // Not yet a user option.
        inputs["half_neigh"]["value"] = false;
    }

    void setupSize()
    {
        if ( inputs.contains( "system_size" ) )
        {
            auto system_size = inputs["system_size"]["value"];
            if ( system_size.size() != 3 )
                log_err( std::cout, "CabanaPD requires 3d (system_size)." );

            for ( std::size_t d = 0; d < system_size.size(); d++ )
            {
                double s = system_size[d];
                double low = -0.5 * s;
                double high = 0.5 * s;
                inputs["low_corner"]["value"][d] = low;
                inputs["high_corner"]["value"][d] = high;
            }
            std::string size_unit = inputs["system_size"]["unit"];
            inputs["low_corner"]["unit"] = size_unit;
            inputs["high_corner"]["unit"] = size_unit;
        }
        else if ( inputs.contains( "low_corner" ) &&
                  inputs.contains( "high_corner" ) )
        {
            auto low_corner = inputs["low_corner"]["value"];
            auto high_corner = inputs["high_corner"]["value"];
            if ( low_corner.size() != 3 )
                log_err( std::cout, "CabanaPD requires 3d (low_corner)." );
            if ( high_corner.size() != 3 )
                log_err( std::cout, "CabanaPD requires 3d (high_corner)." );

            for ( std::size_t d = 0; d < low_corner.size(); d++ )
            {
                double low = low_corner[d];
                double high = high_corner[d];
                inputs["system_size"]["value"][d] = high - low;
            }
            std::string size_unit = inputs["low_corner"]["unit"];
            inputs["system_size"]["unit"] = size_unit;
        }
        else
        {
            throw std::runtime_error( "Must input either system_size or "
                                      "both low_corner and high_corner." );
        }

        if ( inputs.contains( "dx" ) )
        {
            auto dx = inputs["dx"]["value"];
            if ( dx.size() != 3 )
                log_err( std::cout, "CabanaPD requires 3d (dx)." );

            for ( std::size_t d = 0; d < dx.size(); d++ )
            {
                double size_d = inputs["system_size"]["value"][d];
                double dx_d = dx[d];
                inputs["num_cells"]["value"][d] =
                    static_cast<int>( size_d / dx_d );
            }
        }
        else if ( inputs.contains( "num_cells" ) )
        {
            auto nc = inputs["num_cells"]["value"];
            if ( nc.size() != 3 )
                log_err( std::cout, "CabanaPD requires 3d (num_cells)." );

            for ( std::size_t d = 0; d < nc.size(); d++ )
            {
                double size_d = inputs["system_size"]["value"][d];
                double nc_d = nc[d];
                inputs["dx"]["value"][d] = size_d / nc_d;
            }
            std::string size_unit = inputs["system_size"]["unit"];
            inputs["dx"]["unit"] = size_unit;
        }
        else
        {
            throw std::runtime_error( "Must input either num_cells or dx." );
        }

        // Error for inconsistent units. There is currently no conversion.
        if ( inputs["low_corner"]["unit"] != inputs["high_corner"]["unit"] )
            log_err( std::cout,
                     "Units for low_corner and high_corner do not match." );
        if ( inputs["dx"]["unit"] != inputs["high_corner"]["unit"] )
            log_err( std::cout, "Units for dx do not match system units." );
    }

    void computeCriticalTimeStep()
    {
        // Reference: Silling & Askari, Computers & Structures 83(17–18) (2005):
        // 1526-1535.

        // Compute particle volume
        double dx = inputs["dx"]["value"][0];
        double dy = inputs["dx"]["value"][1];
        double dz = inputs["dx"]["value"][2];
        double v_p = dx * dy * dz;

        // Initialize denominator's summation
        double sum = 0;

        // Run over the neighborhood of a point in the bulk of a body
        int m = inputs["m"]["value"];
        double rho = inputs["density"]["value"];
        double K = inputs["bulk_modulus"]["value"];
        double delta = inputs["horizon"]["value"];
        // FIXME: this is copied from the forces
        double c = 18.0 * K / ( pi * delta * delta * delta * delta );

        for ( int i = -( m + 1 ); i < m + 2; i++ )
        {
            // x-component of bond
            double xi_1 = i * dx;

            for ( int j = -( m + 1 ); j < m + 2; j++ )
            {
                // y-component of bond
                double xi_2 = j * dy;

                for ( int k = -( m + 1 ); k < m + 2; k++ )
                {
                    // z-component of bond
                    double xi_3 = k * dz;

                    // Bond length squared
                    double r2 = xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3;

                    // Check if bond is no longer than delta
                    if ( r2 < delta * delta + 1e-10 )
                    {
                        // Check is bond is not 0
                        if ( r2 > 0 )
                        {
                            // Compute denominator
                            sum += v_p * c / std::sqrt( r2 );
                        }
                    }
                }
            }
        }

        double safety_factor = inputs["timestep_safety_factor"]["value"];
        double dt_crit = safety_factor * std::sqrt( 2 * rho / sum );

        double dt = inputs["timestep"]["value"];
        if ( dt > dt_crit )
        {
            log( std::cout, "WARNING: timestep (", dt,
                 ") is larger than estimated stable timestep (", dt_crit,
                 "), using safety factor of ", safety_factor, ".\n" );
        }
        // Store in inputs
        inputs["critical_timestep"]["value"] = dt_crit;
    }

    // Parse JSON file.
    inline nlohmann::json parse( const std::string& filename )
    {
        std::ifstream stream( filename );
        return nlohmann::json::parse( stream );
    }

    // Get a single input.
    auto operator[]( std::string label ) { return inputs[label]["value"]; }

    // Get a single input.
    std::string units( std::string label )
    {
        if ( inputs[label].contains( "units" ) )
            return inputs[label]["units"];
        else
            return "";
    }

    // Check a key exists.
    bool contains( std::string label ) { return inputs.contains( label ); }

  protected:
    nlohmann::json inputs;
};

} // namespace CabanaPD

#endif
