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
#include <vector>

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

        // Final time and time step.
        double tf = inputs["final_time"]["value"];
        double dt = inputs["timestep"]["value"];

        // Mass scaling: ensure that the mass scaling is not applied multiple
        // times.
        if ( inputs.contains( "mass_scaling_factor" ) &&
             !inputs.contains( "density_input" ) )
        {
            double ms_factor = inputs["mass_scaling_factor"]["value"];

            // Density scaling
            double rho0 = inputs["density"]["value"];
            inputs["density_input"]["value"] = rho0;
            inputs["density_input"]["unit"] = inputs["density"]["unit"];
            rho0 *= ms_factor;
            inputs["density"]["value"] = rho0;

            // Time step scaling
            inputs["timestep_input"]["value"] = dt;
            inputs["timestep_input"]["unit"] = inputs["timestep"]["unit"];
            dt *= std::sqrt( ms_factor );
            inputs["timestep"]["value"] = dt;
        }

        // if it contains both m and horizon
        if ( inputs.contains( "horizon" ) && inputs.contains( "m" ) )
        {
            throw std::runtime_error( "Cannot input both horizon and m." );
        }

        // m
        // FIXME: this will be slightly different in y/z
        double dx = inputs["dx"]["value"][0];
        if ( inputs.contains( "horizon" ) )
        {
            double horizon = inputs["horizon"]["value"];
            int m = std::floor( horizon / dx );
            inputs["m"]["value"] = m;
        }

        // if it contains m
        else if ( inputs.contains( "m" ) )
        {
            int m = inputs["m"]["value"];
            double horizon = static_cast<double>( m ) * dx + 0.01 * dx;
            inputs["horizon"]["value"] = horizon;
        }

        // Set timestep safety factor if not set by user.
        if ( !inputs.contains( "timestep_safety_factor" ) )
            inputs["timestep_safety_factor"]["value"] = 0.85;

        // Check one of bulk or elastic modulus set by user.
        if ( !inputs.contains( "bulk_modulus" ) &&
             !inputs.contains( "elastic_modulus" ) )
        {
            throw std::runtime_error( "Must input either bulk_modulus or "
                                      "elastic_modulus." );
        }

        // Number of time steps.
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

    template <class ForceModel>
    void computeCriticalTimeStep( [[maybe_unused]] const ForceModel model )
    {
        // Reference: Silling & Askari, Computers & Structures 83(17–18) (2005):
        // 1526-1535.

        // Compute particle volume.
        double dx = inputs["dx"]["value"][0];
        double dy = inputs["dx"]["value"][1];
        double dz = inputs["dx"]["value"][2];
        double v_p = dx * dy * dz;

        // Initialize denominator's summation.
        double sum = 0;
        double sum_ht = 0;

        // Estimate the bulk modulus if needed.
        std::size_t min_index;
        std::vector<double> K;
        if ( inputs.contains( "bulk_modulus" ) )
        {
            K = getVector( "bulk_modulus" );
        }
        else
        {
            auto E = getVector( "elastic_modulus" );
            // This is only exact for bond-based (PMB).
            double nu = 0.25;

            K.resize( E.size() );
            double denom = ( 3.0 * ( 1.0 - 2.0 * nu ) );
            for ( std::size_t i = 0; i < E.size(); i++ )
                K[i] = E[i] / denom;
        }

        // Support for multi-material: find the minimum density-bulk modulus
        // ratio for the correct critical timestep.
        auto rho = getVector( "density" );
        std::vector<double> rho_over_K( rho.size() );
        for ( std::size_t i = 0; i < rho.size(); i++ )
            rho_over_K[i] = rho[i] / K[i];
        min_index = std::distance( std::begin( rho_over_K ),
                                   std::min_element( std::begin( rho_over_K ),
                                                     std::end( rho_over_K ) ) );
        auto min_K = K[min_index];
        auto min_rho = rho[min_index];

        // Run over the neighborhood of a point in the bulk of a body (at the
        // origin).
        int m = inputs["m"]["value"];
        double horizon = inputs["horizon"]["value"];
        // FIXME: this is copied from the forces.
        // FIXME: if horizon is not constant this needs to be updated
        // accordingly.
        double c =
            18.0 * min_K / ( pi * horizon * horizon * horizon * horizon );

        for ( int i = -( m + 1 ); i < m + 2; i++ )
        {
            // x-component of bond.
            double xi_1 = i * dx;

            for ( int j = -( m + 1 ); j < m + 2; j++ )
            {
                // y-component of bond.
                double xi_2 = j * dy;

                for ( int k = -( m + 1 ); k < m + 2; k++ )
                {
                    // z-component of bond.
                    double xi_3 = k * dz;

                    // Bond length squared.
                    double r2 = xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3;

                    // Check if bond is no longer than horizon.
                    if ( r2 < horizon * horizon + 1e-10 )
                    {
                        // Check if bond is not 0.
                        if ( r2 > 0 )
                        {
                            double xi = std::sqrt( r2 );

                            // Compute denominator for mechanics.
                            sum += v_p * c / xi;

                            // Compute denominator for heat transfer.
                            if constexpr ( is_heat_transfer<
                                               typename ForceModel::
                                                   thermal_type>::value )
                            {
                                double coeff =
                                    model.microconductivity_function( xi );
                                sum_ht += v_p * coeff / r2;
                            }
                        }
                    }
                }
            }
        }

        double dt = inputs["timestep"]["value"];
        // This supports multi-material.
        double dt_crit = std::sqrt( 2.0 * min_rho / sum );
        compareCriticalTimeStep( "mechanics", dt, dt_crit );

        // Heat transfer timestep.
        if constexpr ( is_heat_transfer<
                           typename ForceModel::thermal_type>::value )
        {
            double dt_ht = inputs["thermal_subcycle_steps"]["value"];
            dt_ht *= dt;

            // Does not currently support multi-material.
            double cp = inputs["specific_heat_capacity"]["value"];
            double rho = inputs["density"]["value"];
            double dt_ht_crit = rho * cp / sum_ht;
            compareCriticalTimeStep( "heat_transfer", dt_ht, dt_ht_crit );
        }
    }

    // Return vector for either array or scalar inputs to support both single
    // and multi-material.
    std::vector<double> getVector( const std::string key )
    {
        std::vector<double> v;
        auto j = inputs[key]["value"];
        if ( j.is_array() )
        {
            // Resize to match
            v.resize( j.size() );
            for ( std::size_t i = 0; i < j.size(); i++ )
                v[i] = j[i];
        }
        else
        {
            v.resize( 1 );
            v[0] = j;
        }
        return v;
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
    void compareCriticalTimeStep( std::string name, double dt, double dt_crit )
    {
        double safety_factor = inputs["timestep_safety_factor"]["value"];
        double dt_crit_safety = safety_factor * dt_crit;

        if ( dt > dt_crit_safety )
        {
            log( std::cout, "WARNING: timestep (", dt,
                 ") is larger than estimated stable timestep for ", name, " (",
                 dt_crit_safety, "), using safety factor of ", safety_factor,
                 ".\n" );
        }
        // Store in inputs.
        inputs["critical_timestep_" + name]["value"] = dt_crit_safety;
    }

    nlohmann::json inputs;
};

} // namespace CabanaPD

#endif
