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

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

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
        auto size = inputs["system_size"]["value"];
        for ( std::size_t d = 0; d < size.size(); d++ )
        {
            double s = size[d];
            inputs["low_corner"]["value"][d] = -0.5 * s;
            inputs["high_corner"]["value"][d] = 0.5 * s;
        }
        // Number of steps.
        double tf = inputs["final_time"]["value"];
        double dt = inputs["timestep"]["value"];
        int num_steps = tf / dt;
        inputs["num_steps"] = num_steps;

        // Output files.
        if ( !inputs.contains( "output_file" ) )
            inputs["output_file"] = "cabanaPD.out";
        if ( !inputs.contains( "error_file" ) )
            inputs["error_file"] = "cabanaPD.err";
        inputs["input_file"] = filename;

        // Save inputs (including derived) to new file.
        std::string input_file = "cabanaPD.in.json";
        if ( !inputs.contains( "exported_input_file" ) )
            inputs["exported_input_file"] = input_file;
        std::ofstream in( input_file );
        in << inputs;

        if ( !inputs.contains( "output_reference" ) )
            inputs["output_reference"] = true;

        // Not yet a user option.
        inputs["half_neigh"] = false;
    }
    ~Inputs() {}

    // Parse JSON file.
    inline nlohmann::json parse( const std::string& filename )
    {
        std::ifstream stream( filename );
        return nlohmann::json::parse( stream );
    }

    // Get a single input.
    auto operator[]( std::string label ) { return inputs[label]; }

    // Check a key exists.
    bool contains( std::string label ) { return inputs.contains( label ); }

  protected:
    nlohmann::json inputs;
};

} // namespace CabanaPD

#endif
