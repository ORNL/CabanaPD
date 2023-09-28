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

namespace CabanaPD
{
class Inputs
{
  public:
    std::string output_file = "cabanaPD.out";
    std::string error_file = "cabanaPD.err";
    std::string input_file = "particles.csv";
    std::string device_type = "SERIAL";

    std::array<int, 3> num_cells;
    std::array<double, 3> low_corner;
    std::array<double, 3> high_corner;

    std::size_t num_steps;
    double final_time;
    double timestep;
    int output_frequency;

    bool half_neigh = false;

    Inputs( const std::array<int, 3> nc, std::array<double, 3> lc,
            std::array<double, 3> hc, const double t_f, const double dt,
            const int output_freq );
    ~Inputs();
    void read_args( int argc, char* argv[] );
};

} // namespace CabanaPD

#endif
