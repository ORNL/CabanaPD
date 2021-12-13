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

#ifndef SOLVER_H
#define SOLVER_H

#include <fstream>
#include <iomanip>
#include <iostream>

#include <CabanaPD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Integrate.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Particles.hpp>

namespace CabanaPD
{

class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void run() = 0;
};

template <class DeviceType>
class Solver : public SolverBase
{
  public:
    using exec_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;

    int n_steps;
    int write_freq;

    Solver( Inputs inputs )
    {
        // double init_time = 0;
        Kokkos::Timer init_timer;
        init_timer.reset();

        std::ofstream out( inputs.output_file, std::ofstream::app );
        std::ofstream err( inputs.error_file, std::ofstream::app );

        // Create the inputs.
        n_steps = inputs.n_steps;
        log( out, "Read inputs." );

        if ( print_rank() )
            exec_space::print_configuration( out );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        Particles<memory_space> particles( exec_space(), inputs.low_corner,
                                           inputs.high_corner,
                                           inputs.num_cells );

        auto x = particles.slice_x();
        for ( std::size_t p = 0; p < x.size(); p++ )
            std::cout << x( p, 0 ) << " " << x( p, 1 ) << " " << x( p, 2 )
                      << std::endl;

        // Create integrator.
        // FIXME: hardcoded
        Integrator<exec_space> integrator( 0.001, 1.0 );

        // Add ghosts from other MPI ranks
        // FIXME: add Halo or use with PGhalo
        // Cabana::Halo<device_type> halo();
        // particles.gather( halo );

        // Create the neighbor list.
        double mesh_min[3] = { particles.ghost_mesh_lo[0],
                               particles.ghost_mesh_lo[1],
                               particles.ghost_mesh_lo[2] };
        double mesh_max[3] = { particles.ghost_mesh_hi[0],
                               particles.ghost_mesh_hi[1],
                               particles.ghost_mesh_hi[2] };
        std::cout << particles.n_local << std::endl;
        using verlet_list =
            Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                               Cabana::VerletLayout2D, Cabana::TeamOpTag>;
        verlet_list neighbors( x, 0, particles.n_local, inputs.delta, 1.0,
                               mesh_min, mesh_max );
        Cabana::NeighborList<verlet_list> nlist;
        std::cout << nlist.maxNeighbor( neighbors ) << std::endl;

        Force<exec_space> force( inputs.half_neigh, inputs.K, inputs.delta );

        // Compute initial forces
        auto f = particles.slice_f();
        Cabana::deep_copy( f, 0.0 );
        if ( inputs.force_type == "PMB" )
            force.compute( CabanaPD::PMBModelTag{}, particles, neighbors,
                           Cabana::SerialOpTag{} );
    }

    void run() {}
};

std::shared_ptr<SolverBase> createSolver( Inputs inp )
{
    std::string device_type = inp.device_type;
    if ( device_type.compare( "SERIAL" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return std::make_shared<
            Solver<Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>>>( inp );
#endif
    }
    else if ( device_type.compare( "OPENMP" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return std::make_shared<
            Solver<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>>( inp );
#endif
    }
    else if ( device_type.compare( "CUDA" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            Solver<Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>>( inp );
#endif
    }
    else if ( device_type.compare( "HIP" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<Solver<Kokkos::Device<
            Kokkos::Experimental::HIP, Kokkos::Experimental::HIPSpace>>>( inp );
#endif
    }

    log_err( std::cout, "Unknown backend: ", device_type );
    return nullptr;
}

} // namespace CabanaPD

#endif
