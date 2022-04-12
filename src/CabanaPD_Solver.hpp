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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <CabanaPD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Comm.hpp>
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

template <class DeviceType, class ForceModel>
class Solver : public SolverBase
{
  public:
    using exec_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;

    using particle_type = Particles<memory_space>;
    using integrator_type = Integrator<exec_space>;
    using comm_type = Comm<DeviceType>;
    using force_model_type = ForceModel;
    using force_type = Force<exec_space, force_model_type>;
    using neighbor_type =
        Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    using neigh_iter_tag = Cabana::SerialOpTag;

    int num_steps;
    int output_frequency;
    double init_time;

    Solver( Inputs _inputs, std::shared_ptr<particle_type> _particles,
            force_model_type force_model )
        : particles( _particles )
        , inputs( std::make_shared<Inputs>( _inputs ) )
    {
        Kokkos::Timer init_timer;
        init_timer.reset();
        init_time = 0;

        std::ofstream out( inputs->output_file, std::ofstream::app );
        std::ofstream err( inputs->error_file, std::ofstream::app );

        num_steps = inputs->num_steps;
        output_frequency = inputs->output_frequency;

        auto time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now() );
        log( out, "CabanaPD (", std::ctime( &time ), ")\n" );
        if ( print_rank() )
            exec_space::print_configuration( out );

        // Create integrator.
        // FIXME: hardcoded
        integrator = std::make_shared<integrator_type>( inputs->timestep, 1.0 );

        // Add ghosts from other MPI ranks
        comm = std::make_shared<comm_type>( *particles );

        // Create the neighbor list.
        double mesh_min[3] = { particles->ghost_mesh_lo[0],
                               particles->ghost_mesh_lo[1],
                               particles->ghost_mesh_lo[2] };
        double mesh_max[3] = { particles->ghost_mesh_hi[0],
                               particles->ghost_mesh_hi[1],
                               particles->ghost_mesh_hi[2] };
        auto x = particles->slice_x();
        neighbors = std::make_shared<neighbor_type>(
            x, 0, particles->n_local, inputs->delta, 1.0, mesh_min, mesh_max );

        force = std::make_shared<force_type>( inputs->half_neigh, force_model );

        // Compute initial forces
        auto f = particles->slice_f();
        Cabana::deep_copy( f, 0.0 );
        compute_force( *force, *particles, *neighbors, neigh_iter_tag{} );

        Cajita::Experimental::SiloParticleOutput::writeTimeStep(
            "particles", particles->local_grid->globalGrid(), 0, 0, x,
            particles->slice_W(), particles->slice_f(), particles->slice_u(),
            particles->slice_id() );
        /*
        for ( std::size_t pid = 0; pid < x.size(); pid++ )
            std::cout << x( pid, 0 ) << " " << x( pid, 1 ) << " " << x( pid, 2 )
                      << " " << u( pid, 0 ) << " " << u( pid, 1 ) << " "
                      << u( pid, 2 ) << " " << f( pid, 0 ) << " " << f( pid, 1 )
                      << " " << f( pid, 2 ) << std::endl;
        */
        log( out, "Nlocal Nghost Nglobal\n", particles->n_local, " ",
             particles->n_ghost, " ", particles->n_global );
        init_time += init_timer.seconds();
    }

    void run()
    {
        std::ofstream out( inputs->output_file, std::ofstream::app );
        double force_time = 0;
        double integrate_time = 0;
        double comm_time = 0;
        double other_time = 0;
        double last_time = 0;
        Kokkos::Timer total_timer, force_timer, comm_timer, integrate_timer,
            other_timer;
        total_timer.reset();

        // Main timestep loop
        for ( int step = 1; step <= num_steps; step++ )
        {
            if ( print_rank() )
                std::cout << step << std::endl;

            // Integrate - velocity Verlet first half
            integrate_timer.reset();
            integrator->initial_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Update ghost particles.
            comm_timer.reset();
            comm->gather( *particles );
            comm_time += comm_timer.seconds();

            // Reset forces
            force_timer.reset();
            particles->slice_f();
            auto f = particles->slice_f();
            Cabana::deep_copy( f, 0.0 );

            // Compute short range force
            compute_force( *force, *particles, *neighbors, neigh_iter_tag{} );
            force_time += force_timer.seconds();

            // Integrate - velocity Verlet second half
            integrate_timer.reset();
            integrator->final_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Print output
            other_timer.reset();
            if ( step % output_frequency == 0 )
            {
                auto W = compute_energy( *force, *particles, *neighbors,
                                         neigh_iter_tag() );

                double time = total_timer.seconds();
                double rate = 1.0 * particles->n_global * output_frequency /
                              ( time - last_time );
                log( out, std::fixed, std::setprecision( 6 ), step, " ", W, " ",
                     std::setprecision( 2 ), time, " ", std::scientific, rate );
                last_time = time;

                auto x = particles->slice_x();
                Cajita::Experimental::SiloParticleOutput::writeTimeStep(
                    "particles", particles->local_grid->globalGrid(),
                    step / output_frequency, step * inputs->timestep, x,
                    particles->slice_W(), particles->slice_f(),
                    particles->slice_u(), particles->slice_id() );

                /*
                auto u = particles->slice_u();
                auto f = particles->slice_f();
                for ( std::size_t pid = 0; pid < x.size(); pid++ )
                    std::cout << x( pid, 0 ) << " " << x( pid, 1 ) << " "
                              << x( pid, 2 ) << " " << u( pid, 0 ) << " "
                              << u( pid, 1 ) << " " << u( pid, 2 ) << " "
                              << f( pid, 0 ) << " " << f( pid, 1 ) << " "
                              << f( pid, 2 ) << std::endl;
                */
            }
            other_time += other_timer.seconds();
        }

        double time = total_timer.seconds();

        // Final output and timings
        double steps_per_sec = 1.0 * num_steps / time;
        double p_steps_per_sec = particles->n_global * steps_per_sec;
        log( out, std::fixed, std::setprecision( 2 ),
             "\n#Procs Particles | Time T_Force T_Int T_Other T_Init |\n",
             comm->mpi_size, " ", particles->n_global, " | ", time, " ",
             force_time, " ", comm_time, " ", integrate_time, " ", other_time,
             " ", init_time, " | PERFORMANCE\n", std::fixed, comm->mpi_size,
             " ", particles->n_global, " | ", 1.0, " ", force_time / time, " ",
             comm_time / time, " ", integrate_time / time, " ",
             other_time / time, " ", init_time / time, " | FRACTION\n\n",
             "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
             std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
             p_steps_per_sec / comm->mpi_size );
        out.close();
    }

  private:
    std::shared_ptr<particle_type> particles;
    std::shared_ptr<Inputs> inputs;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<force_type> force;
    std::shared_ptr<neighbor_type> neighbors;
};

template <class MemorySpace, class ForceModel>
std::shared_ptr<SolverBase> createSolver( Inputs inputs,
                                          Particles<MemorySpace> particles )
{
    std::string device_type = inputs.device_type;
    if ( device_type.compare( "SERIAL" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return std::make_shared<Solver<
            Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, ForceModel>>(
            inputs, particles );
#endif
    }
    else if ( device_type.compare( "OPENMP" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return std::make_shared<Solver<
            Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, ForceModel>>(
            inputs, particles );
#endif
    }
    else if ( device_type.compare( "CUDA" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<Solver<
            Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, ForceModel>>(
            inputs, particles );
#endif
    }
    else if ( device_type.compare( "HIP" ) == 0 )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<
            Solver<Kokkos::Device<Kokkos::Experimental::HIP,
                                  Kokkos::Experimental::HIPSpace>,
                   ForceModel>>( inputs, particles );
#endif
    }

    log_err( std::cout, "Unknown backend: ", device_type );
    return nullptr;
}

} // namespace CabanaPD

#endif
