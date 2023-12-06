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

#include <CabanaPD_Boundary.hpp>
#include <CabanaPD_Comm.hpp>
#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Integrate.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Prenotch.hpp>

namespace CabanaPD
{
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void run() = 0;
};

template <class MemorySpace, class InputType, class ParticleType,
          class ForceModel, class BoundaryCondition>
class SolverElastic
{
  public:
    using memory_space = MemorySpace;
    using exec_space = typename memory_space::execution_space;

    using particle_type = ParticleType;
    using integrator_type = Integrator<exec_space>;
    using force_model_type = ForceModel;
    using force_type = Force<exec_space, force_model_type>;
    using comm_type =
        Comm<particle_type, typename force_model_type::base_model>;
    using neighbor_type =
        Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    using neigh_iter_tag = Cabana::SerialOpTag;
    using input_type = InputType;
    using bc_type = BoundaryCondition;

    SolverElastic( input_type _inputs,
                   std::shared_ptr<particle_type> _particles,
                   force_model_type force_model, bc_type bc )
        : inputs( _inputs )
        , particles( _particles )
        , boundary_condition( bc )
    {
        neighbor_time = 0;
        force_time = 0;
        integrate_time = 0;
        comm_time = 0;
        other_time = 0;
        last_time = 0;
        init_time = 0;
        total_timer.reset();
        init_timer.reset();

        num_steps = inputs["num_steps"];
        output_frequency = inputs["output_frequency"];
        output_reference = inputs["output_reference"];

        // Create integrator.
        dt = inputs["timestep"];
        integrator = std::make_shared<integrator_type>( dt );

        // Add ghosts from other MPI ranks.
        comm = std::make_shared<comm_type>( *particles );

        // Create the neighbor list.
        neighbor_timer.reset();
        double mesh_min[3] = { particles->ghost_mesh_lo[0],
                               particles->ghost_mesh_lo[1],
                               particles->ghost_mesh_lo[2] };
        double mesh_max[3] = { particles->ghost_mesh_hi[0],
                               particles->ghost_mesh_hi[1],
                               particles->ghost_mesh_hi[2] };
        auto x = particles->sliceReferencePosition();
        neighbors = std::make_shared<neighbor_type>( x, 0, particles->n_local,
                                                     force_model.delta, 1.0,
                                                     mesh_min, mesh_max );
        neighbor_time += neighbor_timer.seconds();

        unsigned max_neighbors;
        unsigned max_local_neighbors =
            Cabana::NeighborList<neighbor_type>::maxNeighbor( *neighbors );
        unsigned long long total_neighbors;
        unsigned long long total_local_neighbors =
            Cabana::NeighborList<neighbor_type>::totalNeighbor( *neighbors );
        MPI_Reduce( &max_local_neighbors, &max_neighbors, 1, MPI_UNSIGNED,
                    MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &total_local_neighbors, &total_neighbors, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD );

        force =
            std::make_shared<force_type>( inputs["half_neigh"], force_model );

        print = print_rank();
        if ( print )
        {
            log( std::cout, "Local particles: ", particles->n_local,
                 ", Maximum neighbors: ", max_neighbors );
            log( std::cout, "#Timestep/Total-steps Simulation-time" );

            output_file = inputs["output_file"];
            std::ofstream out( output_file, std::ofstream::app );
            error_file = inputs["error_file"];
            std::ofstream err( error_file, std::ofstream::app );

            auto time = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now() );
            log( out, "CabanaPD ", std::ctime( &time ), "\n" );
            exec_space().print_configuration( out );

            log( out, "Local particles, Ghosted particles, Global particles\n",
                 particles->n_local, ", ", particles->n_ghost, ", ",
                 particles->n_global );
            log( out, "Maximum neighbors: ", max_neighbors,
                 ", Total neighbors: ", total_neighbors, "\n" );
            out.close();
        }
        init_time += init_timer.seconds();
    }

    void init_force()
    {
        init_timer.reset();
        // Compute/communicate LPS weighted volume (does nothing for PMB).
        force->computeWeightedVolume( *particles, *neighbors,
                                      neigh_iter_tag{} );
        comm->gatherWeightedVolume();
        // Compute/communicate LPS dilatation (does nothing for PMB).
        force->computeDilatation( *particles, *neighbors, neigh_iter_tag{} );
        comm->gatherDilatation();

        // Compute initial forces.
        computeForce( *force, *particles, *neighbors, neigh_iter_tag{} );
        computeEnergy( *force, *particles, *neighbors, neigh_iter_tag() );

        // Add boundary condition.
        boundary_condition.apply( exec_space(), *particles );

        particles->output( 0, 0.0, output_reference );
        init_time += init_timer.seconds();
    }

    void run()
    {
        init_output();

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            // Integrate - velocity Verlet first half.
            integrate_timer.reset();
            integrator->initialHalfStep( *particles );
            integrate_time += integrate_timer.seconds();

            // Update ghost particles.
            comm_timer.reset();
            comm->gatherDisplacement();
            comm_time += comm_timer.seconds();

            // Do not need to recompute LPS weighted volume here without damage.
            // Compute/communicate LPS dilatation (does nothing for PMB).
            force_timer.reset();
            force->computeDilatation( *particles, *neighbors,
                                      neigh_iter_tag{} );
            force_time += force_timer.seconds();
            comm_timer.reset();
            comm->gatherDilatation();
            comm_time += comm_timer.seconds();

            // Compute internal forces.
            force_timer.reset();
            computeForce( *force, *particles, *neighbors, neigh_iter_tag{} );
            force_time += force_timer.seconds();

            // Add boundary condition.
            boundary_condition.apply( exec_space(), *particles );

            // Integrate - velocity Verlet second half.
            integrate_timer.reset();
            integrator->finalHalfStep( *particles );
            integrate_time += integrate_timer.seconds();

            // Print output.
            other_timer.reset();
            if ( step % output_frequency == 0 )
            {
                auto W = computeEnergy( *force, *particles, *neighbors,
                                        neigh_iter_tag() );

                step_output( step, W );
                particles->output( step / output_frequency, step * dt,
                                   output_reference );
            }
            other_time += other_timer.seconds();
        }

        // Final output and timings.
        final_output();
    }

    void init_output()
    {
        // Output after construction and initial forces.
        std::ofstream out( output_file, std::ofstream::app );
        log( out, "Init-Time(s): ", init_time, "\n" );
        log( out, "#Timestep/Total-steps Simulation-time Total-strain-energy "
                  "Run-Time(s) Force-Time(s) Comm-Time(s) Int-Time(s) "
                  "Other-Time(s) Particle*steps/s" );
    }

    void step_output( const int step, const double W )
    {
        if ( print )
        {
            std::ofstream out( output_file, std::ofstream::app );
            log( std::cout, step, "/", num_steps, " ", std::scientific,
                 std::setprecision( 2 ), step * dt );

            total_time = total_timer.seconds();
            double rate = 1.0 * particles->n_global * output_frequency /
                          ( total_time - last_time );
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", std::fixed, total_time, " ", force_time, " ",
                 comm_time, " ", integrate_time, " ", other_time, " ",
                 std::scientific, rate );
            last_time = total_time;
            out.close();
        }
    }

    void final_output()
    {
        if ( print )
        {
            std::ofstream out( output_file, std::ofstream::app );
            total_time = total_timer.seconds();
            double steps_per_sec = 1.0 * num_steps / total_time;
            double p_steps_per_sec = particles->n_global * steps_per_sec;
            log( out, std::fixed, std::setprecision( 2 ),
                 "\n#Procs Particles | Time T_Force T_Comm T_Int T_Other "
                 "T_Init T_Neigh |\n",
                 comm->mpi_size, " ", particles->n_global, " | ", total_time,
                 " ", force_time, " ", comm_time, " ", integrate_time, " ",
                 other_time, " ", init_time, " | PERFORMANCE\n", std::fixed,
                 comm->mpi_size, " ", particles->n_global, " | ", 1.0, " ",
                 force_time / total_time, " ", comm_time / total_time, " ",
                 integrate_time / total_time, " ", other_time / total_time, " ",
                 init_time / total_time, " ", neighbor_time / total_time,
                 " | FRACTION\n\n",
                 "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
                 std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
                 p_steps_per_sec / comm->mpi_size );
            out.close();
        }
    }

    int num_steps;
    int output_frequency;
    bool output_reference;
    double dt;

  protected:
    input_type inputs;
    std::shared_ptr<particle_type> particles;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<force_type> force;
    std::shared_ptr<neighbor_type> neighbors;
    bc_type boundary_condition;

    std::string output_file;
    std::string error_file;

    double total_time;
    double force_time;
    double integrate_time;
    double comm_time;
    double other_time;
    double init_time;
    double last_time;
    double neighbor_time;
    Kokkos::Timer total_timer;
    Kokkos::Timer init_timer;
    Kokkos::Timer force_timer;
    Kokkos::Timer comm_timer;
    Kokkos::Timer integrate_timer;
    Kokkos::Timer other_timer;
    Kokkos::Timer neighbor_timer;
    bool print;
};

template <class MemorySpace, class InputType, class ParticleType,
          class ForceModel, class BoundaryCondition, class PrenotchType>
class SolverFracture
    : public SolverElastic<MemorySpace, InputType, ParticleType, ForceModel,
                           BoundaryCondition>
{
  public:
    using base_type = SolverElastic<MemorySpace, InputType, ParticleType,
                                    ForceModel, BoundaryCondition>;
    using exec_space = typename base_type::exec_space;
    using memory_space = typename base_type::memory_space;

    using particle_type = typename base_type::particle_type;
    using integrator_type = typename base_type::integrator_type;
    using comm_type = typename base_type::comm_type;
    using neighbor_type = typename base_type::neighbor_type;
    using force_model_type = ForceModel;
    using force_type = typename base_type::force_type;
    using neigh_iter_tag = Cabana::SerialOpTag;
    using bc_type = BoundaryCondition;
    using prenotch_type = PrenotchType;
    using input_type = typename base_type::input_type;

    SolverFracture( input_type _inputs,
                    std::shared_ptr<particle_type> _particles,
                    force_model_type force_model, bc_type bc,
                    prenotch_type prenotch )
        : base_type( _inputs, _particles, force_model, bc )
    {
        init_timer.reset();

        // Create View to track broken bonds.
        int max_neighbors =
            Cabana::NeighborList<neighbor_type>::maxNeighbor( *neighbors );
        mu = NeighborView(
            Kokkos::ViewAllocateWithoutInitializing( "broken_bonds" ),
            particles->n_local, max_neighbors );
        Kokkos::deep_copy( mu, 1 );

        // Create prenotch.
        prenotch.create( exec_space{}, mu, *particles, *neighbors );
        init_time += init_timer.seconds();
    }

    void init_force()
    {
        init_timer.reset();
        // Compute/communicate weighted volume for LPS (does nothing for PMB).
        force->computeWeightedVolume( *particles, *neighbors, mu );
        comm->gatherWeightedVolume();
        // Compute/communicate dilatation for LPS (does nothing for PMB).
        force->computeDilatation( *particles, *neighbors, mu );
        comm->gatherDilatation();

        // Compute initial forces.
        computeForce( *force, *particles, *neighbors, mu, neigh_iter_tag{} );
        computeEnergy( *force, *particles, *neighbors, mu, neigh_iter_tag() );

        // Add boundary condition.
        boundary_condition.apply( exec_space(), *particles );

        particles->output( 0, 0.0, output_reference );
        init_time += init_timer.seconds();
    }

    void run()
    {
        this->init_output();

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            // Integrate - velocity Verlet first half.
            integrate_timer.reset();
            integrator->initialHalfStep( *particles );
            integrate_time += integrate_timer.seconds();

            // Update ghost particles.
            comm_timer.reset();
            comm->gatherDisplacement();
            comm_time += comm_timer.seconds();

            // Compute/communicate LPS weighted volume (does nothing for PMB).
            force_timer.reset();
            force->computeWeightedVolume( *particles, *neighbors, mu );
            force_time += force_timer.seconds();
            comm_timer.reset();
            comm->gatherWeightedVolume();
            comm_time += comm_timer.seconds();
            // Compute/communicate LPS dilatation (does nothing for PMB).
            force_timer.reset();
            force->computeDilatation( *particles, *neighbors, mu );
            force_time += force_timer.seconds();
            comm_timer.reset();
            comm->gatherDilatation();
            comm_time += comm_timer.seconds();

            // Compute internal forces.
            force_timer.reset();
            computeForce( *force, *particles, *neighbors, mu,
                          neigh_iter_tag{} );
            force_time += force_timer.seconds();

            // Add boundary condition.
            boundary_condition.apply( exec_space{}, *particles );

            // Integrate - velocity Verlet second half.
            integrate_timer.reset();
            integrator->finalHalfStep( *particles );
            integrate_time += integrate_timer.seconds();

            // Print output.
            other_timer.reset();
            if ( step % output_frequency == 0 )
            {
                auto W = computeEnergy( *force, *particles, *neighbors, mu,
                                        neigh_iter_tag() );

                this->step_output( step, W );
                particles->output( step / output_frequency, step * dt,
                                   output_reference );
            }
            other_time += other_timer.seconds();
        }

        // Final output and timings.
        this->final_output();
    }

    using base_type::dt;
    using base_type::num_steps;
    using base_type::output_frequency;
    using base_type::output_reference;

  protected:
    using base_type::boundary_condition;
    using base_type::comm;
    using base_type::force;
    using base_type::inputs;
    using base_type::integrator;
    using base_type::neighbors;
    using base_type::particles;

    using NeighborView = typename Kokkos::View<int**, memory_space>;
    NeighborView mu;

    using base_type::comm_time;
    using base_type::force_time;
    using base_type::init_time;
    using base_type::integrate_time;
    using base_type::last_time;
    using base_type::other_time;
    using base_type::total_time;

    using base_type::comm_timer;
    using base_type::force_timer;
    using base_type::init_timer;
    using base_type::integrate_timer;
    using base_type::other_timer;
    using base_type::total_timer;

    using base_type::print;
};

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModel, class BCType>
auto createSolverElastic( InputsType inputs,
                          std::shared_ptr<ParticleType> particles,
                          ForceModel model, BCType bc )
{
    return std::make_shared<SolverElastic<MemorySpace, InputsType, ParticleType,
                                          ForceModel, BCType>>(
        inputs, particles, model, bc );
}

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModel, class BCType, class PrenotchType>
auto createSolverFracture( InputsType inputs,
                           std::shared_ptr<ParticleType> particles,
                           ForceModel model, BCType bc, PrenotchType prenotch )
{
    return std::make_shared<
        SolverFracture<MemorySpace, InputsType, ParticleType, ForceModel,
                       BCType, PrenotchType>>( inputs, particles, model, bc,
                                               prenotch );
}

} // namespace CabanaPD

#endif
