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

template <class DeviceType, class ForceModel>
class SolverElastic
{
  public:
    using exec_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;

    using particle_type = Particles<DeviceType>;
    using integrator_type = Integrator<exec_space>;
    using comm_type = Comm<particle_type>;
    using force_model_type = ForceModel;
    using force_type = Force<exec_space, force_model_type>;
    using neighbor_type =
        Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    using neigh_iter_tag = Cabana::SerialOpTag;

    SolverElastic( Inputs _inputs, std::shared_ptr<particle_type> _particles,
                   force_model_type force_model )
        : particles( _particles )
        , inputs( std::make_shared<Inputs>( _inputs ) )
    {
        force_time = 0;
        integrate_time = 0;
        comm_time = 0;
        other_time = 0;
        last_time = 0;
        init_time = 0;
        total_timer.reset();
        init_timer.reset();

        std::ofstream out( inputs->output_file, std::ofstream::app );
        std::ofstream err( inputs->error_file, std::ofstream::app );

        num_steps = inputs->num_steps;
        output_frequency = inputs->output_frequency;

        auto time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now() );
        log( out, "CabanaPD ", std::ctime( &time ), "\n" );
        if ( print_rank() )
            exec_space().print_configuration( out );

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
        std::cout << x.size() << " " << particles->n_local << std::endl;
        neighbors = std::make_shared<neighbor_type>( x, 0, particles->n_local,
                                                     force_model.delta, 1.0,
                                                     mesh_min, mesh_max );
        int max_neighbors =
            Cabana::NeighborList<neighbor_type>::maxNeighbor( *neighbors );
        log( std::cout, "Local particles: ", particles->n_local,
             ", Maximum local neighbors: ", max_neighbors );
        log( std::cout, "#Timestep/Total-steps Simulation-time" );

        force = std::make_shared<force_type>( inputs->half_neigh, force_model );

        log( out, "Local particles, Ghosted particles, Global particles\n",
             particles->n_local, ", ", particles->n_ghost, ", ",
             particles->n_global );
        log( out, "Maximum local neighbors: ", max_neighbors, "\n" );
        log( out, "#Timestep/Total-steps Simulation-time Total-strain-energy "
                  "Run-Time(s) Particle*steps/s" );
        init_time += init_timer.seconds();
        out.close();
    }

    void init_force()
    {
        // These are split and called here to facilitate the alternating
        // compute/communicate for LPS.
        // Compute weighted volume for LPS (does nothing for PMB).
        force->compute_weighted_volume( *particles, *neighbors,
                                        neigh_iter_tag{} );
        comm->gatherWeightedVolume();
        // Compute dilatation for LPS (does nothing for PMB).
        force->compute_dilatation( *particles, *neighbors, neigh_iter_tag{} );
        // Communicate dilatation for LPS (FIXME: should not be done for PMB).
        comm->gatherDilatation();

        // Compute initial forces
        compute_force( *force, *particles, *neighbors, neigh_iter_tag{} );
        compute_energy( *force, *particles, *neighbors, neigh_iter_tag() );

        particle_output( 0 );
    }

    void run()
    {
        // Main timestep loop
        for ( int step = 1; step <= num_steps; step++ )
        {
            // Integrate - velocity Verlet first half
            integrate_timer.reset();
            integrator->initial_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Update ghost particles.
            comm_timer.reset();
            comm->gatherDisplacement();
            comm_time += comm_timer.seconds();

            // Reset forces
            force_timer.reset();
            // Compute short range force
            // Do not need to recompute LPS weighted volume here without damage.
            // Compute dilatation for LPS (does nothing for PMB).
            force->compute_dilatation( *particles, *neighbors,
                                       neigh_iter_tag{} );
            // Communicate dilatation for LPS (FIXME: should not be done for
            // PMB).
            comm->gatherDilatation();
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

                step_output( step, W );
                particle_output( step );
            }
            other_time += other_timer.seconds();
        }

        // Final output and timings
        final_output();
    }

    void step_output( const int step, const double W )
    {
        std::ofstream out( inputs->output_file, std::ofstream::app );
        log( std::cout, step, "/", num_steps, " ", std::scientific,
             std::setprecision( 2 ), step * inputs->timestep );

        total_time = total_timer.seconds();
        double rate = 1.0 * particles->n_global * output_frequency /
                      ( total_time - last_time );
        log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps, " ",
             std::scientific, std::setprecision( 2 ), step * inputs->timestep,
             " ", W, " ", std::fixed, total_time, " ", std::scientific, rate );
        last_time = total_time;
        out.close();
    }

    void final_output()
    {
        std::ofstream out( inputs->output_file, std::ofstream::app );
        total_time = total_timer.seconds();
        double steps_per_sec = 1.0 * num_steps / total_time;
        double p_steps_per_sec = particles->n_global * steps_per_sec;
        log(
            out, std::fixed, std::setprecision( 2 ),
            "\n#Procs Particles | Time T_Force T_Comm T_Int T_Other T_Init |\n",
            comm->mpi_size, " ", particles->n_global, " | ", total_time, " ",
            force_time, " ", comm_time, " ", integrate_time, " ", other_time,
            " ", init_time, " | PERFORMANCE\n", std::fixed, comm->mpi_size, " ",
            particles->n_global, " | ", 1.0, " ", force_time / total_time, " ",
            comm_time / total_time, " ", integrate_time / total_time, " ",
            other_time / total_time, " ", init_time / total_time,
            " | FRACTION\n\n",
            "#Steps/s Particle-steps/s Particle-steps/proc/s\n",
            std::scientific, steps_per_sec, " ", p_steps_per_sec, " ",
            p_steps_per_sec / comm->mpi_size );
        out.close();
    }

    void particle_output( const int step )
    {
        Cajita::Experimental::SiloParticleOutput::writePartialRangeTimeStep(
            "particles", particles->local_grid->globalGrid(),
            step / output_frequency, step * inputs->timestep, 0,
            particles->n_local, particles->slice_x(), particles->slice_W(),
            particles->slice_f(), particles->slice_u(), particles->slice_v() );
    }

    int num_steps;
    int output_frequency;

  protected:
    std::shared_ptr<particle_type> particles;
    std::shared_ptr<Inputs> inputs;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<force_type> force;
    std::shared_ptr<neighbor_type> neighbors;

    double total_time;
    double force_time;
    double integrate_time;
    double comm_time;
    double other_time;
    double init_time;
    double last_time;
    Kokkos::Timer total_timer;
    Kokkos::Timer init_timer;
    Kokkos::Timer force_timer;
    Kokkos::Timer comm_timer;
    Kokkos::Timer integrate_timer;
    Kokkos::Timer other_timer;
};

template <class DeviceType, class ForceModel>
class SolverFractureAlso
    : public SolverElastic<DeviceType, typename ForceModel::elastic_model>
{
  public:
    using base_type =
        SolverElastic<DeviceType, typename ForceModel::elastic_model>;
    using exec_space = typename base_type::exec_space;
    using memory_space = typename base_type::memory_space;

    using particle_type = typename base_type::particle_type;
    using integrator_type = typename base_type::integrator_type;
    using comm_type = typename base_type::comm_type;
    using neighbor_type = typename base_type::neighbor_type;
    using force_model_type = ForceModel;
    using force_type = Force<exec_space, force_model_type>;
    using neigh_iter_tag = Cabana::SerialOpTag;

    SolverFractureAlso( Inputs _inputs,
                        std::shared_ptr<particle_type> _particles,
                        force_model_type force_model )
        : base_type( _inputs, _particles,
                     typename force_model_type::elastic_model{} )
    {
        std::ofstream out( inputs->output_file, std::ofstream::app );
        std::ofstream err( inputs->error_file, std::ofstream::app );

        auto time = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now() );
        log( out, "CabanaPD (", std::ctime( &time ), ")\n" );
        if ( print_rank() )
            exec_space::print_configuration( out );

        // Create View to track broken bonds.
        int max_neighbors =
            Cabana::NeighborList<neighbor_type>::maxNeighbor( *neighbors );
        mu = NeighborView(
            Kokkos::ViewAllocateWithoutInitializing( "broken_bonds" ),
            particles->n_local, max_neighbors );
        Kokkos::deep_copy( mu, 1 );
        std::cout << mu.extent( 0 ) << " " << mu.extent( 1 ) << std::endl;

        // Create force.
        force = std::make_shared<force_type>( inputs->half_neigh, force_model );

        Cajita::Experimental::SiloParticleOutput::writePartialRangeTimeStep(
            "particles", particles->local_grid->globalGrid(), 0, 0, 0,
            particles->n_local, particles->slice_x(), particles->slice_W(),
            particles->slice_f(), particles->slice_u(), particles->slice_v(),
            particles->slice_phi() );

        log( out, "Nlocal Nghost Nglobal\n", particles->n_local, " ",
             particles->n_ghost, " ", particles->n_global );
        init_time += init_timer.seconds();
        out.close();
    }

    void init_force()
    {
        // Only needed for LPS.
        force->initialize( *particles, *neighbors, neigh_iter_tag{} );

        // Compute initial forces
        auto f = particles->slice_f();
        Cabana::deep_copy( f, 0.0 );
        compute_force( *force, *particles, *neighbors, mu, neigh_iter_tag{} );
        compute_energy( *force, *particles, *neighbors, mu, neigh_iter_tag() );
    }

    void run()
    {
        // Main timestep loop
        for ( int step = 1; step <= num_steps; step++ )
        {
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
            compute_force( *force, *particles, *neighbors, mu,
                           neigh_iter_tag{} );
            force_time += force_timer.seconds();

            // Integrate - velocity Verlet second half
            integrate_timer.reset();
            integrator->final_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Print output
            other_timer.reset();
            if ( step % output_frequency == 0 )
            {
                auto W = compute_energy( *force, *particles, *neighbors, mu,
                                         neigh_iter_tag() );

                this->step_output( step, W );

                auto x = particles->slice_x();
                Cajita::Experimental::SiloParticleOutput::
                    writePartialRangeTimeStep(
                        "particles", particles->local_grid->globalGrid(),
                        step / output_frequency, step * inputs->timestep, 0,
                        particles->n_local, x, particles->slice_W(),
                        particles->slice_f(), particles->slice_u(),
                        particles->slice_v(), particles->slice_phi() );
            }
            other_time += other_timer.seconds();
        }

        // Final output and timings
        this->final_output();
    }

    using base_type::num_steps;
    using base_type::output_frequency;

  protected:
    using base_type::comm;
    using base_type::inputs;
    using base_type::integrator;
    using base_type::neighbors;
    using base_type::particles;
    std::shared_ptr<force_type> force;

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
};

template <class DeviceType, class ForceModel, class BoundaryCondition,
          class PrenotchType>
class SolverFracture : public SolverElastic<DeviceType, ForceModel>
{
  public:
    using base_type = SolverElastic<DeviceType, ForceModel>;
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

    SolverFracture( Inputs _inputs, std::shared_ptr<particle_type> _particles,
                    force_model_type force_model, bc_type bc,
                    prenotch_type prenotch )
        : base_type( _inputs, _particles, force_model )
        , boundary_condition( bc )
    {
        std::ofstream out( inputs->output_file, std::ofstream::app );
        std::ofstream err( inputs->error_file, std::ofstream::app );

        // Create View to track broken bonds.
        int max_neighbors =
            Cabana::NeighborList<neighbor_type>::maxNeighbor( *neighbors );
        mu = NeighborView(
            Kokkos::ViewAllocateWithoutInitializing( "broken_bonds" ),
            particles->n_local, max_neighbors );
        Kokkos::deep_copy( mu, 1 );

        // Create prenotch.
        prenotch.create( exec_space{}, mu, *particles, *neighbors );
    }

    void init_force()
    {
        // Compute weighted volume for LPS (does nothing for PMB).
        force->compute_weighted_volume( *particles, *neighbors, mu );
        comm->gatherWeightedVolume();
        // Compute dilatation for LPS (does nothing for PMB).
        force->compute_dilatation( *particles, *neighbors, mu );
        // Communicate dilatation for LPS (FIXME: should not be done for PMB).
        comm->gatherDilatation();

        // Compute initial forces
        compute_force( *force, *particles, *neighbors, mu, neigh_iter_tag{} );
        compute_energy( *force, *particles, *neighbors, mu, neigh_iter_tag() );

        // Add boundary condition - resetting boundary forces to zero.
        boundary_condition.apply( exec_space(), *particles );

        particle_output( 0 );
    }

    void run()
    {
        // Main timestep loop
        for ( int step = 1; step <= num_steps; step++ )
        {
            // Integrate - velocity Verlet first half
            integrate_timer.reset();
            integrator->initial_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Update ghost particles.
            comm_timer.reset();
            comm->gatherDisplacement();
            comm_time += comm_timer.seconds();

            // Reset forces
            force_timer.reset();
            // Compute weighted volume for LPS (does nothing for PMB).
            force->compute_weighted_volume( *particles, *neighbors, mu );
            comm->gatherWeightedVolume();
            // Compute dilatation for LPS (does nothing for PMB).
            force->compute_dilatation( *particles, *neighbors, mu );
            // Communicate dilatation for LPS (FIXME: should not be done for
            // PMB).
            comm->gatherDilatation();

            // Compute short range force
            compute_force( *force, *particles, *neighbors, mu,
                           neigh_iter_tag{} );
            force_time += force_timer.seconds();

            // Add boundary condition - resetting boundary forces to zero.
            boundary_condition.apply( exec_space{}, *particles );

            // Integrate - velocity Verlet second half
            integrate_timer.reset();
            integrator->final_integrate( *particles );
            integrate_time += integrate_timer.seconds();

            // Print output
            other_timer.reset();
            if ( step % output_frequency == 0 )
            {
                auto W = compute_energy( *force, *particles, *neighbors, mu,
                                         neigh_iter_tag() );

                this->step_output( step, W );
                particle_output( step );
            }
            other_time += other_timer.seconds();
        }

        // Final output and timings
        this->final_output();
    }

    void particle_output( const int step )
    {
        Cajita::Experimental::SiloParticleOutput::writePartialRangeTimeStep(
            "particles", particles->local_grid->globalGrid(),
            step / output_frequency, step * inputs->timestep, 0,
            particles->n_local, particles->slice_x(), particles->slice_W(),
            particles->slice_f(), particles->slice_u(), particles->slice_v(),
            particles->slice_phi(), particles->slice_theta(),
            particles->slice_m() );
    }

    using base_type::num_steps;
    using base_type::output_frequency;

  protected:
    using base_type::comm;
    using base_type::force;
    using base_type::inputs;
    using base_type::integrator;
    using base_type::neighbors;
    using base_type::particles;
    bc_type boundary_condition;

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
};

template <class DeviceType, class ParticleType, class ForceModel>
auto createSolverElastic( Inputs inputs, ParticleType particles,
                          ForceModel model )
{
    return std::make_shared<SolverElastic<DeviceType, ForceModel>>(
        inputs, particles, model );
}

template <class DeviceType, class ParticleType, class ForceModel>
auto createSolverFractureAlso( Inputs inputs, ParticleType particles,
                               ForceModel model )
{
    return std::make_shared<SolverFractureAlso<DeviceType, ForceModel>>(
        inputs, particles, model );
}

template <class DeviceType, class ParticleType, class ForceModel, class BCType,
          class PrenotchType>
auto createSolverFracture( Inputs inputs, ParticleType particles,
                           ForceModel model, BCType bc, PrenotchType prenotch )
{
    return std::make_shared<
        SolverFracture<DeviceType, ForceModel, BCType, PrenotchType>>(
        inputs, particles, model, bc, prenotch );
}

/*
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
*/
} // namespace CabanaPD

#endif
