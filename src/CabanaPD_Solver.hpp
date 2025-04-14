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
#include <CabanaPD_HeatTransfer.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Integrate.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Prenotch.hpp>
#include <CabanaPD_Timer.hpp>
#include <CabanaPD_Types.hpp>
#include <force_models/CabanaPD_Contact.hpp>

namespace CabanaPD
{
template <class ParticleType, class ForceModelType,
          class ContactModelType = NoContact>
class Solver
{
  public:
    using memory_space = typename ParticleType::memory_space;
    using exec_space = typename memory_space::execution_space;

    // Core module types - required for all problems.
    using force_model_type = ForceModelType;
    using force_type = Force<memory_space, force_model_type,
                             typename force_model_type::model_type,
                             typename force_model_type::fracture_type,
                             typename force_model_type::density_type>;
    using comm_type =
        Comm<ParticleType, typename force_model_type::base_model::base_type,
             typename ParticleType::thermal_type>;
    using neighbor_type =
        Neighbor<memory_space, typename force_model_type::fracture_type>;

    // Optional module types.
    using heat_transfer_type = HeatTransfer<memory_space, force_model_type>;
    using contact_type = Force<memory_space, ContactModelType, ContactModel>;
    using contact_model_type = ContactModelType;

    // Flexible module types.
    // Integration should include max displacement tracking if either model
    // involves contact.
    using integrator_type =
        VelocityVerlet<typename either_contact<force_model_type,
                                               contact_model_type>::base_type>;

    Solver( Inputs _inputs, ParticleType _particles,
            force_model_type force_model )
        : particles( _particles )
        , inputs( _inputs )
        , _init_time( 0.0 )
    {
        setup( force_model );
    }

    Solver( Inputs _inputs, ParticleType _particles,
            force_model_type force_model, contact_model_type contact_model )
        : particles( _particles )
        , inputs( _inputs )
        , _init_time( 0.0 )
    {
        setup( force_model );

        contact = std::make_shared<contact_type>( contact_model );
    }

    void setup( force_model_type force_model )
    {
        // This timestep is not valid for DEM-only.
        if constexpr ( !is_contact<force_model_type>::value )
            inputs.computeCriticalTimeStep( force_model );

        num_steps = inputs["num_steps"];
        output_frequency = inputs["output_frequency"];
        output_reference = inputs["output_reference"];

        // Create integrator.
        dt = inputs["timestep"];
        integrator = std::make_shared<integrator_type>( dt );

        // Add ghosts from other MPI ranks.
        comm = std::make_shared<comm_type>( particles );

        if constexpr ( is_contact<contact_model_type>::value )
        {
            if ( comm->size() > 1 )
                throw std::runtime_error(
                    "Contact with MPI is currently disabled." );
        }

        // Update temperature ghost size if needed.
        if constexpr ( is_temperature_dependent<
                           typename force_model_type::thermal_type>::value )
            force_model.update( particles );

        neighbor = std::make_shared<neighbor_type>( inputs["half_neigh"],
                                                    force_model, particles );
        // Plastic models need correct size for the bond array.
        force_model.updateBonds( particles.localOffset(),
                                 neighbor->getMaxLocal() );

        // This will either be PD or DEM forces.
        force = std::make_shared<force_type>( force_model );

        _init_timer.start();
        unsigned max_neighbors;
        unsigned long long total_neighbors;
        neighbor->getStatistics( max_neighbors, total_neighbors );

        // Create heat transfer if needed, using the same neighbor list as
        // the mechanics.
        if constexpr ( is_heat_transfer<
                           typename force_model_type::thermal_type>::value )
        {
            thermal_subcycle_steps = inputs["thermal_subcycle_steps"];
            heat_transfer = std::make_shared<heat_transfer_type>( force_model );
        }

        print = print_rank();
        if ( print )
        {
            log( std::cout, "Local particles: ", particles.numLocal(),
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
                 particles.numLocal(), ", ", particles.numGhost(), ", ",
                 particles.numGlobal() );
            log( out, "Maximum neighbors: ", max_neighbors,
                 ", Total neighbors: ", total_neighbors, "\n" );
            out.close();
        }
        _init_timer.stop();
    }

    void init( const bool initial_output = true )
    {
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB). Only computed once without fracture (and inside updateForce for
        // fracture).
        if constexpr ( !is_fracture<
                           typename force_model_type::fracture_type>::value )
        {
            force->computeWeightedVolume( particles, *neighbor );
            comm->gatherWeightedVolume();
        }
        // Compute initial internal forces and energy.
        updateForce();
        computeEnergy( *force, particles, *neighbor );

        if ( initial_output )
            particles.output( 0, 0.0, output_reference );
    }

    template <typename BoundaryType>
    void init( BoundaryType boundary_condition,
               const bool initial_output = true )
    {
        // Add non-force boundary condition.
        if ( !boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, 0.0 );

        // Communicate temperature.
        if constexpr ( is_temperature_dependent<
                           typename force_model_type::thermal_type>::value )
            comm->gatherTemperature();

        // Force init without particle output.
        init( false );

        // Add force boundary condition.
        if ( boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, 0.0 );

        if ( initial_output )
            particles.output( 0, 0.0, output_reference );
    }

    // Initialize with prenotch, but no BC.
    template <std::size_t NumPrenotch>
    void init( Prenotch<NumPrenotch> prenotch,
               const bool initial_output = true )
    {
        init_prenotch( prenotch );
        init( initial_output );
    }

    // Initialize with prenotch and BC.
    template <typename BoundaryType, std::size_t NumPrenotch>
    void init( BoundaryType boundary_condition, Prenotch<NumPrenotch> prenotch,
               const bool initial_output = true )
    {
        init_prenotch( prenotch );
        init( boundary_condition, initial_output );
    }

    // Given a user functor, remove certain particles.
    void remove( const double threshold, const bool use_frozen = false )
    {
        // Remove any points that are too close.
        Kokkos::View<int*, memory_space> keep( "keep_points",
                                               particles.numLocal() );
        Kokkos::deep_copy( keep, 1 );
        auto f = particles.sliceForce();
        std::size_t min_particle = particles.numFrozen();
        if ( use_frozen )
            min_particle = 0;
        auto remove_functor = KOKKOS_LAMBDA( const int pid, int& k )
        {
            auto f_mag = Kokkos::hypot( f( pid, 0 ), f( pid, 1 ), f( pid, 2 ) );
            if ( f_mag > threshold )
                keep( pid - min_particle ) = 0;
            else
                k++;
        };

        int num_keep;
        Kokkos::RangePolicy<exec_space> policy( min_particle,
                                                particles.localOffset() );
        Kokkos::parallel_reduce( "remove", policy, remove_functor,
                                 Kokkos::Sum<int>( num_keep ) );

        particles.remove( num_keep, keep );
        // FIXME: Will need to rebuild ghosts.
    }

    void updateNeighbors() { force->update( particles, 0.0, true ); }

    template <typename BoundaryType>
    void run( BoundaryType boundary_condition )
    {
        init_output( boundary_condition.timeInit() );

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            _step_timer.start();

            // Integrate - velocity Verlet first half.
            integrator->initialHalfStep( exec_space{}, particles );

            // Update ghost particles.
            comm->gatherDisplacement();

            if constexpr ( is_heat_transfer<
                               typename force_model_type::thermal_type>::value )
            {
                if ( step % thermal_subcycle_steps == 0 )
                    computeHeatTransfer( *heat_transfer, particles, *neighbor,
                                         thermal_subcycle_steps * dt );
            }

            // Add non-force boundary condition.
            if ( !boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space(), particles, step * dt );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, particles, *neighbor, false );

            // Add force boundary condition.
            if ( boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space(), particles, step * dt );

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( exec_space{}, particles );

            output( step );
        }

        // Final output and timings.
        final_output();
    }

    void run()
    {
        init_output();

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            _step_timer.start();

            // Integrate - velocity Verlet first half.
            integrator->initialHalfStep( exec_space{}, particles );

            // Update ghost particles.
            comm->gatherDisplacement();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, particles, *neighbor, false );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( exec_space{}, particles );

            output( step );
        }

        // Final output and timings.
        final_output();
    }

    // Compute and communicate fields needed for force computation and update
    // forces.
    void updateForce()
    {
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB). Only computed once without fracture.
        if constexpr ( is_fracture<
                           typename force_model_type::fracture_type>::value )
        {
            force->computeWeightedVolume( particles, *neighbor );
            comm->gatherWeightedVolume();
        }
        // Compute and communicate dilatation for LPS (does nothing for PMB).
        force->computeDilatation( particles, *neighbor );
        comm->gatherDilatation();

        // Compute internal forces.
        computeForce( *force, particles, *neighbor );
    }

    void output( const int step )
    {
        // Print output.
        if ( step % output_frequency == 0 )
        {
            auto W = computeEnergy( *force, particles, *neighbor );

            particles.output( step / output_frequency, step * dt,
                              output_reference );
            _step_timer.stop();
            step_output( step, W );
        }
        else
        {
            _step_timer.stop();
        }
    }

    void init_output( double boundary_init_time = 0.0 )
    {
        // Output after construction and initial forces.
        std::ofstream out( output_file, std::ofstream::app );
        _init_time += _init_timer.time() + neighbor->time() +
                      particles.timeInit() + comm->timeInit() +
                      integrator->timeInit() + boundary_init_time;
        log( out, "Init-Time(s): ", _init_time );
        log( out, "Init-Neighbor-Time(s): ", neighbor->time(), "\n" );
        log( out, "#Timestep/Total-steps Simulation-time Total-strain-energy "
                  "Step-Time(s) Force-Time(s) Comm-Time(s) Integrate-Time(s) "
                  "Energy-Time(s) Output-Time(s) Particle*steps/s" );
    }

    void step_output( const int step, const double W )
    {
        if ( print )
        {
            std::ofstream out( output_file, std::ofstream::app );
            log( std::cout, step, "/", num_steps, " ", std::scientific,
                 std::setprecision( 2 ), step * dt );

            double step_time = _step_timer.time();
            double comm_time = comm->time();
            double integrate_time = integrator->time();
            double force_time = force->time();
            double energy_time = force->timeEnergy();
            double neigh_time = neighbor->time();
            double output_time = particles.timeOutput();
            _total_time += step_time;
            // Instantaneous rate.
            double p_steps_per_sec =
                static_cast<double>( particles.numGlobal() ) / step_time;

            _step_timer.reset();
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", std::fixed, _total_time, " ", force_time, " ",
                 comm_time, " ", integrate_time, " ", energy_time, " ",
                 neigh_time, " ", output_time, " ", std::scientific,
                 p_steps_per_sec );
            out.close();
        }
    }

    void final_output()
    {
        if ( print )
        {
            std::ofstream out( output_file, std::ofstream::app );
            double comm_time = comm->time();
            double integrate_time = integrator->time();
            double force_time = force->time();
            double energy_time = force->timeEnergy();
            double output_time = particles.timeOutput();
            double neighbor_time = neighbor->time();
            _total_time = _init_time + comm_time + integrate_time + force_time +
                          energy_time + output_time + particles.time();

            // Rates over the whole simulation.
            double steps_per_sec =
                static_cast<double>( num_steps ) / _total_time;
            double p_steps_per_sec =
                static_cast<double>( particles.numGlobal() ) * steps_per_sec;
            log( out, std::fixed, std::setprecision( 2 ),
                 "\n#Procs Particles | Total Force Comm Integrate Energy "
                 "Output Init Init_Neighbor |\n",
                 comm->mpi_size, " ", particles.numGlobal(), " | \t",
                 _total_time, " ", force_time, " ", comm_time, " ",
                 integrate_time, " ", energy_time, " ", output_time, " ",
                 _init_time, " ", neighbor_time, " | PERFORMANCE\n", std::fixed,
                 comm->mpi_size, " ", particles.numGlobal(), " | \t", 1.0, " ",
                 force_time / _total_time, " ", comm_time / _total_time, " ",
                 integrate_time / _total_time, " ", energy_time / _total_time,
                 " ", output_time / _total_time, " ", _init_time / _total_time,
                 " ", neighbor_time / _total_time, " | FRACTION\n\n",
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
    int thermal_subcycle_steps;
    // Sometimes necessary to update particles after solver creation.
    ParticleType particles;

  protected:
    template <std::size_t NumPrenotch>
    void init_prenotch( Prenotch<NumPrenotch> prenotch )
    {
        static_assert(
            is_fracture<typename force_model_type::fracture_type>::value,
            "Cannot create prenotch in system without fracture." );

        // Create prenotch.
        neighbor->prenotch( exec_space{}, particles, prenotch );
        _init_time += prenotch.time();
    }

    // Core modules.
    Inputs inputs;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<neighbor_type> neighbor;
    std::shared_ptr<force_type> force;
    // Optional modules.
    std::shared_ptr<heat_transfer_type> heat_transfer;
    std::shared_ptr<contact_type> contact;

    // Output files.
    std::string output_file;
    std::string error_file;

    // Note: init_time is combined from many class timers.
    double _init_time;
    Timer _init_timer;
    Timer _step_timer;
    double _total_time;
    bool print;
};

} // namespace CabanaPD

#endif
