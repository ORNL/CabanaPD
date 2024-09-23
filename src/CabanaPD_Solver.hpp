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

namespace CabanaPD
{
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void run() = 0;
};

template <class MemorySpace, class InputType, class ParticleType,
          class ForceModelType, class ContactModelType = NoContact>
class SolverNoFracture
{
  public:
    using memory_space = MemorySpace;
    using exec_space = typename memory_space::execution_space;

    // Core module types - required for all problems.
    using particle_type = ParticleType;
    using integrator_type = Integrator<exec_space>;
    using force_model_type = ForceModelType;
    using force_type = Force<memory_space, force_model_type>;
    using comm_type = Comm<particle_type, typename force_model_type::base_model,
                           typename particle_type::thermal_type>;
    using neigh_iter_tag = Cabana::SerialOpTag;
    using input_type = InputType;

    // Optional module types.
    using heat_transfer_type = HeatTransfer<memory_space, force_model_type>;
    using contact_type = Force<memory_space, ContactModelType>;
    using contact_model_type = ContactModelType;

    SolverNoFracture( input_type _inputs,
                      std::shared_ptr<particle_type> _particles,
                      force_model_type force_model )
        : inputs( _inputs )
        , particles( _particles )
        , _init_time( 0.0 )
    {
        setup( force_model );
    }

    SolverNoFracture( input_type _inputs,
                      std::shared_ptr<particle_type> _particles,
                      force_model_type force_model,
                      contact_model_type contact_model )
        : inputs( _inputs )
        , particles( _particles )
        , _init_time( 0.0 )
    {
        setup( force_model );

        _neighbor_timer.start();
        contact = std::make_shared<contact_type>( inputs["half_neigh"],
                                                  *particles, contact_model );
        _neighbor_timer.stop();
    }

    void setup( force_model_type force_model )
    {
        num_steps = inputs["num_steps"];
        output_frequency = inputs["output_frequency"];
        output_reference = inputs["output_reference"];

        // Create integrator.
        dt = inputs["timestep"];
        integrator = std::make_shared<integrator_type>( dt );

        // Add ghosts from other MPI ranks.
        comm = std::make_shared<comm_type>( *particles );

        if constexpr ( is_contact<contact_model_type>::value )
        {
            if ( comm->size() > 1 )
                throw std::runtime_error(
                    "Contact with MPI is currently disabled." );
        }

        // Update temperature ghost size if needed.
        if constexpr ( is_temperature_dependent<
                           typename force_model_type::thermal_type>::value )
            force_model.update( particles->sliceTemperature() );

        // Create heat transfer if needed.
        if constexpr ( is_heat_transfer<
                           typename force_model_type::thermal_type>::value )
        {
            thermal_subcycle_steps = inputs["thermal_subcycle_steps"];
            heat_transfer = std::make_shared<heat_transfer_type>(
                inputs["half_neigh"], force->_neigh_list, force_model );
        }

        _neighbor_timer.start();
        // This will either be PD or DEM forces.
        force = std::make_shared<force_type>( inputs["half_neigh"], *particles,
                                              force_model );
        _neighbor_timer.stop();

        _init_timer.start();
        unsigned max_neighbors;
        unsigned long long total_neighbors;
        force->getNeighborStatistics( max_neighbors, total_neighbors );

        // Create heat transfer if needed.
        if constexpr ( is_heat_transfer<
                           typename force_model_type::thermal_type>::value )
        {
            thermal_subcycle_steps = inputs["thermal_subcycle_steps"];
            heat_transfer = std::make_shared<heat_transfer_type>(
                inputs["half_neigh"], force->_neigh_list, force_model );
        }

        print = print_rank();
        if ( print )
        {
            log( std::cout, "Local particles: ", particles->numLocal(),
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
                 particles->numLocal(), ", ", particles->numGhost(), ", ",
                 particles->numGlobal() );
            log( out, "Maximum neighbors: ", max_neighbors,
                 ", Total neighbors: ", total_neighbors, "\n" );
            out.close();
        }
        _init_timer.stop();
    }

    void init( const bool initial_output = true )
    {
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB). Only computed once.
        force->computeWeightedVolume( *particles, neigh_iter_tag{} );
        comm->gatherWeightedVolume();

        // Compute initial internal forces and energy.
        updateForce();
        computeEnergy( *force, *particles, neigh_iter_tag() );

        if ( initial_output )
            particles->output( 0, 0.0, output_reference );
    }

    template <typename BoundaryType>
    void init( BoundaryType boundary_condition,
               const bool initial_output = true )
    {
        // Add non-force boundary condition.
        if ( !boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), *particles, 0.0 );

        // Communicate temperature.
        if constexpr ( is_temperature_dependent<
                           typename force_model_type::thermal_type>::value )
            comm->gatherTemperature();

        // Force init without particle output.
        init( false );

        // Add force boundary condition.
        if ( boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), *particles, 0.0 );

        if ( initial_output )
            particles->output( 0, 0.0, output_reference );
    }

    template <typename BoundaryType>
    void run( BoundaryType boundary_condition )
    {
        init_output( boundary_condition.timeInit() );

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            _step_timer.start();

            // Integrate - velocity Verlet first half.
            integrator->initialHalfStep( *particles );

            // Update ghost particles.
            comm->gatherDisplacement();

            if constexpr ( is_heat_transfer<
                               typename force_model_type::thermal_type>::value )
            {
                if ( step % thermal_subcycle_steps == 0 )
                    computeHeatTransfer( *heat_transfer, *particles,
                                         neigh_iter_tag{},
                                         thermal_subcycle_steps * dt );
            }

            // Add non-force boundary condition.
            if ( !boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space(), *particles, step * dt );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, *particles, neigh_iter_tag{}, false );

            // Add force boundary condition.
            if ( boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space(), *particles, step * dt );

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( *particles );

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
            integrator->initialHalfStep( *particles );

            // Update ghost particles.
            comm->gatherDisplacement();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, *particles, neigh_iter_tag{}, false );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( *particles );

            output( step );
        }

        // Final output and timings.
        final_output();
    }

    // Compute and communicate fields needed for force computation and update
    // forces.
    void updateForce()
    {
        // Compute and communicate dilatation for LPS (does nothing for PMB).
        force->computeDilatation( *particles, neigh_iter_tag{} );
        comm->gatherDilatation();

        // Compute internal forces.
        computeForce( *force, *particles, neigh_iter_tag{} );
    }

    void output( const int step )
    {
        // Print output.
        if ( step % output_frequency == 0 )
        {
            auto W = computeEnergy( *force, *particles, neigh_iter_tag() );

            particles->output( step / output_frequency, step * dt,
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
        _init_time += _init_timer.time() + _neighbor_timer.time() +
                      particles->timeInit() + comm->timeInit() +
                      integrator->timeInit() + boundary_init_time;
        log( out, "Init-Time(s): ", _init_time );
        log( out, "Init-Neighbor-Time(s): ", _neighbor_timer.time(), "\n" );
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
            double output_time = particles->timeOutput();
            _total_time += step_time;
            auto rate = static_cast<double>( particles->numGlobal() *
                                             output_frequency / ( step_time ) );
            _step_timer.reset();
            log( out, std::fixed, std::setprecision( 6 ), step, "/", num_steps,
                 " ", std::scientific, std::setprecision( 2 ), step * dt, " ",
                 W, " ", std::fixed, _total_time, " ", force_time, " ",
                 comm_time, " ", integrate_time, " ", energy_time, " ",
                 output_time, " ", std::scientific, rate );
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
            double output_time = particles->timeOutput();
            double neighbor_time = _neighbor_timer.time();
            _total_time = _init_time + comm_time + integrate_time + force_time +
                          energy_time + output_time + particles->time();

            double steps_per_sec = 1.0 * num_steps / _total_time;
            double p_steps_per_sec = particles->numGlobal() * steps_per_sec;
            log( out, std::fixed, std::setprecision( 2 ),
                 "\n#Procs Particles | Total Force Comm Integrate Energy "
                 "Output Init Init_Neighbor |\n",
                 comm->mpi_size, " ", particles->numGlobal(), " | \t",
                 _total_time, " ", force_time, " ", comm_time, " ",
                 integrate_time, " ", energy_time, " ", output_time, " ",
                 _init_time, " ", neighbor_time, " | PERFORMANCE\n", std::fixed,
                 comm->mpi_size, " ", particles->numGlobal(), " | \t", 1.0, " ",
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

  protected:
    // Core modules.
    input_type inputs;
    std::shared_ptr<particle_type> particles;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<force_type> force;
    // Optional modules.
    std::shared_ptr<heat_transfer_type> heat_transfer;
    std::shared_ptr<contact_type> contact;
    contact_model_type contact_model;

    // Output files.
    std::string output_file;
    std::string error_file;

    // Note: init_time is combined from many class timers.
    double _init_time;
    Timer _init_timer;
    Timer _neighbor_timer;
    Timer _step_timer;
    double _total_time;
    bool print;
};

template <class MemorySpace, class InputType, class ParticleType,
          class ForceModelType, class ContactModelType = NoContact>
class SolverFracture
    : public SolverNoFracture<MemorySpace, InputType, ParticleType,
                              ForceModelType, ContactModelType>
{
  public:
    using base_type = SolverNoFracture<MemorySpace, InputType, ParticleType,
                                       ForceModelType, ContactModelType>;
    using exec_space = typename base_type::exec_space;
    using memory_space = typename base_type::memory_space;

    using particle_type = typename base_type::particle_type;
    using integrator_type = typename base_type::integrator_type;
    using comm_type = typename base_type::comm_type;
    using force_model_type = ForceModelType;
    using force_type = typename base_type::force_type;
    using neigh_iter_tag = Cabana::SerialOpTag;
    using input_type = typename base_type::input_type;

    using contact_model_type = ContactModelType;

    SolverFracture( input_type _inputs,
                    std::shared_ptr<particle_type> _particles,
                    force_model_type force_model )
        : base_type( _inputs, _particles, force_model )
    {
        init_mu();
    }

    SolverFracture( input_type _inputs,
                    std::shared_ptr<particle_type> _particles,
                    force_model_type force_model,
                    contact_model_type contact_model )
        : base_type( _inputs, _particles, force_model, contact_model )
    {
        init_mu();
    }

    void init_mu()
    {
        _init_timer.start();
        // Create View to track broken bonds.
        auto max_neighbors = force->getMaxLocalNeighbors();
        // TODO: this could be optimized to ignore frozen particle bonds.
        mu = NeighborView(
            Kokkos::ViewAllocateWithoutInitializing( "broken_bonds" ),
            particles->localOffset(), max_neighbors );
        Kokkos::deep_copy( mu, 1 );
        _init_timer.stop();
    }

    template <std::size_t NumPrenotch>
    void init_prenotch( Prenotch<NumPrenotch> prenotch )
    {
        // Create prenotch.
        prenotch.create( exec_space{}, mu, *particles, force->_neigh_list );
        _init_time += prenotch.time();
    }

    void init( const bool initial_output = true )
    {
        // Compute initial internal forces and energy.
        updateForce();
        computeEnergy( *force, *particles, mu, neigh_iter_tag() );

        if ( initial_output )
            particles->output( 0, 0.0, output_reference );
    }

    template <std::size_t NumPrenotch>
    void init( Prenotch<NumPrenotch> prenotch,
               const bool initial_output = true )
    {
        init_prenotch( prenotch );
        init( initial_output );
    }

    template <typename BoundaryType>
    void init( BoundaryType boundary_condition,
               const bool initial_output = true )
    {
        // Add non-force boundary condition.
        if ( !boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), *particles, 0.0 );

        // Communicate temperature.
        if constexpr ( is_temperature_dependent<
                           typename force_model_type::thermal_type>::value )
            comm->gatherTemperature();

        // Force init without particle output.
        init( false );

        // Add force boundary condition.
        if ( boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), *particles, 0.0 );

        if ( initial_output )
            particles->output( 0, 0.0, output_reference );
    }

    template <typename BoundaryType, std::size_t NumPrenotch>
    void init( BoundaryType boundary_condition, Prenotch<NumPrenotch> prenotch,
               const bool initial_output = true )
    {
        init_prenotch( prenotch );
        init( boundary_condition, initial_output );
    }

    template <typename BoundaryType>
    void run( BoundaryType boundary_condition )
    {
        this->init_output( boundary_condition.timeInit() );

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            _step_timer.start();

            // Integrate - velocity Verlet first half.
            integrator->initialHalfStep( *particles );

            // Add non-force boundary condition.
            if ( !boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space(), *particles, step * dt );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Update ghost particles.
            comm->gatherDisplacement();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, *particles, neigh_iter_tag{}, false );

            // Add force boundary condition.
            if ( boundary_condition.forceUpdate() )
                boundary_condition.apply( exec_space{}, *particles, step * dt );

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( *particles );

            output( step );
        }

        // Final output and timings.
        this->final_output();
    }

    void run()
    {
        this->init_output( 0.0 );

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            _step_timer.start();

            // Integrate - velocity Verlet first half.
            integrator->initialHalfStep( *particles );

            if constexpr ( is_temperature_dependent<
                               typename force_model_type::thermal_type>::value )
                comm->gatherTemperature();

            // Update ghost particles.
            comm->gatherDisplacement();

            // Compute internal forces.
            updateForce();

            if constexpr ( is_contact<contact_model_type>::value )
                computeForce( *contact, *particles, neigh_iter_tag{}, false );

            // Integrate - velocity Verlet second half.
            integrator->finalHalfStep( *particles );

            output( step );
        }

        // Final output and timings.
        this->final_output();
    }

    // Compute and communicate fields needed for force computation and update
    // forces.
    void updateForce()
    {
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB).
        force->computeWeightedVolume( *particles, mu );
        comm->gatherWeightedVolume();

        // Compute and communicate dilatation for LPS (does nothing for PMB).
        force->computeDilatation( *particles, mu );
        comm->gatherDilatation();

        // Compute internal forces.
        computeForce( *force, *particles, mu, neigh_iter_tag{} );
    }

    void output( const int step )
    {
        // Print output.
        if ( step % output_frequency == 0 )
        {
            auto W = computeEnergy( *force, *particles, mu, neigh_iter_tag() );

            particles->output( step / output_frequency, step * dt,
                               output_reference );
            _step_timer.stop();
            this->step_output( step, W );
        }
        else
        {
            _step_timer.stop();
        }
    }

    using base_type::dt;
    using base_type::num_steps;
    using base_type::output_frequency;
    using base_type::output_reference;

  protected:
    using base_type::comm;
    using base_type::contact;
    using base_type::force;
    using base_type::inputs;
    using base_type::integrator;
    using base_type::particles;

    using NeighborView = typename Kokkos::View<int**, memory_space>;
    NeighborView mu;

    using base_type::_init_time;
    using base_type::_init_timer;
    using base_type::_step_timer;
    using base_type::print;
};

// ===============================================================

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModelType>
auto createSolverNoFracture( InputsType inputs,
                             std::shared_ptr<ParticleType> particles,
                             ForceModelType model )
{
    return std::make_shared<SolverNoFracture<MemorySpace, InputsType,
                                             ParticleType, ForceModelType>>(
        inputs, particles, model );
}

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModelType, class ContactModelType>
auto createSolverNoFracture( InputsType inputs,
                             std::shared_ptr<ParticleType> particles,
                             ForceModelType model,
                             ContactModelType contact_model )
{
    return std::make_shared<
        SolverNoFracture<MemorySpace, InputsType, ParticleType, ForceModelType,
                         ContactModelType>>( inputs, particles, model,
                                             contact_model );
}

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModelType>
auto createSolverFracture( InputsType inputs,
                           std::shared_ptr<ParticleType> particles,
                           ForceModelType model )
{
    return std::make_shared<
        SolverFracture<MemorySpace, InputsType, ParticleType, ForceModelType>>(
        inputs, particles, model );
}

template <class MemorySpace, class InputsType, class ParticleType,
          class ForceModelType, class ContactModelType>
auto createSolverFracture( InputsType inputs,
                           std::shared_ptr<ParticleType> particles,
                           ForceModelType model,
                           ContactModelType contact_model )
{
    return std::make_shared<
        SolverFracture<MemorySpace, InputsType, ParticleType, ForceModelType,
                       ContactModelType>>( inputs, particles, model,
                                           contact_model );
}

} // namespace CabanaPD

#endif
