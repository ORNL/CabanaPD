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
    using force_model_tag = typename ForceModelType::force_tag;
    using force_fracture_type = typename ForceModelType::fracture_type;
    using force_type =
        Force<memory_space, force_model_tag, force_fracture_type>;
    using force_thermal_type = typename ForceModelType::thermal_type::base_type;
    using comm_type =
        Comm<ParticleType, typename ForceModelType::model_tag::base_type,
             typename ForceModelType::material_type, force_thermal_type>;
    using neighbor_type = Neighbor<memory_space, force_fracture_type>;

    // Optional module types.
    using heat_transfer_type = HeatTransfer<memory_space, force_fracture_type>;
    using contact_model_tag = typename ContactModelType::force_tag;
    using contact_fracture_type = typename ContactModelType::fracture_type;
    using contact_type =
        Force<memory_space, contact_model_tag, contact_fracture_type>;
    using contact_neighbor_type = Neighbor<memory_space, contact_fracture_type>;

    // Flexible module types.
    // Integration should include max displacement tracking if either model
    // involves contact.
    using integrator_type = VelocityVerlet<
        typename either_contact<ForceModelType, ContactModelType>::base_type>;
    // Different timings are output based on models.
    using output_type = TimingOutput<ForceModelType, ContactModelType>;

    Solver( Inputs _inputs, ParticleType _particles,
            ForceModelType _force_model )
        : particles( _particles )
        , inputs( _inputs )
        , force_model( _force_model )
        , _other_time( 0.0 )
        , _total_time( 0.0 )
    {
        _total_timer.start();
        setup();
        _total_timer.stop();
    }

    Solver( Inputs _inputs, ParticleType _particles,
            ForceModelType _force_model, ContactModelType _contact_model )
        : particles( _particles )
        , inputs( _inputs )
        , force_model( _force_model )
        , _other_time( 0.0 )
        , _total_time( 0.0 )
    {
        _total_timer.start();
        setup();

        contact_model = std::make_unique<ContactModelType>( _contact_model );
        contact = std::make_shared<contact_type>();
        contact_neighbor = std::make_shared<contact_neighbor_type>(
            *contact_model, particles );
        _total_timer.stop();
    }

    void setup()
    {
        // This timestep is not valid for DEM-only.
        if constexpr ( !is_contact<ForceModelType>::value )
            inputs.computeCriticalTimeStep( force_model );

        num_steps = inputs["num_steps"];
        output_frequency = inputs["output_frequency"];
        output_reference = inputs["output_reference"];

        // Create integrator.
        dt = inputs["timestep"];
        integrator = std::make_shared<integrator_type>( dt );

        // Add ghosts from other MPI ranks.
        comm = std::make_shared<comm_type>( particles );

        if constexpr ( is_contact<ContactModelType>::value )
        {
            if ( comm->size() > 1 )
                throw std::runtime_error(
                    "Contact with MPI is currently disabled." );
        }

        // Update optional property ghost sizes if needed.
        if constexpr ( std::is_same<typename ForceModelType::material_type,
                                    MultiMaterial>::value ||
                       is_temperature_dependent<
                           typename ForceModelType::thermal_type>::value )
            force_model.update( particles );

        neighbor = std::make_shared<neighbor_type>( force_model, particles );
        // Plastic models need correct size for the bond array.
        force_model.updateBonds( particles.localOffset(),
                                 neighbor->getMaxLocal() );

        // This will either be PD or DEM forces.
        force = std::make_shared<force_type>();

        unsigned max_neighbors;
        unsigned long long total_neighbors;
        neighbor->getStatistics( max_neighbors, total_neighbors );

        // Create heat transfer if needed, using the same neighbor list as
        // the mechanics.
        if constexpr ( is_heat_transfer<
                           typename ForceModelType::thermal_type>::value )
        {
            thermal_subcycle_steps = inputs["thermal_subcycle_steps"];
            heat_transfer = std::make_shared<heat_transfer_type>();
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
    }

    void init( const bool initial_output = true )
    {
        _total_timer.start();
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB). Only computed once without fracture (and inside updateForce for
        // fracture).
        if constexpr ( !is_fracture<ForceModelType>::value )
        {
            force->computeWeightedVolume( force_model, particles, *neighbor );
            comm->gatherWeightedVolume();
        }
        // Compute initial internal forces and energy.
        updateForce();
        computeEnergy( force_model, *force, particles, *neighbor );
        computeStress( force_model, *force, particles, *neighbor );

        if ( initial_output )
            particles.output( 0, 0.0, output_reference );
        _total_timer.stop();
    }

    template <typename BoundaryType>
    void init( BoundaryType boundary_condition,
               const bool initial_output = true )
    {
        _total_timer.start();
        // Add non-force boundary condition.
        if ( !boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, 0.0 );

        // Communicate optional properties.
        if constexpr ( std::is_same<typename ForceModelType::material_type,
                                    MultiMaterial>::value )
            comm->gatherMaterial();
        if constexpr ( is_temperature_dependent<
                           typename ForceModelType::thermal_type>::value )
            comm->gatherTemperature();
        _total_timer.stop();

        // Force init without particle output.
        init( false );

        _total_timer.start();
        // Add force boundary condition.
        if ( boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, 0.0 );

        if ( initial_output )
            particles.output( 0, 0.0, output_reference );
        _total_timer.stop();
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
        _total_timer.start();
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
        Kokkos::fence();
        particles.remove( num_keep, keep );
        // FIXME: Will need to rebuild ghosts.
        _total_timer.stop();
        _other_time += _total_timer.lastTime();
    }

    template <typename BoundaryType>
    void runStep( const int step, BoundaryType boundary_condition )
    {
        _step_timer.start();
        // Integrate - velocity Verlet first half.
        integrator->initialHalfStep( exec_space{}, particles );

        // Update ghost particles.
        comm->gatherDisplacement();

        if constexpr ( is_heat_transfer<
                           typename ForceModelType::thermal_type>::value )
        {
            if ( step % thermal_subcycle_steps == 0 )
                computeHeatTransfer( force_model, *heat_transfer, particles,
                                     *neighbor, thermal_subcycle_steps * dt );
        }

        // Add non-force boundary condition.
        if ( !boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, step * dt );

        if constexpr ( is_temperature_dependent<
                           typename ForceModelType::thermal_type>::value )
            comm->gatherTemperature();

        // Compute internal forces.
        updateForce();

        if constexpr ( is_contact<ContactModelType>::value )
            computeForce( *contact_model, *contact, particles,
                          *contact_neighbor, false );

        // Add force boundary condition.
        if ( boundary_condition.forceUpdate() )
            boundary_condition.apply( exec_space(), particles, step * dt );

        // Integrate - velocity Verlet second half.
        integrator->finalHalfStep( exec_space{}, particles );

        // Separate output time.
        _step_timer.stop();
        output( step );
    }

    void runStep( const int step )
    {
        _step_timer.start();

        // Integrate - velocity Verlet first half.
        integrator->initialHalfStep( exec_space{}, particles );

        // Update ghost particles.
        comm->gatherDisplacement();

        // Compute internal forces.
        updateForce();

        if constexpr ( is_contact<ContactModelType>::value )
            computeForce( *contact_model, *contact, particles,
                          *contact_neighbor, false );

        if constexpr ( is_temperature_dependent<
                           typename ForceModelType::thermal_type>::value )
            comm->gatherTemperature();

        // Integrate - velocity Verlet second half.
        integrator->finalHalfStep( exec_space{}, particles );

        // Separate output time.
        _step_timer.stop();
        output( step );
    }

    template <typename... OutputType>
    void run( OutputType&... region_output )
    {
        init_output();
        _total_timer.start();

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            runStep( step );
            // FIXME: not included in timing
            if ( step % output_frequency == 0 )
                updateRegion( region_output... );
        }

        // Final output and timings.
        _total_timer.stop();
        final_output( region_output... );
    }

    template <typename BoundaryType, typename... OutputType>
    void run( BoundaryType& boundary_condition, OutputType&... region_output )
    {
        init_output();
        _total_timer.start();
        _other_time += boundary_condition.timeInit();

        // Main timestep loop.
        for ( int step = 1; step <= num_steps; step++ )
        {
            runStep( step, boundary_condition );
            // FIXME: not included in timing
            if ( step % output_frequency == 0 )
                updateRegion( region_output... );
        }

        // Final output and timings.
        _total_timer.stop();
        final_output( region_output... );
    }

    // Iterate over all regions.
    template <typename... RegionType>
    void updateRegion( RegionType&... region )
    {
        ( updateRegion( region ), ... );
    }
    template <typename RegionType>
    void updateRegion( RegionType& region )
    {
        region.update();
    }

    // Compute and communicate fields needed for force computation and update
    // forces.
    void updateForce()
    {
        // Compute and communicate weighted volume for LPS (does nothing for
        // PMB). Only computed once without fracture.
        if constexpr ( is_fracture<ForceModelType>::value )
        {
            force->computeWeightedVolume( force_model, particles, *neighbor );
            comm->gatherWeightedVolume();
        }
        // Compute and communicate dilatation for LPS (does nothing for PMB).
        force->computeDilatation( force_model, particles, *neighbor );
        comm->gatherDilatation();

        // Compute internal forces.
        computeForce( force_model, *force, particles, *neighbor );
    }

    void output( const int step )
    {
        // Print output.
        if ( step % output_frequency == 0 )
        {
            _step_timer.start();
            computeEnergy( force_model, *force, particles, *neighbor );
            computeStress( force_model, *force, particles, *neighbor );

            particles.output( step / output_frequency, step * dt,
                              output_reference );

            // Timer has to be stopped before printing output.
            _step_timer.stop();
            step_output( step );
        }
    }

    void init_output()
    {
        // Add timings that aren't included elsewhere (particles are created
        // prior to the solver and therefore not included in any other timers).
        double particle_time = particles.timeInit() + particles.time();
        // All time up to now is init.
        double total_init_time = _total_timer.time() + particle_time;
        _step_timer.set( total_init_time );

        // Use total init time here.
        timing_output.header( output_file, total_init_time );
    }

    void step_output( const int step )
    {
        // All ranks must reduce - do this outside the if block since only rank
        // 0 will print below.
        double global_damage = 0.0;
        if constexpr ( is_fracture<ForceModelType>::value )
        {
            global_damage = updateGlobal( force->totalDamage() );
        }
        double relative_damage = global_damage / particles.numGlobal();
        double total_strain_energy = updateGlobal( force->totalStrainEnergy() );

        if ( print )
        {
            double step_time = _step_timer.time();
            // Instantaneous rate.
            double p_steps_per_sec =
                static_cast<double>( particles.numGlobal() ) *
                output_frequency / step_time;

            double contact_time = 0.0;
            double neigh_time = neighbor->time();
            if constexpr ( is_contact<ContactModelType>::value )
            {
                contact_time = contact->time();
                neigh_time += contact_neighbor->time();
            }

            timing_output.step(
                output_file, step, num_steps, dt, total_strain_energy,
                relative_damage, step_time, force->time(), contact_time,
                neigh_time, comm->time(), integrator->time(),
                force->timeEnergy(), particles.timeOutput(), p_steps_per_sec );
        }
    }

    void final_output()
    {
        if ( print )
        {
            // Add timings that aren't included elsewhere (particles are created
            // prior to the solver and therefore not included in any other
            // timers).
            double particle_time = particles.timeInit() + particles.time();
            double total_time = _total_timer.time() + particle_time;
            // Already included in total, but not in other.
            _other_time +=
                comm->timeInit() + integrator->timeInit() + particle_time;

            // Rates over the whole simulation.
            double steps_per_sec =
                static_cast<double>( num_steps ) / total_time;
            double p_steps_per_sec =
                static_cast<double>( particles.numGlobal() ) * steps_per_sec;

            double neigh_time = neighbor->time();
            double contact_time = 0.0;
            double contact_neigh_time = 0.0;
            if constexpr ( is_contact<ContactModelType>::value )
            {
                contact_time = contact->time();
                contact_neigh_time += contact_neighbor->time();
            }

            timing_output.final(
                output_file, comm->mpi_size, particles.numGlobal(), total_time,
                force->time(), contact_time, neigh_time, contact_neigh_time,
                comm->time(), integrator->time(), force->timeEnergy(),
                particles.timeOutput(), _other_time, steps_per_sec,
                p_steps_per_sec );
        }
    }

    template <typename... RegionType>
    void final_output( RegionType... region )
    {
        final_output();
        printRegion( region... );
    }

    // Iterate over all regions.
    template <typename... RegionType>
    void printRegion( RegionType... region )
    {
        ( printRegion( region ), ... );
    }
    template <typename RegionType>
    void printRegion( RegionType region )
    {
        region.print( particles.comm() );
    }

    auto updateGlobal( double local )
    {
        double global = 0.0;
        // Not using Allreduce because global values are only used for printing.
        MPI_Reduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        return global;
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
        _total_timer.start();
        static_assert( is_fracture<ForceModelType>::value,
                       "Cannot create prenotch in system without fracture." );

        // Create prenotch.
        neighbor->prenotch( exec_space{}, particles, prenotch );
        _other_time += prenotch.time();
        _total_timer.stop();
    }

    // Core modules.
    Inputs inputs;
    std::shared_ptr<comm_type> comm;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<neighbor_type> neighbor;
    std::shared_ptr<force_type> force;
    ForceModelType force_model;
    // Optional modules.
    std::shared_ptr<heat_transfer_type> heat_transfer;
    std::shared_ptr<contact_type> contact;
    std::unique_ptr<ContactModelType> contact_model;
    std::shared_ptr<contact_neighbor_type> contact_neighbor;

    // Output files.
    std::string output_file;
    std::string error_file;

    // Note: _other_time is combined from many class timers.
    double _other_time;
    double _total_time;
    Timer _step_timer;
    Timer _total_timer;
    bool print;
    output_type timing_output;
};

} // namespace CabanaPD

#endif
