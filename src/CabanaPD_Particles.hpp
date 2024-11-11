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

#ifndef PARTICLES_H
#define PARTICLES_H

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <CabanaPD_Comm.hpp>
#include <CabanaPD_Fields.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Timer.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <class MemorySpace, class ModelType, class ThermalType,
          int Dimension = 3>
class Particles;

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>
{
  public:
    using self_type =
        Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>;
    using thermal_type = TemperatureIndependent;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    // Per particle.
    unsigned long long int n_global = 0;
    std::size_t n_local = 0;
    std::size_t n_ghost = 0;
    std::size_t size = 0;

    // x, u, f (vector matching system dimension).
    using vector_type = Cabana::MemberTypes<double[dim]>;
    // volume, dilatation, weighted_volume.
    using scalar_type = Cabana::MemberTypes<double>;
    // no-fail.
    using int_type = Cabana::MemberTypes<int>;
    // v, W, rho, damage,  type.
    using other_types =
        Cabana::MemberTypes<double[dim], double, double, double, int>;
    // Potentially needed later: body force (b), ID.

    // FIXME: add vector length.
    // FIXME: enable variable aosoa.
    using aosoa_u_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_y_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_vol_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    using aosoa_nofail_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_other_type = Cabana::AoSoA<other_types, memory_space>;
    // Using grid here for the particle init.
    using plist_x_type =
        Cabana::Grid::ParticleList<memory_space, 1,
                                   CabanaPD::Field::ReferencePosition>;
    using plist_f_type =
        Cabana::ParticleList<memory_space, 1, CabanaPD::Field::Force>;

    // Per type.
    int n_types = 1;

    // Simulation total domain.
    std::array<double, dim> global_mesh_ext;

    // Simulation sub domain (single MPI rank).
    std::array<double, dim> local_mesh_ext;
    std::array<double, dim> local_mesh_lo;
    std::array<double, dim> local_mesh_hi;
    std::array<double, dim> ghost_mesh_lo;
    std::array<double, dim> ghost_mesh_hi;
    std::shared_ptr<
        Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<double, dim>>>
        local_grid;
    Kokkos::Array<double, dim> dx;

    int halo_width;

    // Default constructor.
    Particles()
    {
        _init_timer.start();
        for ( int d = 0; d < dim; d++ )
        {
            global_mesh_ext[d] = 0.0;
            local_mesh_lo[d] = 0.0;
            local_mesh_hi[d] = 0.0;
            ghost_mesh_lo[d] = 0.0;
            ghost_mesh_hi[d] = 0.0;
            local_mesh_ext[d] = 0.0;
        }
        _init_timer.stop();
        resize( 0, 0 );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width )
        : halo_width( max_halo_width )
        , _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        createDomain( low_corner, high_corner, num_cells );
        createParticles( exec_space );
    }

    // Constructor which initializes particles on regular grid with
    // customization.
    template <class ExecSpace, class UserFunctor>
    Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               UserFunctor user_create )
        : halo_width( max_halo_width )
        , _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        createDomain( low_corner, high_corner, num_cells );
        createParticles( exec_space, user_create );
    }

    void createDomain( std::array<double, dim> low_corner,
                       std::array<double, dim> high_corner,
                       const std::array<int, dim> num_cells )
    {
        _init_timer.start();
        // Create the MPI partitions.
        Cabana::Grid::DimBlockPartitioner<dim> partitioner;

        // Create global mesh of MPI partitions.
        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            low_corner, high_corner, num_cells );
        for ( int d = 0; d < 3; d++ )
            dx[d] = global_mesh->cellSize( d );

        std::array<bool, dim> is_periodic;
        for ( int d = 0; d < dim; d++ )
        {
            global_mesh_ext[d] = global_mesh->extent( d );
            is_periodic[d] = false;
        }
        // Create the global grid.
        auto global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, is_periodic, partitioner );

        // Create a local mesh.
        local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
        auto local_mesh =
            Cabana::Grid::createLocalMesh<memory_space>( *local_grid );

        for ( int d = 0; d < dim; d++ )
        {
            local_mesh_lo[d] = local_mesh.lowCorner( Cabana::Grid::Own(), d );
            local_mesh_hi[d] = local_mesh.highCorner( Cabana::Grid::Own(), d );
            ghost_mesh_lo[d] = local_mesh.lowCorner( Cabana::Grid::Ghost(), d );
            ghost_mesh_hi[d] =
                local_mesh.highCorner( Cabana::Grid::Ghost(), d );
            local_mesh_ext[d] = local_mesh.extent( Cabana::Grid::Own(), d );
        }
        _init_timer.stop();
    }

    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
        auto empty = KOKKOS_LAMBDA( const int, const double[dim] )
        {
            return true;
        };
        createParticles( exec_space, empty );
    }

    template <class ExecSpace, class UserFunctor>
    void createParticles( const ExecSpace& exec_space, UserFunctor user_create )
    {
        _init_timer.start();
        // Create a local mesh and owned space.
        auto owned_cells = local_grid->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        int particles_per_cell = 1;
        int num_particles = particles_per_cell * owned_cells.size();

        // Use default aosoa construction and resize.
        resize( num_particles, 0 );

        auto x = sliceReferencePosition();
        auto v = sliceVelocity();
        auto f = sliceForce();
        auto type = sliceType();
        auto rho = sliceDensity();
        auto u = sliceDisplacement();
        auto y = sliceCurrentPosition();
        auto vol = sliceVolume();
        auto nofail = sliceNoFail();

        // Initialize particles.
        auto create_functor =
            KOKKOS_LAMBDA( const int pid, const double px[dim], const double pv,
                           typename plist_x_type::particle_type& particle )
        {
            // Customize new particle.
            bool create = user_create( pid, px );
            if ( !create )
                return create;

            // Set the particle position.
            for ( int d = 0; d < 3; d++ )
            {
                Cabana::get( particle, CabanaPD::Field::ReferencePosition(),
                             d ) = px[d];
                u( pid, d ) = 0.0;
                y( pid, d ) = 0.0;
                v( pid, d ) = 0.0;
                f( pid, d ) = 0.0;
            }
            // Get the volume of the cell.
            vol( pid ) = pv;

            // FIXME: hardcoded.
            type( pid ) = 0;
            nofail( pid ) = 0;
            rho( pid ) = 1.0;

            return create;
        };
        n_local = Cabana::Grid::createParticles( Cabana::InitUniform{},
                                                 exec_space, create_functor,
                                                 _plist_x, 1, *local_grid );
        resize( n_local, 0 );
        size = _plist_x.size();

        // Not using Allreduce because global count is only used for printing.
        MPI_Reduce( &n_local, &n_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                    MPI_COMM_WORLD );
        _init_timer.stop();
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor )
    {
        _timer.start();
        Kokkos::RangePolicy<ExecSpace> policy( 0, n_local );
        Kokkos::parallel_for(
            "CabanaPD::Particles::update_particles", policy,
            KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
        _timer.stop();
    }

    auto sliceReferencePosition()
    {
        return _plist_x.slice( CabanaPD::Field::ReferencePosition() );
    }
    auto sliceReferencePosition() const
    {
        return _plist_x.slice( CabanaPD::Field::ReferencePosition() );
    }
    auto sliceCurrentPosition()
    {
        // Update before returning data.
        updateCurrentPosition();
        return Cabana::slice<0>( _aosoa_y, "current_positions" );
    }
    auto sliceCurrentPosition() const
    {
        // Update before returning data.
        updateCurrentPosition();
        return Cabana::slice<0>( _aosoa_y, "current_positions" );
    }
    auto sliceDisplacement()
    {
        return Cabana::slice<0>( _aosoa_u, "displacements" );
    }
    auto sliceDisplacement() const
    {
        return Cabana::slice<0>( _aosoa_u, "displacements" );
    }
    auto sliceForce() { return _plist_f.slice( CabanaPD::Field::Force() ); }
    auto sliceForceAtomic()
    {
        auto f = sliceForce();
        using slice_type = decltype( f );
        using atomic_type = typename slice_type::atomic_access_slice;
        atomic_type f_a = f;
        return f_a;
    }
    auto sliceVolume() { return Cabana::slice<0>( _aosoa_vol, "volume" ); }
    auto sliceVolume() const
    {
        return Cabana::slice<0>( _aosoa_vol, "volume" );
    }
    auto sliceType() { return Cabana::slice<4>( _aosoa_other, "type" ); }
    auto sliceType() const { return Cabana::slice<4>( _aosoa_other, "type" ); }
    auto sliceStrainEnergy()
    {
        return Cabana::slice<1>( _aosoa_other, "strain_energy" );
    }
    auto sliceStrainEnergy() const
    {
        return Cabana::slice<1>( _aosoa_other, "strain_energy" );
    }
    auto sliceVelocity()
    {
        return Cabana::slice<0>( _aosoa_other, "velocities" );
    }
    auto sliceVelocity() const
    {
        return Cabana::slice<0>( _aosoa_other, "velocities" );
    }
    auto sliceDensity() { return Cabana::slice<2>( _aosoa_other, "density" ); }
    auto sliceDensity() const
    {
        return Cabana::slice<2>( _aosoa_other, "density" );
    }
    auto sliceDamage() { return Cabana::slice<3>( _aosoa_other, "damage" ); }
    auto sliceDamage() const
    {
        return Cabana::slice<3>( _aosoa_other, "damage" );
    }
    auto sliceNoFail()
    {
        return Cabana::slice<0>( _aosoa_nofail, "no_fail_region" );
    }
    auto sliceNoFail() const
    {
        return Cabana::slice<0>( _aosoa_nofail, "no_fail_region" );
    }

    auto getForce() { return _plist_f; }
    auto getReferencePosition() { return _plist_x; }

    void updateCurrentPosition()
    {
        _timer.start();
        // Not using slice function because this is called inside.
        auto y = Cabana::slice<0>( _aosoa_y, "current_positions" );
        auto x = sliceReferencePosition();
        auto u = sliceDisplacement();
        Kokkos::RangePolicy<execution_space> policy( 0, n_local + n_ghost );
        auto sum_x_u = KOKKOS_LAMBDA( const std::size_t pid )
        {
            for ( int d = 0; d < 3; d++ )
                y( pid, d ) = x( pid, d ) + u( pid, d );
        };
        Kokkos::parallel_for( "CabanaPD::CalculateCurrentPositions", policy,
                              sum_x_u );
        _timer.stop();
    }

    void resize( int new_local, int new_ghost )
    {
        _timer.start();
        n_local = new_local;
        n_ghost = new_ghost;

        _plist_x.aosoa().resize( new_local + new_ghost );
        _aosoa_u.resize( new_local + new_ghost );
        _aosoa_y.resize( new_local + new_ghost );
        _aosoa_vol.resize( new_local + new_ghost );
        _plist_f.aosoa().resize( new_local );
        _aosoa_other.resize( new_local );
        _aosoa_nofail.resize( new_local + new_ghost );
        size = _plist_x.size();
        _timer.stop();
    };

    auto getPosition( const bool use_reference )
    {
        if ( use_reference )
            return sliceReferencePosition();
        else
            return sliceCurrentPosition();
    }

    void output( [[maybe_unused]] const int output_step,
                 [[maybe_unused]] const double output_time,
                 [[maybe_unused]] const bool use_reference = true )
    {
        _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", MPI_COMM_WORLD, output_step, output_time,
            n_local, getPosition( use_reference ), sliceStrainEnergy(),
            sliceForce(), sliceDisplacement(), sliceVelocity(), sliceDamage() );
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::
            writePartialRangeTimeStep(
                "particles", local_grid->globalGrid(), output_step, output_time,
                0, n_local, getPosition( use_reference ), sliceStrainEnergy(),
                sliceForce(), sliceDisplacement(), sliceVelocity(),
                sliceDamage() );
#else
        log( std::cout, "No particle output enabled." );
#endif
#endif

        _output_timer.stop();
    }

    auto timeInit() { return _init_timer.time(); };
    auto timeOutput() { return _output_timer.time(); };
    auto time() { return _timer.time(); };

    friend class Comm<self_type, PMB, TemperatureIndependent>;
    friend class Comm<self_type, PMB, TemperatureDependent>;

  protected:
    aosoa_u_type _aosoa_u;
    aosoa_y_type _aosoa_y;
    aosoa_vol_type _aosoa_vol;
    aosoa_nofail_type _aosoa_nofail;
    aosoa_other_type _aosoa_other;

    plist_x_type _plist_x;
    plist_f_type _plist_f;

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif

    Timer _init_timer;
    Timer _output_timer;
    Timer _timer;
};

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, LPS, TemperatureIndependent, Dimension>
    : public Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>
{
  public:
    using self_type =
        Particles<MemorySpace, LPS, TemperatureIndependent, Dimension>;
    using base_type =
        Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>;
    using thermal_type = TemperatureIndependent;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

    // Per particle.
    using base_type::n_ghost;
    using base_type::n_global;
    using base_type::n_local;
    using base_type::size;

    // These are split since weighted volume only needs to be communicated once
    // and dilatation only needs to be communicated for LPS.
    using scalar_type = typename base_type::scalar_type;
    using aosoa_theta_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    using aosoa_m_type = Cabana::AoSoA<scalar_type, memory_space, 1>;

    // Per type.
    using base_type::n_types;

    // Simulation total domain.
    using base_type::global_mesh_ext;

    // Simulation sub domain (single MPI rank).
    using base_type::ghost_mesh_hi;
    using base_type::ghost_mesh_lo;
    using base_type::local_mesh_ext;
    using base_type::local_mesh_hi;
    using base_type::local_mesh_lo;

    using base_type::dx;
    using base_type::local_grid;

    using base_type::halo_width;

    // Default constructor.
    Particles()
        : base_type()
    {
        _init_timer.start();
        _aosoa_m = aosoa_m_type( "Particle Weighted Volumes", 0 );
        _aosoa_theta = aosoa_theta_type( "Particle Dilatations", 0 );
        _init_timer.stop();
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width )
        : base_type( exec_space, low_corner, high_corner, num_cells,
                     max_halo_width )
    {
        _init_timer.start();
        _aosoa_m = aosoa_m_type( "Particle Weighted Volumes", n_local );
        _aosoa_theta = aosoa_theta_type( "Particle Dilatations", n_local );
        init_lps();
        _init_timer.stop();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _init_timer.start();
        _aosoa_m.resize( n_local );
        _aosoa_theta.resize( n_local );
        _init_timer.stop();
    }

    auto sliceDilatation()
    {
        return Cabana::slice<0>( _aosoa_theta, "dilatation" );
    }
    auto sliceDilatation() const
    {
        return Cabana::slice<0>( _aosoa_theta, "dilatation" );
    }
    auto sliceWeightedVolume()
    {
        return Cabana::slice<0>( _aosoa_m, "weighted_volume" );
    }
    auto sliceWeightedVolume() const
    {
        return Cabana::slice<0>( _aosoa_m, "weighted_volume" );
    }

    void resize( int new_local, int new_ghost )
    {
        base_type::resize( new_local, new_ghost );
        _timer.start();
        _aosoa_theta.resize( new_local + new_ghost );
        _aosoa_m.resize( new_local + new_ghost );
        _timer.stop();
    }

    void output( [[maybe_unused]] const int output_step,
                 [[maybe_unused]] const double output_time,
                 [[maybe_unused]] const bool use_reference = true )
    {
        _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", MPI_COMM_WORLD, output_step, output_time,
            n_local, base_type::getPosition( use_reference ),
            base_type::sliceStrainEnergy(), base_type::sliceForce(),
            base_type::sliceDisplacement(), base_type::sliceVelocity(),
            base_type::sliceDamage(), sliceWeightedVolume(),
            sliceDilatation() );
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::
            writePartialRangeTimeStep(
                "particles", local_grid->globalGrid(), output_step, output_time,
                0, n_local, base_type::getPosition( use_reference ),
                base_type::sliceStrainEnergy(), base_type::sliceForce(),
                base_type::sliceDisplacement(), base_type::sliceVelocity(),
                base_type::sliceDamage(), sliceWeightedVolume(),
                sliceDilatation() );
#else
        log( std::cout, "No particle output enabled." );
#endif
#endif

        _output_timer.stop();
    }

    friend class Comm<self_type, PMB, TemperatureIndependent>;
    friend class Comm<self_type, LPS, TemperatureIndependent>;

  protected:
    void init_lps()
    {
        auto theta = sliceDilatation();
        Cabana::deep_copy( theta, 0.0 );
        auto m = sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );
    }

    aosoa_theta_type _aosoa_theta;
    aosoa_m_type _aosoa_m;

#ifdef Cabana_ENABLE_HDF5
    using base_type::h5_config;
#endif

    using base_type::_init_timer;
    using base_type::_output_timer;
    using base_type::_timer;
};

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, PMB, TemperatureDependent, Dimension>
    : public Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>
{
  public:
    using self_type =
        Particles<MemorySpace, PMB, TemperatureDependent, Dimension>;
    using base_type =
        Particles<MemorySpace, PMB, TemperatureIndependent, Dimension>;
    using thermal_type = TemperatureDependent;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

    // Per particle.
    using base_type::n_ghost;
    using base_type::n_global;
    using base_type::n_local;
    using base_type::size;

    // These are split since weighted volume only needs to be communicated once
    // and dilatation only needs to be communicated for LPS.
    using temp_types = Cabana::MemberTypes<double, double>;
    using aosoa_temp_type = Cabana::AoSoA<temp_types, memory_space, 1>;

    // Per type.
    using base_type::n_types;

    // Simulation total domain.
    using base_type::global_mesh_ext;

    // Simulation sub domain (single MPI rank).
    using base_type::ghost_mesh_hi;
    using base_type::ghost_mesh_lo;
    using base_type::local_mesh_ext;
    using base_type::local_mesh_hi;
    using base_type::local_mesh_lo;

    using base_type::dx;
    using base_type::local_grid;

    using base_type::halo_width;

    // Default constructor.
    Particles()
        : base_type()
    {
        _aosoa_temp = aosoa_temp_type( "Particle Temperature", 0 );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width )
        : base_type( exec_space, low_corner, high_corner, num_cells,
                     max_halo_width )
    {
        _aosoa_temp = aosoa_temp_type( "Particle Temperature", n_local );
        init_temp();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _aosoa_temp.resize( n_local );
    }

    auto sliceTemperature()
    {
        return Cabana::slice<0>( _aosoa_temp, "temperature" );
    }
    auto sliceTemperature() const
    {
        return Cabana::slice<0>( _aosoa_temp, "temperature" );
    }
    auto sliceTemperatureConduction()
    {
        return Cabana::slice<1>( _aosoa_temp, "temperature_conduction" );
    }
    auto sliceTemperatureConduction() const
    {
        return Cabana::slice<1>( _aosoa_temp, "temperature_conduction" );
    }
    auto sliceTemperatureConductionAtomic()
    {
        auto temp = sliceTemperature();
        using slice_type = decltype( temp );
        using atomic_type = typename slice_type::atomic_access_slice;
        atomic_type temp_a = temp;
        return temp_a;
    }

    void resize( int new_local, int new_ghost )
    {
        base_type::resize( new_local, new_ghost );
        _aosoa_temp.resize( new_local + new_ghost );
    }

    void output( [[maybe_unused]] const int output_step,
                 [[maybe_unused]] const double output_time,
                 [[maybe_unused]] const bool use_reference = true )
    {
#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", MPI_COMM_WORLD, output_step, output_time,
            n_local, base_type::getPosition( use_reference ),
            base_type::sliceStrainEnergy(), base_type::sliceForce(),
            base_type::sliceDisplacement(), base_type::sliceVelocity(),
            base_type::sliceDamage(), sliceTemperature() );
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::
            writePartialRangeTimeStep(
                "particles", local_grid->globalGrid(), output_step, output_time,
                0, n_local, base_type::getPosition( use_reference ),
                base_type::sliceStrainEnergy(), base_type::sliceForce(),
                base_type::sliceDisplacement(), base_type::sliceVelocity(),
                base_type::sliceDamage(), sliceTemperature() );
#else
        log( std::cout, "No particle output enabled." );
#endif
#endif
    }

    friend class Comm<self_type, PMB, TemperatureIndependent>;
    friend class Comm<self_type, LPS, TemperatureIndependent>;
    friend class Comm<self_type, PMB, TemperatureDependent>;
    friend class Comm<self_type, LPS, TemperatureDependent>;

  protected:
    void init_temp()
    {
        auto temp = sliceTemperature();
        Cabana::deep_copy( temp, 0.0 );
    }

    aosoa_temp_type _aosoa_temp;

#ifdef Cabana_ENABLE_HDF5
    using base_type::h5_config;
#endif
};

template <typename MemorySpace, typename ModelType, typename ExecSpace>
auto createParticles( ExecSpace exec_space, CabanaPD::Inputs inputs )
{
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    double delta = inputs["horizon"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    return std::make_shared<
        CabanaPD::Particles<MemorySpace, typename ModelType::base_model,
                            typename ModelType::thermal_type>>(
        exec_space, low_corner, high_corner, num_cells, halo_width );
}

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          std::size_t Dim>
auto createParticles( const ExecSpace& exec_space,
                      std::array<double, Dim> low_corner,
                      std::array<double, Dim> high_corner,
                      const std::array<int, Dim> num_cells,
                      const int max_halo_width )
{
    return std::make_shared<
        CabanaPD::Particles<MemorySpace, typename ModelType::base_model,
                            typename ModelType::thermal_type>>(
        exec_space, low_corner, high_corner, num_cells, max_halo_width );
}

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, std::size_t Dim>
auto createParticles(
    const ExecSpace& exec_space, std::array<double, Dim> low_corner,
    std::array<double, Dim> high_corner, const std::array<int, Dim> num_cells,
    const int max_halo_width,
    typename std::enable_if<( is_temperature_dependent<ThermalType>::value ),
                            int>::type* = 0 )
{
    return std::make_shared<CabanaPD::Particles<
        MemorySpace, ModelType, typename ThermalType::base_type>>(
        exec_space, low_corner, high_corner, num_cells, max_halo_width );
}

} // namespace CabanaPD

#endif
