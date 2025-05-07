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
          class OutputType = BaseOutput, int Dimension = 3>
class Particles;

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, PMB, TemperatureIndependent, BaseOutput, Dimension>
{
  public:
    using self_type = Particles<MemorySpace, PMB, TemperatureIndependent,
                                BaseOutput, Dimension>;
    using thermal_type = TemperatureIndependent;
    using output_type = BaseOutput;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    // Per particle.
    unsigned long long int num_global = 0;
    std::size_t frozen_offset = 0;
    std::size_t local_offset = 0;
    std::size_t num_ghost = 0;
    std::size_t _size = 0;

    // x, u, f (vector matching system dimension).
    using vector_type = Cabana::MemberTypes<double[dim]>;
    // volume, dilatation, weighted_volume.
    using scalar_type = Cabana::MemberTypes<double>;
    // no-fail.
    using int_type = Cabana::MemberTypes<int>;
    // v, rho, type.
    using other_types = Cabana::MemberTypes<double[dim], double, int>;

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
    // FIXME: this is for neighborlist construction.
    double ghost_mesh_lo[dim];
    double ghost_mesh_hi[dim];
    std::shared_ptr<
        Cabana::Grid::LocalGrid<Cabana::Grid::UniformMesh<double, dim>>>
        local_grid;
    Kokkos::Array<double, dim> dx;

    int halo_width;

    // Constructor which initializes particles on regular grid.
    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, TemperatureIndependent,
               std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               const ExecSpace exec_space, const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width,
                Cabana::InitUniform{}, exec_space, 0, create_frozen );
    }
    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               const ExecSpace exec_space, const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width,
                Cabana::InitUniform{}, exec_space, 0, create_frozen );
    }

    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, BaseOutput,
               std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               const ExecSpace exec_space, const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width,
                Cabana::InitUniform{}, exec_space, 0, create_frozen );
    }

    // Constructor which initializes particles on regular grid with
    // customization.
    template <class ModelType, class ExecSpace, class UserFunctor>
    Particles( MemorySpace, ModelType, TemperatureIndependent,
               std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               UserFunctor user_create, const ExecSpace exec_space,
               const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width,
                Cabana::InitUniform{}, user_create, exec_space, 0,
                create_frozen );
    }
    template <class ModelType, class ExecSpace, class UserFunctor>
    Particles( MemorySpace, ModelType, BaseOutput,
               std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               UserFunctor user_create, const ExecSpace exec_space,
               const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width,
                Cabana::InitUniform{}, user_create, exec_space, 0,
                create_frozen );
    }

    // Constructor which initializes particles on regular grid, randomly per
    // cell.
    template <class ModelType, class InitType, class ExecSpace>
    Particles( MemorySpace, ModelType, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               InitType init_type, const ExecSpace exec_space,
               const bool create_frozen = false )
        : halo_width( max_halo_width )
        , _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width, init_type,
                exec_space, 0, create_frozen );
    }

    // Constructor which initializes particles on regular grid with
    // customization, randomly per cell.
    template <class ModelType, class InitType, class UserFunctor,
              class ExecSpace>
    Particles( MemorySpace, ModelType, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               InitType init_type, UserFunctor user_create,
               const ExecSpace exec_space, const bool create_frozen = false )
        : halo_width( max_halo_width )
        , _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( low_corner, high_corner, num_cells, max_halo_width, init_type,
                user_create, exec_space, 0, create_frozen );
    }

    // Constructor with existing particle data.
    template <class ModelType, class ExecSpace, class PositionType,
              class VolumeType>
    Particles( MemorySpace, ModelType, TemperatureIndependent,
               const PositionType& x, const VolumeType& vol,
               std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               const ExecSpace exec_space, const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        createDomain( low_corner, high_corner, num_cells, max_halo_width );
        createParticles( exec_space, x, vol, 0, create_frozen );
    }
    template <class ModelType, class ExecSpace, class PositionType,
              class VolumeType>
    Particles( MemorySpace, ModelType, const PositionType& x,
               const VolumeType& vol, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width,
               const ExecSpace exec_space, const bool create_frozen = false )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        createDomain( low_corner, high_corner, num_cells, max_halo_width );
        createParticles( exec_space, x, vol, 0, create_frozen );
    }

    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, TemperatureIndependent, BaseOutput,
               CabanaPD::Inputs inputs, ExecSpace exec_space )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( inputs, exec_space );
    }
    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, BaseOutput, CabanaPD::Inputs inputs,
               ExecSpace exec_space )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( inputs, exec_space );
    }
    template <class ModelType, class ExecSpace>
    Particles( MemorySpace, ModelType, CabanaPD::Inputs inputs,
               ExecSpace exec_space )
        : _plist_x( "positions" )
        , _plist_f( "forces" )
    {
        create( inputs, exec_space );
    }

    void createDomain( std::array<double, dim> low_corner,
                       std::array<double, dim> high_corner,
                       const std::array<int, dim> num_cells,
                       const int max_halo_width )
    {
        _init_timer.start();
        halo_width = max_halo_width;

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

    KOKKOS_INLINE_FUNCTION bool operator()( const int, const double[dim] ) const
    {
        return true;
    }

    template <class ExecSpace, class InitType>
    void
    createParticles( const ExecSpace& exec_space, InitType init_type,
                     const std::size_t num_previous,
                     const bool create_frozen = false,
                     typename std::enable_if<
                         ( std::is_same<InitType, Cabana::InitUniform>::value ||
                           std::is_same<InitType, Cabana::InitRandom>::value ),
                         int>::type* = 0 )
    {
        createParticles( exec_space, init_type, *this, num_previous,
                         create_frozen );
    }

    template <class ExecSpace, class InitType, class UserFunctor>
    void
    createParticles( const ExecSpace& exec_space, InitType init_type,
                     UserFunctor user_create, const std::size_t num_previous,
                     const bool create_frozen = false,
                     typename std::enable_if<
                         ( std::is_same<InitType, Cabana::InitUniform>::value ||
                           std::is_same<InitType, Cabana::InitRandom>::value ),
                         int>::type* = 0 )
    {
        _init_timer.start();
        // Create a local mesh and owned space.
        auto owned_cells = local_grid->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

        int particles_per_cell = 1;
        int num_particles = particles_per_cell * owned_cells.size();

        // Use default aosoa construction and resize.
        assert( num_previous <= referenceOffset() );
        resize( num_particles + num_previous, 0, create_frozen );

        auto x = sliceReferencePosition();
        auto v = sliceVelocity();
        auto f = sliceForce();
        auto type = sliceType();
        auto rho = sliceDensity();
        auto u = sliceDisplacement();
        auto vol = sliceVolume();
        auto nofail = sliceNoFail();

        // Initialize particles.
        auto create_functor =
            KOKKOS_LAMBDA( const int pid, const double px[dim], const double pv,
                           typename plist_x_type::particle_type& particle )
        {
            // Customize new particle.
            // NOTE: we fill information for all particles because only the
            // positions are correctly selectively created. This will only work
            // when setting all values uniformly, as is currently the case!
            bool create = user_create( pid, px );

            // Set the particle position.
            for ( int d = 0; d < 3; d++ )
            {
                Cabana::get( particle, CabanaPD::Field::ReferencePosition(),
                             d ) = px[d];
                u( pid, d ) = 0.0;
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
        // Fence inside create.
        local_offset = Cabana::Grid::createParticles(
            init_type, exec_space, create_functor, _plist_x, particles_per_cell,
            *local_grid, num_previous, false );
        resize( local_offset, 0, create_frozen );

        updateGlobal();
        _init_timer.stop();
    }

    // Store custom created particle positions and volumes.
    template <class ExecSpace, class PositionType, class VolumeType>
    void createParticles(
        const ExecSpace, const PositionType& x, const VolumeType& vol,
        const std::size_t num_previous, const bool create_frozen = false,
        typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                                  Kokkos::is_view<PositionType>::value ) &&
                                    ( Cabana::is_slice<VolumeType>::value ||
                                      Kokkos::is_view<VolumeType>::value ),
                                int>::type* = 0 )
    {
        // Ensure valid previous particles.
        assert( num_previous <= referenceOffset() );
        // Ensure matching input sizes.
        assert( vol.size() == x.extent( 0 ) );
        resize( vol.size() + num_previous, 0, create_frozen );

        auto p_x = sliceReferencePosition();
        auto p_vol = sliceVolume();
        auto v = sliceVelocity();
        auto f = sliceForce();
        auto type = sliceType();
        auto rho = sliceDensity();
        auto u = sliceDisplacement();
        auto nofail = sliceNoFail();

        static_assert(
            Cabana::is_accessible_from<
                memory_space, typename PositionType::execution_space>::value );

        Kokkos::parallel_for(
            "copy_to_particles",
            Kokkos::RangePolicy<ExecSpace>( num_previous, localOffset() ),
            KOKKOS_LAMBDA( const int pid ) {
                auto pid_offset = pid - num_previous;
                // Set the particle position and volume.
                // Set everything else to zero.
                p_vol( pid ) = vol( pid_offset );
                for ( int d = 0; d < 3; d++ )
                {
                    p_x( pid, d ) = x( pid_offset, d );
                    u( pid, d ) = 0.0;
                    v( pid, d ) = 0.0;
                    f( pid, d ) = 0.0;
                }
                type( pid ) = 0;
                nofail( pid ) = 0;
                rho( pid ) = 1.0;
            } );
        Kokkos::fence();

        updateGlobal();
    }

    template <class ExecSpace, class InitType>
    void
    create( std::array<double, dim> low_corner,
            std::array<double, dim> high_corner,
            const std::array<int, dim> num_cells, const int max_halo_width,
            InitType init_type, const ExecSpace exec_space,
            const std::size_t num_previous, const bool create_frozen = false,
            typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                    int>::type* = 0 )
    {
        createDomain( low_corner, high_corner, num_cells, max_halo_width );
        createParticles( exec_space, init_type, *this, num_previous,
                         create_frozen );
    }

    template <class ExecSpace, class InitType, class UserFunctor>
    void create(
        std::array<double, dim> low_corner, std::array<double, dim> high_corner,
        const std::array<int, dim> num_cells, const int max_halo_width,
        InitType init_type, UserFunctor user_create, const ExecSpace exec_space,
        const std::size_t num_previous, const bool create_frozen = false,
        typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                int>::type* = 0 )
    {
        createDomain( low_corner, high_corner, num_cells, max_halo_width );
        createParticles( exec_space, init_type, user_create, num_previous,
                         create_frozen );
    }

    template <class ExecSpace>
    void create( CabanaPD::Inputs inputs, ExecSpace exec_space )
    {
        std::array<double, 3> low_corner = inputs["low_corner"];
        std::array<double, 3> high_corner = inputs["high_corner"];
        std::array<int, 3> num_cells = inputs["num_cells"];
        double delta = inputs["horizon"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        halo_width = m + 1; // Just to be safe.

        createDomain( low_corner, high_corner, num_cells, halo_width );
        createParticles( exec_space, Cabana::InitUniform{}, *this, 0, false );
    }

    void updateGlobal()
    {
        // Not using Allreduce because global count is only used for printing.
        auto local_offset_mpi =
            static_cast<unsigned long long int>( local_offset );
        MPI_Reduce( &local_offset_mpi, &num_global, 1, MPI_UNSIGNED_LONG_LONG,
                    MPI_SUM, 0, MPI_COMM_WORLD );
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor,
                          const bool update_frozen = false )
    {
        _timer.start();
        std::size_t start = frozen_offset;
        if ( update_frozen )
            start = 0;
        Kokkos::RangePolicy<ExecSpace> policy( start, local_offset );
        Kokkos::parallel_for(
            "CabanaPD::Particles::update_particles", policy,
            KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
        Kokkos::fence();
        _timer.stop();
    }

    // Particles are always in order frozen, local, ghost.
    // Values for offsets are distinguished from separate (num) values.
    auto numFrozen() const { return frozen_offset; }
    auto frozenOffset() const { return frozen_offset; }
    auto numLocal() const { return local_offset - frozen_offset; }
    auto localOffset() const { return local_offset; }
    auto numGhost() const { return num_ghost; }
    // This is currently size because contact ghosts are not added yet.
    auto referenceOffset() const { return _size; }
    auto size() const { return _size; }
    auto numGlobal() const { return num_global; }

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
    auto sliceForce() const
    {
        return _plist_f.slice( CabanaPD::Field::Force() );
    }
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
    auto sliceVelocity()
    {
        return Cabana::slice<0>( _aosoa_other, "velocities" );
    }
    auto sliceVelocity() const
    {
        return Cabana::slice<0>( _aosoa_other, "velocities" );
    }
    auto sliceDensity() { return Cabana::slice<1>( _aosoa_other, "density" ); }
    auto sliceDensity() const
    {
        return Cabana::slice<1>( _aosoa_other, "density" );
    }
    auto sliceType() { return Cabana::slice<2>( _aosoa_other, "type" ); }
    auto sliceType() const { return Cabana::slice<2>( _aosoa_other, "type" ); }

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

    void updateCurrentPosition() const
    {
        //_timer.start();
        // Not using slice function because this is called inside.
        auto y = Cabana::slice<0>( _aosoa_y, "current_positions" );
        auto x = sliceReferencePosition();
        auto u = sliceDisplacement();
        // Frozen particles are included in output so we include them in this
        // loop to guarantee they are correct even though they never change.
        Kokkos::RangePolicy<execution_space> policy( 0, referenceOffset() );
        auto sum_x_u = KOKKOS_LAMBDA( const std::size_t pid )
        {
            for ( int d = 0; d < 3; d++ )
                y( pid, d ) = x( pid, d ) + u( pid, d );
        };
        Kokkos::parallel_for( "CabanaPD::CalculateCurrentPositions", policy,
                              sum_x_u );
        Kokkos::fence();
        //_timer.stop();
    }

    void resize( int new_local, int new_ghost,
                 const bool create_frozen = false )
    {
        _timer.start();
        if ( new_ghost > 0 )
            assert( create_frozen == false );

        local_offset = new_local;
        num_ghost = new_ghost;
        _size = new_local + new_ghost;

        _plist_x.aosoa().resize( referenceOffset() );
        _aosoa_u.resize( referenceOffset() );
        _aosoa_y.resize( referenceOffset() );
        _aosoa_vol.resize( referenceOffset() );
        _plist_f.aosoa().resize( localOffset() );
        _aosoa_other.resize( localOffset() );
        _aosoa_nofail.resize( referenceOffset() );

        if ( create_frozen )
            frozen_offset = _size;
        _timer.stop();
    };

    void shrink()
    {
        _timer.start();
        _plist_x.aosoa().shrinkToFit();
        _aosoa_u.shrinkToFit();
        _aosoa_y.shrinkToFit();
        _aosoa_vol.shrinkToFit();
        _plist_f.aosoa().shrinkToFit();
        _aosoa_other.shrinkToFit();
        _aosoa_nofail.shrinkToFit();
        _timer.stop();
    };

    template <typename KeepType>
    void remove( const int num_keep, const KeepType& keep )
    {
        Cabana::remove( execution_space(), num_keep, keep, _plist_x.aosoa(),
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _plist_f.aosoa(),
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _aosoa_u,
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _aosoa_vol,
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _aosoa_y,
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _aosoa_other,
                        numFrozen() );
        Cabana::remove( execution_space(), num_keep, keep, _aosoa_nofail,
                        numFrozen() );
        resize( frozen_offset + num_keep, 0 );
    }

    auto getPosition( const bool use_reference )
    {
        if ( use_reference )
            return sliceReferencePosition();
        else
            return sliceCurrentPosition();
    }

    // TODO: enable ignoring frozen particles.
    template <typename... OtherFields>
    void output( [[maybe_unused]] const int output_step,
                 [[maybe_unused]] const double output_time,
                 [[maybe_unused]] const bool use_reference,
                 [[maybe_unused]] OtherFields&&... other )
    {
        _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", MPI_COMM_WORLD, output_step, output_time,
            localOffset(), getPosition( use_reference ), sliceForce(),
            sliceDisplacement(), sliceVelocity(),
            std::forward<OtherFields>( other )... );
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::
            writePartialRangeTimeStep(
                "particles", local_grid->globalGrid(), output_step, output_time,
                0, localOffset(), getPosition( use_reference ), sliceForce(),
                sliceDisplacement(), sliceVelocity(),
                std::forward<OtherFields>( other )... );

#else
        log( std::cout, "No particle output enabled." );
#endif
#endif

        _output_timer.stop();
    }

    auto rank() { return local_grid->globalGrid().blockId(); }
    auto comm() { return local_grid->globalGrid().comm(); }

    auto timeInit() { return _init_timer.time(); };
    auto timeOutput() { return _output_timer.time(); };
    auto time() { return _timer.time(); };

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, Pair, TemperatureDependent>;

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
class Particles<MemorySpace, LPS, TemperatureIndependent, BaseOutput, Dimension>
    : public Particles<MemorySpace, PMB, TemperatureIndependent, BaseOutput,
                       Dimension>
{
  public:
    using self_type = Particles<MemorySpace, LPS, TemperatureIndependent,
                                BaseOutput, Dimension>;
    using base_type = Particles<MemorySpace, PMB, TemperatureIndependent,
                                BaseOutput, Dimension>;
    using output_type = typename base_type::output_type;
    using thermal_type = TemperatureIndependent;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

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

    // Constructor which initializes particles on regular grid.
    template <typename ModelType, typename... Args>
    Particles( MemorySpace space, ModelType, Args&&... args )
        : base_type( space, PMB{}, std::forward<Args>( args )... )
    {
        _init_timer.start();
        _aosoa_m = aosoa_m_type( "Particle Weighted Volumes",
                                 base_type::localOffset() );
        _aosoa_theta = aosoa_theta_type( "Particle Dilatations",
                                         base_type::localOffset() );
        init_lps();
        _init_timer.stop();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _init_timer.start();
        _aosoa_m.resize( base_type::localOffset() );
        _aosoa_theta.resize( base_type::localOffset() );
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
        _aosoa_theta.resize( base_type::referenceOffset() );
        _aosoa_m.resize( base_type::referenceOffset() );
        _timer.stop();
    }

    template <typename... OtherFields>
    void output( const int output_step, const double output_time,
                 const bool use_reference, OtherFields&&... other )
    {
        base_type::output( output_step, output_time, use_reference,
                           sliceWeightedVolume(), sliceDilatation(),
                           std::forward<OtherFields>( other )... );
    }

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, State, TemperatureIndependent>;

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

    using base_type::_init_timer;
    using base_type::_timer;
};

template <class MemorySpace, class ModelType, int Dimension>
class Particles<MemorySpace, ModelType, TemperatureDependent, BaseOutput,
                Dimension>
    : public Particles<MemorySpace, ModelType, TemperatureIndependent,
                       BaseOutput, Dimension>
{
  public:
    using self_type = Particles<MemorySpace, ModelType, TemperatureDependent,
                                BaseOutput, Dimension>;
    using base_type = Particles<MemorySpace, ModelType, TemperatureIndependent,
                                BaseOutput, Dimension>;
    using thermal_type = TemperatureDependent;
    using output_type = typename base_type::output_type;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

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

    template <typename... Args>
    Particles( MemorySpace space, ModelType model, TemperatureDependent,
               Args&&... args )
        : base_type( space, model, TemperatureIndependent{},
                     std::forward<Args>( args )... )
    {
        _aosoa_temp =
            aosoa_temp_type( "Particle Temperature", base_type::localOffset() );
        init_temp();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _aosoa_temp.resize( base_type::localOffset() );
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

    template <typename... Args>
    void resize( Args&&... args )
    {
        base_type::resize( std::forward<Args>( args )... );
        _aosoa_temp.resize( base_type::referenceOffset() );
    }

    template <typename... OtherFields>
    void output( const int output_step, const double output_time,
                 const bool use_reference, OtherFields&&... other )
    {
        base_type::output( output_step, output_time, use_reference,
                           sliceTemperature(),
                           std::forward<OtherFields>( other )... );
    }

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, State, TemperatureIndependent>;
    friend class Comm<self_type, Pair, TemperatureDependent>;
    friend class Comm<self_type, State, TemperatureDependent>;

  protected:
    void init_temp()
    {
        auto temp = sliceTemperature();
        Cabana::deep_copy( temp, 0.0 );
    }

    aosoa_temp_type _aosoa_temp;
};

template <class MemorySpace, class ThermalType, int Dimension>
class Particles<MemorySpace, Contact, ThermalType, BaseOutput, Dimension>
    : public Particles<MemorySpace, PMB, ThermalType, BaseOutput, Dimension>
{
    // Note: no overloaded output() since there are very few cases where this
    // is a desired output field.

  public:
    using self_type =
        Particles<MemorySpace, Contact, ThermalType, BaseOutput, Dimension>;
    using base_type =
        Particles<MemorySpace, PMB, ThermalType, BaseOutput, Dimension>;
    using thermal_type = typename base_type::thermal_type;
    using output_type = typename base_type::output_type;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

    using aosoa_u_neigh_type =
        Cabana::AoSoA<typename base_type::vector_type, memory_space, 1>;

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

    // Base constructor.
    template <typename... Args>
    Particles( Args&&... args )
        : base_type( std::forward<Args>( args )... )
    {
        _aosoa_u_neigh = aosoa_u_neigh_type( "Particle Contact Fields",
                                             base_type::localOffset() );
        init();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _aosoa_u_neigh.resize( base_type::localOffset() );
    }

    auto sliceDisplacementNeighborBuild()
    {
        return Cabana::slice<0>( _aosoa_u_neigh, "displacement_since_rebuild" );
    }
    auto sliceDisplacementNeighborBuild() const
    {
        return Cabana::slice<0>( _aosoa_u_neigh, "displacement_since_rebuild" );
    }

    template <typename... Args>
    void resize( Args&&... args )
    {
        base_type::resize( std::forward<Args>( args )... );
        _aosoa_u_neigh.resize( base_type::localOffset() );
    }

    void setMaxDisplacement( double new_max ) { _max_displacement = new_max; }
    double getMaxDisplacement() const { return _max_displacement; }

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, State, TemperatureIndependent>;
    friend class Comm<self_type, Pair, TemperatureDependent>;
    friend class Comm<self_type, State, TemperatureDependent>;

  protected:
    void init()
    {
        auto u_neigh = sliceDisplacementNeighborBuild();
        Cabana::deep_copy( u_neigh, 0.0 );
    }

    // Used for delaying neighbor construction.
    double _max_displacement;

    aosoa_u_neigh_type _aosoa_u_neigh;
};

template <class MemorySpace, class ModelType, class ThermalType, int Dimension>
class Particles<MemorySpace, ModelType, ThermalType, EnergyOutput, Dimension>
    : public Particles<MemorySpace, ModelType, ThermalType, BaseOutput,
                       Dimension>
{
  public:
    using self_type =
        Particles<MemorySpace, ModelType, ThermalType, EnergyOutput, Dimension>;
    using base_type =
        Particles<MemorySpace, ModelType, ThermalType, BaseOutput, Dimension>;
    using thermal_type = typename base_type::thermal_type;
    using output_type = EnergyOutput;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

    // energy, damage
    using output_types = Cabana::MemberTypes<double, double>;
    using aosoa_output_type = Cabana::AoSoA<output_types, memory_space, 1>;

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

    // Constructor forwarding.
    template <typename... Args>
    Particles( MemorySpace space, ModelType model, ThermalType thermal,
               EnergyOutput, Args&&... args )
        : base_type( space, model, thermal, BaseOutput{},
                     std::forward<Args>( args )... )
    {
        _aosoa_output = aosoa_output_type( "Particle Output Fields",
                                           base_type::localOffset() );
        init();
    }

    template <typename... Args>
    Particles( MemorySpace space, ModelType model, EnergyOutput,
               Args&&... args )
        : base_type( space, model, std::forward<Args>( args )... )
    {
        _aosoa_output = aosoa_output_type( "Particle Output Fields",
                                           base_type::localOffset() );
        init();
    }

    template <typename... Args>
    Particles( Args&&... args )
        : base_type( std::forward<Args>( args )... )
    {
        _aosoa_output = aosoa_output_type( "Particle Output Fields",
                                           base_type::localOffset() );
        init();
    }

    template <typename... Args>
    void createParticles( Args&&... args )
    {
        // Forward arguments to standard or custom particle creation.
        base_type::createParticles( std::forward<Args>( args )... );
        _aosoa_output.resize( base_type::localOffset() );
    }

    auto sliceStrainEnergy()
    {
        return Cabana::slice<0>( _aosoa_output, "strain_energy" );
    }
    auto sliceStrainEnergy() const
    {
        return Cabana::slice<0>( _aosoa_output, "strain_energy" );
    }
    auto sliceDamage() { return Cabana::slice<1>( _aosoa_output, "damage" ); }
    auto sliceDamage() const
    {
        return Cabana::slice<1>( _aosoa_output, "damage" );
    }

    template <typename... Args>
    void resize( Args&&... args )
    {
        base_type::resize( std::forward<Args>( args )... );
        _aosoa_output.resize( base_type::localOffset() );
    }

    template <typename... OtherFields>
    void output( const int output_step, const double output_time,
                 const bool use_reference, OtherFields&&... other )
    {
        base_type::output( output_step, output_time, use_reference,
                           sliceStrainEnergy(), sliceDamage(),
                           std::forward<OtherFields>( other )... );
    }

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, State, TemperatureIndependent>;
    friend class Comm<self_type, Pair, TemperatureDependent>;
    friend class Comm<self_type, State, TemperatureDependent>;

  protected:
    void init()
    {
        auto energy = sliceStrainEnergy();
        Cabana::deep_copy( energy, 0.0 );
        auto phi = sliceDamage();
        Cabana::deep_copy( phi, 0.0 );
    }

    aosoa_output_type _aosoa_output;
};

template <class MemorySpace, class ModelType, class ThermalType, int Dimension>
class Particles<MemorySpace, ModelType, ThermalType, EnergyStressOutput,
                Dimension>
    : public Particles<MemorySpace, ModelType, ThermalType, EnergyOutput,
                       Dimension>
{
  public:
    using self_type = Particles<MemorySpace, ModelType, ThermalType,
                                EnergyStressOutput, Dimension>;
    using base_type =
        Particles<MemorySpace, ModelType, ThermalType, EnergyOutput, Dimension>;
    using thermal_type = typename base_type::thermal_type;
    using output_type = EnergyStressOutput;
    using memory_space = typename base_type::memory_space;
    using base_type::dim;

    using output_types = Cabana::MemberTypes<double[dim][dim]>;
    using aosoa_stress_type = Cabana::AoSoA<output_types, memory_space, 1>;

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

    // Constructor forwarding.
    template <typename... Args>
    Particles( MemorySpace space, ModelType model, ThermalType thermal,
               EnergyStressOutput, Args&&... args )
        : base_type( space, model, thermal, EnergyOutput{},
                     std::forward<Args>( args )... )
    {
        _aosoa_stress = aosoa_stress_type( "Particle Output Fields",
                                           base_type::localOffset() );
        init();
    }

    template <typename... Args>
    Particles( MemorySpace space, ModelType model, EnergyStressOutput,
               Args&&... args )
        : base_type( space, model, EnergyOutput{},
                     std::forward<Args>( args )... )
    {
        _aosoa_stress = aosoa_stress_type( "Particle Output Fields",
                                           base_type::localOffset() );
        init();
    }

    auto sliceStress() { return Cabana::slice<0>( _aosoa_stress, "stress" ); }
    auto sliceStress() const
    {
        return Cabana::slice<0>( _aosoa_stress, "stress" );
    }

    void resize( int new_local, int new_ghost )
    {
        base_type::resize( new_local, new_ghost );
        _aosoa_stress.resize( new_local + new_ghost );
    }

    template <typename... OtherFields>
    void output( const int output_step, const double output_time,
                 const bool use_reference, OtherFields&&... other )
    {
        base_type::output( output_step, output_time, use_reference,
                           sliceStress(),
                           std::forward<OtherFields>( other )... );
    }

    friend class Comm<self_type, Pair, TemperatureIndependent>;
    friend class Comm<self_type, State, TemperatureIndependent>;
    friend class Comm<self_type, Pair, TemperatureDependent>;
    friend class Comm<self_type, State, TemperatureDependent>;

  protected:
    void init()
    {
        auto stress = sliceStress();
        Cabana::deep_copy( stress, 0.0 );
    }

    aosoa_stress_type _aosoa_stress;
};

/******************************************************************************
  Template deduction guides.
******************************************************************************/
template <typename MemorySpace, typename ModelType, typename ExecSpace,
          typename OutputType>
Particles( MemorySpace, ModelType, CabanaPD::Inputs, OutputType, ExecSpace )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

// Backwards compatible versions with energy output by default.
template <typename MemorySpace, typename ModelType, typename ExecSpace>
Particles( MemorySpace, ModelType, CabanaPD::Inputs, ExecSpace )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          typename OutputType>
Particles( MemorySpace, ModelType, OutputType, Inputs, ExecSpace )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, std::size_t Dim, typename OutputType>
Particles( MemorySpace, ModelType, ThermalType, OutputType,
           std::array<double, Dim>, std::array<double, Dim>,
           const std::array<int, Dim>, int, ExecSpace, const bool = false,
           typename std::enable_if<(is_temperature<ThermalType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 typename ThermalType::base_type, OutputType>;

template <typename MemorySpace, typename ModelType, typename OutputType,
          typename ExecSpace, class UserFunctor, std::size_t Dim>
Particles( MemorySpace, ModelType, OutputType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           UserFunctor, const ExecSpace, const bool = false,
           typename std::enable_if<(is_output<OutputType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

template <typename MemorySpace, typename ModelType, typename OutputType,
          typename ExecSpace, class InitType, class UserFunctor,
          std::size_t Dim>
Particles( MemorySpace, ModelType, OutputType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           InitType, UserFunctor, const ExecSpace, const bool = false,
           typename std::enable_if<(is_output<OutputType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

template <typename MemorySpace, typename ModelType, typename OutputType,
          typename ExecSpace, std::size_t Dim>
Particles( MemorySpace, ModelType, OutputType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           const ExecSpace, const bool = false,
           typename std::enable_if<(is_output<OutputType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          std::size_t Dim>
Particles( MemorySpace, ModelType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           const ExecSpace, const bool = false,
           typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          class UserFunctor, std::size_t Dim>
Particles( MemorySpace, ModelType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           UserFunctor, const ExecSpace, const bool = false,
           typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          class InitType, class UserFunctor, std::size_t Dim>
Particles( MemorySpace, ModelType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           InitType, UserFunctor, const ExecSpace, const bool = false,
           typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, std::size_t Dim>
Particles( MemorySpace, ModelType, ThermalType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           const ExecSpace, const bool = false,
           typename std::enable_if<(is_temperature<ThermalType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 typename ThermalType::base_type, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, class UserFunctor, std::size_t Dim>
Particles( MemorySpace, ModelType, ThermalType, std::array<double, Dim>,
           std::array<double, Dim>, const std::array<int, Dim>, const int,
           UserFunctor, const ExecSpace, const bool = false,
           typename std::enable_if<(is_temperature<ThermalType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 typename ThermalType::base_type, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, typename PositionType, typename VolumeType,
          std::size_t Dim>
Particles( MemorySpace, ModelType, ThermalType, const PositionType&,
           const VolumeType&, std::array<double, Dim>, std::array<double, Dim>,
           const std::array<int, Dim>, const int, const ExecSpace,
           const bool = false,
           typename std::enable_if<(is_temperature<ThermalType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 typename ThermalType::base_type, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ThermalType,
          typename ExecSpace, typename PositionType, typename VolumeType,
          std::size_t Dim, typename OutputType>
Particles( MemorySpace, ModelType, ThermalType, OutputType, const PositionType&,
           const VolumeType&, std::array<double, Dim>, std::array<double, Dim>,
           const std::array<int, Dim>, const int, const ExecSpace,
           const bool = false,
           typename std::enable_if<(is_temperature<ThermalType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 typename ThermalType::base_type, OutputType>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          typename PositionType, typename VolumeType, std::size_t Dim>
Particles( MemorySpace, ModelType, const PositionType&, const VolumeType&,
           std::array<double, Dim>, std::array<double, Dim>,
           const std::array<int, Dim>, const int, const ExecSpace,
           const bool = false,
           typename std::enable_if<(Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, EnergyOutput>;

template <typename MemorySpace, typename ModelType, typename ExecSpace,
          typename PositionType, typename VolumeType, std::size_t Dim,
          typename OutputType>
Particles( MemorySpace, ModelType, OutputType, const PositionType&,
           const VolumeType&, std::array<double, Dim>, std::array<double, Dim>,
           const std::array<int, Dim>, const int, const ExecSpace,
           const bool = false,
           typename std::enable_if<(is_output<OutputType>::value &&
                                    Kokkos::is_execution_space_v<ExecSpace>),
                                   int>::type* = 0 )
    -> Particles<MemorySpace, typename ModelType::base_model,
                 TemperatureIndependent, OutputType>;

} // namespace CabanaPD

#endif
