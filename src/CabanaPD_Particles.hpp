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

#ifndef PARTICLES_H
#define PARTICLES_H

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <CabanaPD_Comm.hpp>
#include <CabanaPD_Fields.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <class MemorySpace, class ModelType, int Dimension = 3>
class Particles;

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, PMB, Dimension>
{
  public:
    using self_type = Particles<MemorySpace, PMB, Dimension>;
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
    // type, W, v, rho, damage.
    using other_types =
        Cabana::MemberTypes<int, double, double[dim], double, double>;
    // Potentially needed later: body force (b), ID.

    // FIXME: add vector length.
    // FIXME: enable variable aosoa.
    using aosoa_u_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_y_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_vol_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    using aosoa_nofail_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_other_type = Cabana::AoSoA<other_types, memory_space>;
    using plist_x_type =
        Cabana::ParticleList<memory_space, 1,
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
    double dx[dim];

    int halo_width;

    // Default constructor.
    Particles()
    {
        for ( int d = 0; d < dim; d++ )
        {
            global_mesh_ext[d] = 0.0;
            local_mesh_lo[d] = 0.0;
            local_mesh_hi[d] = 0.0;
            ghost_mesh_lo[d] = 0.0;
            ghost_mesh_hi[d] = 0.0;
            local_mesh_ext[d] = 0.0;
        }
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

    void createDomain( std::array<double, dim> low_corner,
                       std::array<double, dim> high_corner,
                       const std::array<int, dim> num_cells )
    {
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
    }

    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
        // Create a local mesh and owned space.
        auto local_mesh =
            Cabana::Grid::createLocalMesh<memory_space>( *local_grid );
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

        auto created = Kokkos::View<bool*, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
            num_particles );

        // Initialize particles.
        int mpi_rank = -1;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        int local_num_create = 0;
        Kokkos::parallel_reduce(
            "CabanaPD::Particles::init_particles_uniform",
            Cabana::Grid::createExecutionPolicy( owned_cells, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           int& create_count ) {
                // Compute the owned local cell id.
                int i_own = i - owned_cells.min( Cabana::Grid::Dim::I );
                int j_own = j - owned_cells.min( Cabana::Grid::Dim::J );
                int k_own = k - owned_cells.min( Cabana::Grid::Dim::K );
                int pid =
                    i_own + owned_cells.extent( Cabana::Grid::Dim::I ) *
                                ( j_own + k_own * owned_cells.extent(
                                                      Cabana::Grid::Dim::J ) );

                // Get the coordinates of the cell.
                int node[3] = { i, j, k };
                double cell_coord[3];
                local_mesh.coordinates( Cabana::Grid::Cell(), node,
                                        cell_coord );

                // Set the particle position.
                for ( int d = 0; d < 3; d++ )
                {
                    x( pid, d ) = cell_coord[d];
                    u( pid, d ) = 0.0;
                    y( pid, d ) = 0.0;
                    v( pid, d ) = 0.0;
                    f( pid, d ) = 0.0;
                }
                // FIXME: hardcoded.
                type( pid ) = 0;
                nofail( pid ) = 0;
                rho( pid ) = 1.0;

                // Get the volume of the cell.
                int empty[3];
                vol( pid ) = local_mesh.measure( Cabana::Grid::Cell(), empty );

                // Customize new particle.
                // created( pid ) = create_functor( px, particle );
                created( pid ) = true;

                // If we created a new particle insert it into the
                // list.
                if ( created( pid ) )
                {
                    ++create_count;
                }
            },
            local_num_create );
        n_local = local_num_create;
        resize( n_local, 0 );
        size = _plist_x.size();

        // Not using Allreduce because global count is only used for printing.
        MPI_Reduce( &n_local, &n_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                    MPI_COMM_WORLD );
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor )
    {
        Kokkos::RangePolicy<ExecSpace> policy( 0, n_local );
        Kokkos::parallel_for(
            "CabanaPD::Particles::update_particles", policy,
            KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
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
    auto sliceType() { return Cabana::slice<0>( _aosoa_other, "type" ); }
    auto sliceType() const { return Cabana::slice<0>( _aosoa_other, "type" ); }
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
        return Cabana::slice<2>( _aosoa_other, "velocities" );
    }
    auto sliceVelocity() const
    {
        return Cabana::slice<2>( _aosoa_other, "velocities" );
    }
    auto sliceDensity() { return Cabana::slice<3>( _aosoa_other, "density" ); }
    auto sliceDensity() const
    {
        return Cabana::slice<3>( _aosoa_other, "density" );
    }
    auto sliceDamage() { return Cabana::slice<4>( _aosoa_other, "damage" ); }
    auto sliceDamage() const
    {
        return Cabana::slice<4>( _aosoa_other, "damage" );
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
    }

    void resize( int new_local, int new_ghost )
    {
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
    }

    friend class Comm<self_type, PMB>;

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
};

template <class MemorySpace, int Dimension>
class Particles<MemorySpace, LPS, Dimension>
    : public Particles<MemorySpace, PMB, Dimension>
{
  public:
    using self_type = Particles<MemorySpace, LPS, Dimension>;
    using base_type = Particles<MemorySpace, PMB, Dimension>;
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
        _aosoa_m = aosoa_m_type( "Particle Weighted Volumes", 0 );
        _aosoa_theta = aosoa_theta_type( "Particle Dilatations", 0 );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
               std::array<double, dim> high_corner,
               const std::array<int, dim> num_cells, const int max_halo_width )
        : base_type( exec_space, low_corner, high_corner, num_cells,
                     max_halo_width )
    {
        _aosoa_m = aosoa_m_type( "Particle Weighted Volumes", n_local );
        _aosoa_theta = aosoa_theta_type( "Particle Dilatations", n_local );
        init_lps();
    }

    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
        base_type::createParticles( exec_space );
        _aosoa_m.resize( 0 );
        _aosoa_theta.resize( 0 );
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
        _aosoa_theta.resize( new_local + new_ghost );
        _aosoa_m.resize( new_local + new_ghost );
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
    }

    friend class Comm<self_type, PMB>;
    friend class Comm<self_type, LPS>;

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
};

} // namespace CabanaPD

#endif
