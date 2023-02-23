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
#include <Cajita.hpp>

#include <CabanaPD_Comm.hpp>

namespace CabanaPD
{

// FIXME: this should use MemorySpace directly, but DeviceType enables the
// friend class with Comm (which only uses DeviceType because Cabana::Halo
// currently does)
template <class DeviceType, int Dimension = 3>
class Particles
{
  public:
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    static constexpr int dim = Dimension;

    // Per particle
    unsigned long long int n_global = 0;
    std::size_t n_local = 0;
    std::size_t n_ghost = 0;
    std::size_t size = 0;

    // x, u, f (vector matching system dimension)
    using vector_type = Cabana::MemberTypes<double[dim]>;
    // volume, dilatation, weighted_volume
    using scalar_type = Cabana::MemberTypes<double>;
    // type, W, v, rho, damage
    using other_types =
        Cabana::MemberTypes<int, double, double[dim], double, double>;
    // Potentially needed later: body force (b), ID

    // FIXME: add vector length
    // FIXME: enable variable aosoa
    using aosoa_x_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_u_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_f_type = Cabana::AoSoA<vector_type, memory_space, 1>;
    using aosoa_vol_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    // These are split since weighted volume only needs to be communicated once
    // and dilatation only needs to be communicated for LPS.
    using aosoa_theta_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    using aosoa_m_type = Cabana::AoSoA<scalar_type, memory_space, 1>;
    using aosoa_other_type = Cabana::AoSoA<other_types, memory_space>;

    // Per type
    int n_types = 1;

    // Simulation total domain
    std::array<double, 3> global_mesh_ext;

    // Simulation sub domain (single MPI rank)
    std::array<double, 3> local_mesh_ext;
    std::array<double, 3> local_mesh_lo;
    std::array<double, 3> local_mesh_hi;
    std::array<double, 3> ghost_mesh_lo;
    std::array<double, 3> ghost_mesh_hi;
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> local_grid;
    double dx;
    double dy;
    double dz;

    std::vector<int> halo_neighbors;
    int halo_width;

    // Default constructor.
    Particles()
    {
        global_mesh_ext = { 0.0, 0.0, 0.0 };
        local_mesh_lo = { 0.0, 0.0, 0.0 };
        local_mesh_hi = { 0.0, 0.0, 0.0 };
        ghost_mesh_lo = { 0.0, 0.0, 0.0 };
        ghost_mesh_hi = { 0.0, 0.0, 0.0 };
        local_mesh_ext = { 0.0, 0.0, 0.0 };

        aosoa_x_type _aosoa_x( "Particles", n_local );
        aosoa_u_type _aosoa_u( "Particles", n_local );
        aosoa_f_type _aosoa_f( "Particles", n_local );
        aosoa_vol_type _aosoa_vol( "Particles", n_local );
        aosoa_other_type _aosoa_other( "Particles", n_local );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::array<double, 3> low_corner,
               std::array<double, 3> high_corner,
               const std::array<int, 3> num_cells, const int max_halo_width,
               const bool build_cylinder = false )
        : halo_width( max_halo_width )
    {
        create_domain( low_corner, high_corner, num_cells );
        create_particles( exec_space, build_cylinder );
    }

    ~Particles() {}

    void create_domain( std::array<double, 3> low_corner,
                        std::array<double, 3> high_corner,
                        const std::array<int, 3> num_cells )
    {
        // Create the MPI partitions.
        Cajita::DimBlockPartitioner<3> partitioner;

        // Create global mesh of MPI partitions.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            low_corner, high_corner, num_cells );

        for ( int d = 0; d < 3; d++ )
            global_mesh_ext[d] = global_mesh->extent( d );

        // Create the global grid.
        std::array<bool, 3> is_periodic = { false, false, false };
        auto global_grid = Cajita::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, is_periodic, partitioner );

        // Create a local mesh
        local_grid = Cajita::createLocalGrid( global_grid, halo_width );
        auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

        for ( int d = 0; d < 3; d++ )
        {
            local_mesh_lo[d] = local_mesh.lowCorner( Cajita::Own(), d );
            local_mesh_hi[d] = local_mesh.highCorner( Cajita::Own(), d );
            ghost_mesh_lo[d] = local_mesh.lowCorner( Cajita::Ghost(), d );
            ghost_mesh_hi[d] = local_mesh.highCorner( Cajita::Ghost(), d );
            local_mesh_ext[d] = local_mesh.extent( Cajita::Own(), d );
        }

        // Uniform mesh spacing.
        int zero[3] = { 0, 0, 0 };
        dx = local_mesh.measure( Cajita::Edge<Cajita::Dim::I>(), zero );
        dy = local_mesh.measure( Cajita::Edge<Cajita::Dim::J>(), zero );
        dz = local_mesh.measure( Cajita::Edge<Cajita::Dim::K>(), zero );

        // FIXME: remove Impl
        halo_neighbors = Cajita::Impl::getTopology( *local_grid );
    }

    // FIXME: extra option here only meant for quick route to disk impact.
    template <class ExecSpace>
    void create_particles( const ExecSpace& exec_space,
                           const bool build_cylinder = false )
    {
        // Create a local mesh and owned space.
        auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );
        auto owned_cells = local_grid->indexSpace(
            Cajita::Own(), Cajita::Cell(), Cajita::Local() );

        int particles_per_cell = 1;
        int num_particles = particles_per_cell * owned_cells.size();
        _aosoa_x.resize( num_particles );
        _aosoa_u.resize( num_particles );
        _aosoa_f.resize( num_particles );
        _aosoa_vol.resize( num_particles );
        _aosoa_m.resize( num_particles );
        _aosoa_theta.resize( num_particles );
        _aosoa_other.resize( num_particles );
        size = _aosoa_x.size();
        auto x = slice_x();
        auto v = slice_v();
        auto type = slice_type();
        auto rho = slice_rho();
        auto u = slice_u();
        auto vol = slice_vol();

        auto f = slice_f();
        Cabana::deep_copy( f, 0.0 );
        auto theta = slice_theta();
        Cabana::deep_copy( theta, 0.0 );
        auto m = slice_m();
        Cabana::deep_copy( m, 0.0 );

        auto count = Kokkos::View<int*, memory_space>( "particle_count", 1 );

        // Initialize particles.
        int mpi_rank = -1;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::parallel_for(
            "CabanaPD::Particles::init_particles_uniform",
            Cajita::createExecutionPolicy( owned_cells, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get the coordinates of the cell.
                int node[3] = { i, j, k };
                double cell_coord[3];
                local_mesh.coordinates( Cajita::Cell(), node, cell_coord );

                // Customize new particle.
                // created( pid ) = create_functor( px, particle );
                bool create = true;
                if ( build_cylinder )
                {
                    auto width = global_mesh_ext[0] / 2.0;
                    auto r2 = cell_coord[0] * cell_coord[0] +
                              cell_coord[1] * cell_coord[1];
                    if ( r2 > width * width )
                        create = false;
                }

                if ( create )
                {
                    auto pid = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                    // Set the particle position.
                    for ( int d = 0; d < 3; d++ )
                    {
                        x( pid, d ) = cell_coord[d];
                        u( pid, d ) = 0.0;
                        v( pid, d ) = 0.0;
                    }
                    // FIXME: hardcoded
                    type( pid ) = 0;
                    rho( pid ) = 1.0;

                    // Get the volume of the cell.
                    int empty[3];
                    vol( pid ) = local_mesh.measure( Cajita::Cell(), empty );
                }
            } );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );
        n_local = count_host( 0 );
        _aosoa_x.resize( n_local );
        _aosoa_u.resize( n_local );
        _aosoa_f.resize( n_local );
        _aosoa_vol.resize( n_local );
        _aosoa_m.resize( n_local );
        _aosoa_theta.resize( n_local );
        _aosoa_other.resize( n_local );
        _aosoa_x.shrinkToFit();
        _aosoa_u.shrinkToFit();
        _aosoa_f.shrinkToFit();
        _aosoa_vol.shrinkToFit();
        _aosoa_m.shrinkToFit();
        _aosoa_theta.shrinkToFit();
        _aosoa_other.shrinkToFit();

        // Not using Allreduce because global count is only used for printing.
        MPI_Reduce( &n_local, &n_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                    MPI_COMM_WORLD );
    }

    template <class ExecSpace, class FunctorType>
    void update_particles( const ExecSpace, const FunctorType init_functor )
    {
        Kokkos::RangePolicy<ExecSpace> policy( 0, n_local );
        Kokkos::parallel_for(
            "CabanaPD::Particles::update_particles", policy,
            KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
    }

    auto slice_x() { return Cabana::slice<0>( _aosoa_x, "positions" ); }
    auto slice_x() const { return Cabana::slice<0>( _aosoa_x, "positions" ); }
    auto slice_u() { return Cabana::slice<0>( _aosoa_u, "displacements" ); }
    auto slice_u() const
    {
        return Cabana::slice<0>( _aosoa_u, "displacements" );
    }
    auto slice_f() { return Cabana::slice<0>( _aosoa_f, "forces" ); }
    auto slice_f_a()
    {
        auto f = slice_f();
        using slice_type = decltype( f );
        using atomic_type = typename slice_type::atomic_access_slice;
        atomic_type f_a = f;
        return f_a;
    }
    auto slice_vol() { return Cabana::slice<0>( _aosoa_vol, "volume" ); }
    auto slice_vol() const { return Cabana::slice<0>( _aosoa_vol, "volume" ); }
    auto slice_type() { return Cabana::slice<0>( _aosoa_other, "type" ); }
    auto slice_W() { return Cabana::slice<1>( _aosoa_other, "strain_energy" ); }
    auto slice_v() { return Cabana::slice<2>( _aosoa_other, "velocities" ); }
    auto slice_rho() { return Cabana::slice<3>( _aosoa_other, "density" ); }
    auto slice_phi() { return Cabana::slice<4>( _aosoa_other, "damage" ); }
    auto slice_theta()
    {
        return Cabana::slice<0>( _aosoa_theta, "dilatation" );
    }
    auto slice_theta() const
    {
        return Cabana::slice<0>( _aosoa_theta, "dilatation" );
    }
    auto slice_m() { return Cabana::slice<0>( _aosoa_m, "weighted_volume" ); }
    auto slice_m() const
    {
        return Cabana::slice<0>( _aosoa_m, "weighted_volume" );
    }

    void resize( int new_local, int new_ghost )
    {
        n_local = new_local;
        n_ghost = new_ghost;

        _aosoa_x.resize( new_local + new_ghost );
        _aosoa_u.resize( new_local + new_ghost );
        _aosoa_vol.resize( new_local + new_ghost );
        _aosoa_theta.resize( new_local + new_ghost );
        _aosoa_m.resize( new_local + new_ghost );
        _aosoa_f.resize( new_local );
        _aosoa_other.resize( new_local );
        size = _aosoa_x.size();
    };

    friend class Comm<Particles<device_type>>;

  protected:
    aosoa_x_type _aosoa_x;
    aosoa_u_type _aosoa_u;
    aosoa_f_type _aosoa_f;
    aosoa_vol_type _aosoa_vol;
    aosoa_theta_type _aosoa_theta;
    aosoa_m_type _aosoa_m;
    aosoa_other_type _aosoa_other;
};

} // namespace CabanaPD

#endif
