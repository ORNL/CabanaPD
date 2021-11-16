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

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

namespace CabanaPD
{

template <class MemorySpace, int Dimension = 3>
class Particles
{
  public:
    using memory_space = MemorySpace;
    using device_type = typename memory_space::device_type;
    static constexpr int dim = Dimension;

    // Per particle
    int n_alloc;
    int n_global;
    int n_local;
    int n_ghost;
    // x, u, f, b, type, W, v, rho, id, volume
    using member_types =
        Cabana::MemberTypes<double[dim], double[dim], double[dim], double[dim],
                            int, double, double[dim], double, int, double>;
    // FIXME: add vector length
    // FIXME: enable separate aosoa
    using aosoa_type = Cabana::AoSoA<member_types, memory_space>;

    // Per type
    int n_types;

    // Simulation total domain
    std::array<double, 3> global_mesh_ext;

    // Simulation sub domain (single MPI rank)
    std::array<double, 3> local_mesh_ext;
    std::array<double, 3> local_mesh_lo;
    std::array<double, 3> local_mesh_hi;
    std::array<double, 3> ghost_mesh_lo;
    std::array<double, 3> ghost_mesh_hi;
    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> local_grid;

    std::vector<int> halo_neighbors;

    // Default constructor.
    Particles()
    {
        n_global = 0;
        n_local = 0;
        n_ghost = 0;
        n_types = 1;

        global_mesh_ext = { 0.0, 0.0, 0.0 };
        local_mesh_lo = { 0.0, 0.0, 0.0 };
        local_mesh_hi = { 0.0, 0.0, 0.0 };
        ghost_mesh_lo = { 0.0, 0.0, 0.0 };
        ghost_mesh_hi = { 0.0, 0.0, 0.0 };
        local_mesh_ext = { 0.0, 0.0, 0.0 };

        aosoa_type _aosoa( "Particles", n_local );
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace &exec_space, std::array<double, 3> low_corner,
               std::array<double, 3> high_corner,
               const std::array<int, 3> num_cells )
    {
        create_domain( low_corner, high_corner, num_cells );
        create_particles( exec_space );
    }

    ~Particles() {}

    void create_domain( std::array<double, 3> low_corner,
                        std::array<double, 3> high_corner,
                        const std::array<int, 3> num_cells )
    {
        // Create the MPI partitions.
        Cajita::UniformDimPartitioner partitioner;

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
        int halo_width = 1;
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

        halo_neighbors = Cajita::Impl::getTopology( *local_grid );
    }

    template <class ExecSpace>
    void create_particles( const ExecSpace &exec_space )
    {
        // Create a local mesh and owned space.
        auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );
        auto owned_cells = local_grid->indexSpace(
            Cajita::Own(), Cajita::Cell(), Cajita::Local() );

        int particles_per_cell = 1;
        int num_particles = particles_per_cell * owned_cells.size();
        _aosoa.resize( num_particles );
        auto x = slice_x();
        auto type = slice_type();
        auto rho = slice_rho();
        auto u = slice_u();
        auto vol = slice_vol();
        auto created = Kokkos::View<bool *, device_type>(
            Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
            num_particles );

        // Initialize particles.
        int local_num_create = 0;
        Kokkos::parallel_reduce(
            "init_particles_uniform",
            Cajita::createExecutionPolicy( owned_cells, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           int &create_count ) {
                // Compute the owned local cell id.
                int i_own = i - owned_cells.min( Cajita::Dim::I );
                int j_own = j - owned_cells.min( Cajita::Dim::J );
                int k_own = k - owned_cells.min( Cajita::Dim::K );
                int pid = i_own + owned_cells.extent( Cajita::Dim::I ) *
                                      ( j_own + k_own * owned_cells.extent(
                                                            Cajita::Dim::J ) );

                // Get the coordinates of the cell.
                int node[3] = { i, j, k };
                double cell_coord[3];
                local_mesh.coordinates( Cajita::Cell(), node, cell_coord );

                // Set the particle position.
                for ( int d = 0; d < 3; d++ )
                {
                    x( pid, d ) = cell_coord[d];
                    // FIXME - needs to be done in the unit test
                    u( pid, d ) = 2.0 * x( pid, d );
                }
                // FIXME: hardcoded
                type( pid ) = 0;
                rho( pid ) = 1.0;

                // Get the volume of the cell.
                int empty[3];
                vol( pid ) = local_mesh.measure( Cajita::Cell(), empty );

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
        // FIXME: global all reduce
    }

    // x, u, f, b, type, W, v, rho, id
    auto slice_x() { return Cabana::slice<0>( _aosoa ); }
    auto slice_u() { return Cabana::slice<1>( _aosoa ); }
    auto slice_f() { return Cabana::slice<2>( _aosoa ); }
    auto slice_f_a()
    {
        auto f = slice_f();
        using slice_type = decltype( f );
        using atomic_type = typename slice_type::atomic_access_slice;
        atomic_type f_a = f;
        return f_a;
    }
    auto slice_b() { return Cabana::slice<3>( _aosoa ); }
    auto slice_type() { return Cabana::slice<4>( _aosoa ); }
    auto slice_W() { return Cabana::slice<5>( _aosoa ); }
    auto slice_v() { return Cabana::slice<6>( _aosoa ); }
    auto slice_rho() { return Cabana::slice<7>( _aosoa ); }
    auto slice_id() { return Cabana::slice<8>( _aosoa ); }
    auto slice_vol() { return Cabana::slice<9>( _aosoa ); }

    void resize( int n_new )
    {
        if ( n_new > n_alloc )
        {
            n_alloc = n_new;
        }
        _aosoa.resize( n_new );
    };

    void gather( Cabana::Halo<device_type> halo )
    {
        Cabana::gather( halo, _aosoa );
    };

  protected:
    aosoa_type _aosoa;
};

} // namespace CabanaPD

#endif
