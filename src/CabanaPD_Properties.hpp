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

#ifndef PROPERTIES_HPP
#define PROPERTIES_HPP

#include <Cabana_Core.hpp>

namespace CabanaPD
{
//---------------------------------------------------------------------------//
// Properties.
//---------------------------------------------------------------------------//
namespace Field
{

template <typename MemorySpace>
class Property;

template <>
class Property<SingleSpecies>
{
    using species_type = SingleSpecies;
    double value;

    Property( const double _value )
        : value( _value ){};
};

template <typename MemorySpace>
struct Property
{
    using species_type = MultiSpecies;

    using memory_space = MemorySpace;
    using view_type_1d = Kokkos::View<double*, memory_space>;
    view_type_1d delta;
    double max_delta;
    std::size_t num_types;

    using view_type_2d = Kokkos::View<double**, memory_space>;

    using view_type_map = Kokkos::View<int[2]*, memory_space>;
    view_type_map map;

    Property( const double _delta )
        : delta( view_type_1d( "delta", 1 ) )
        , num_types( 1 )
    {
        Kokkos::deep_copy( delta, _delta );
    }

    template <typename ArrayType>
    Property( const int _num_types, const ArrayType& _delta )
        : delta( view_type_1d( "delta", _delta.size() ) )
        , num_types( _num_types )
        , map( view_type_map( "csr_to_2d_indexing", _delta.size() ) )
    {
        // 00 01 02 03    0 5 4    0 2
        // 10 11 12 13     1 3      1
        // 20 21 22 23       2
        // 30 31 32 33
        std::vector<int, num_types> vec_i;
        std::vector<int, num_types> vec_j;
        for ( std::size_t j = i; j < num_types; j++ )
            vec_i[i] = ;
        // i 0 1 2 3 2 1 0 0 0 1
        // j 0 1 2 3 3 3 3 2 1 2
        // 0 1 2 1 0 0
        // 0 1 0
        setParameters( _delta );
    }

    template <typename ArrayType>
    void setParameters( const ArrayType& _K )
    {
        // Copy into Views.
        auto K_copy = K;
        auto num_types_copy = num_types;
        auto init_self_func = KOKKOS_LAMBDA( const int i )
        {
            K_copy( map_i( i ), map_j( i ) ) = _K[i];
        };

        using exec_space = typename memory_space::execution_space;
        // This could be number of types (averaging for cross terms)
        // or number of types and cross terms
        Kokkos::RangePolicy<Kokkos::Serial> policy( 0, _K.size() );
        Kokkos::parallel_for( "CabanaPD::Model::Copy", policy, init_self_func );
        Kokkos::fence();

        // Initialize model parameters.
        if ( _K.size() == num_types )
            Kokkos::parallel_for( "CabanaPD::Model::Init", policy, *this );
    }
};

} // namespace Field
} // namespace CabanaPD

#endif
