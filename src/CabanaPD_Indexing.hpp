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

#ifndef INDEXING_H
#define INDEXING_H

namespace CabanaPD
{
// Index along each diagonal sequentially (symmetric).
template <unsigned NumBaseModels>
struct DiagonalIndexing
{
    static_assert( NumBaseModels > 0, "NumBaseModels must be larger than 0" );

    KOKKOS_FUNCTION unsigned operator()( unsigned firstType,
                                         unsigned secondType ) const
    {
        if ( firstType >= NumBaseModels || secondType >= NumBaseModels )
            Kokkos::abort( "Index out of range of DiagonalIndexing" );

        unsigned diagonalOrder = Kokkos::abs( static_cast<int>( secondType ) -
                                              static_cast<int>( firstType ) );
        unsigned offset = 0;
        for ( unsigned order = 0; order < diagonalOrder; ++order )
        {
            offset += NumBaseModels - order;
        }

        unsigned indexAlongDiagonal =
            firstType < secondType ? firstType : secondType;

        return offset + indexAlongDiagonal;
    }
};

// Index same type as 0 and differing types as 1.
struct BinaryIndexing
{
    KOKKOS_FUNCTION unsigned operator()( unsigned firstType,
                                         unsigned secondType ) const
    {
        return firstType != secondType;
    }
};

template <unsigned NumBaseModels>
struct FullIndexing
{
    static_assert( NumBaseModels > 0, "NumBaseModels must be larger than 0" );

    KOKKOS_FUNCTION unsigned operator()( unsigned firstType,
                                         unsigned secondType ) const
    {
        if ( firstType >= NumBaseModels || secondType >= NumBaseModels )
            Kokkos::abort( "Index out of range of FullIndexing" );

        return firstType * NumBaseModels + secondType;
    }
};

} // namespace CabanaPD

#endif
