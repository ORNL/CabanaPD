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

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Indexing.hpp>
#include <CabanaPD_config.hpp>

#include <type_traits>

namespace Test
{

TEST( TEST_CATEGORY, test_diagonalIndexing )
{
    // since DiagonalIndexing is a generative indexing following the diagonals
    // of a matrix of combinations of types,
    // it is tested with an induction proof. for the first element (N=1) this is
    // trivial. If it holds for N=2 and N=3 then the indexing should also be
    // valid for higher values of N.

    // test diagonal indexing for N=1
    CabanaPD::DiagonalIndexing<1> indexing1;

    ASSERT_EQ( indexing1( 0, 0 ), 0 );

    // test diagonal indexing for N=2
    CabanaPD::DiagonalIndexing<2> indexing2;

    // main diagonal
    ASSERT_EQ( indexing2( 0, 0 ), 0 );
    ASSERT_EQ( indexing2( 1, 1 ), 1 );

    // first minor diagonal (incl symmetric terms)
    ASSERT_EQ( indexing2( 0, 1 ), 2 );
    ASSERT_EQ( indexing2( 1, 0 ), 2 );

    // test diagonal indexing for N=3
    CabanaPD::DiagonalIndexing<3> indexing3;

    // main diagonal
    ASSERT_EQ( indexing3( 0, 0 ), 0 );
    ASSERT_EQ( indexing3( 1, 1 ), 1 );
    ASSERT_EQ( indexing3( 2, 2 ), 2 );

    // first minor diagonal (incl symmetric terms)
    ASSERT_EQ( indexing3( 0, 1 ), 3 );
    ASSERT_EQ( indexing3( 1, 2 ), 4 );
    ASSERT_EQ( indexing3( 1, 0 ), 3 );
    ASSERT_EQ( indexing3( 2, 1 ), 4 );

    // second minor diagonal (incl symmetric terms)
    ASSERT_EQ( indexing3( 0, 2 ), 5 );
    ASSERT_EQ( indexing3( 2, 0 ), 5 );
}

TEST( TEST_CATEGORY, test_diagonalIndexing_death )
{
    // assert death for out of scope indexing
    ASSERT_DEATH(
        {
            CabanaPD::DiagonalIndexing<2> indexing;
            auto i = indexing( 2, 0 );
            (void)i;
        },
        "Index out of range of DiagonalIndexing" );
    ASSERT_DEATH(
        {
            CabanaPD::DiagonalIndexing<2> indexing;
            auto i = indexing( 0, 2 );
            (void)i;
        },
        "Index out of range of DiagonalIndexing" );
}

TEST( TEST_CATEGORY, test_binaryIndexing )
{
    // test binary indexing for N=1
    CabanaPD::BinaryIndexing<1> indexing1;

    ASSERT_EQ( indexing1( 0, 0 ), 0 );

    // test binary indexing for N=2
    CabanaPD::BinaryIndexing<2> indexing2;

    // main binary
    ASSERT_EQ( indexing2( 0, 0 ), 0 );
    ASSERT_EQ( indexing2( 1, 1 ), 0 );

    // first minor binary (incl symmetric terms)
    ASSERT_EQ( indexing2( 0, 1 ), 1 );
    ASSERT_EQ( indexing2( 1, 0 ), 1 );

    // test binary indexing for N=3
    CabanaPD::BinaryIndexing<3> indexing3;

    for ( unsigned i = 0; 3 < i; ++i )
    {
        for ( unsigned j = 0; 3 < j; ++j )
        {
            if ( i == j )
                ASSERT_EQ( indexing3( i, j ), 0 );
            else
                ASSERT_EQ( indexing3( i, j ), 1 );
        }
    }
}

TEST( TEST_CATEGORY, test_binaryIndexing_death )
{
    // assert death for out of scope indexing
    ASSERT_DEATH(
        {
            CabanaPD::BinaryIndexing<2> indexing;
            auto i = indexing( 2, 0 );
            (void)i;
        },
        "" );
    ASSERT_DEATH(
        {
            CabanaPD::BinaryIndexing<2> indexing;
            auto i = indexing( 0, 2 );
            (void)i;
        },
        "" );
}

} // end namespace Test
