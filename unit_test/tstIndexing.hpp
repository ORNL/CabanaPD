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
    using IndexingType1 = CabanaPD::DiagonalIndexing<1>;
    IndexingType1 indexing1;

    ASSERT_EQ( indexing1( 0, 0 ), 0 );

    // test inverse indexing
    ASSERT_EQ( IndexingType1::template getInverseIndexPair<0>().first, 0 );
    ASSERT_EQ( IndexingType1::template getInverseIndexPair<0>().second, 0 );

    // test diagonal indexing for N=2
    using IndexingType2 = CabanaPD::DiagonalIndexing<2>;
    IndexingType2 indexing2;

    // main diagonal
    ASSERT_EQ( indexing2( 0, 0 ), 0 );
    ASSERT_EQ( indexing2( 1, 1 ), 1 );

    // first minor diagonal (incl symmetric terms)
    ASSERT_EQ( indexing2( 0, 1 ), 2 );
    ASSERT_EQ( indexing2( 1, 0 ), 2 );

    // test inverse indexing
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<2>().first, 0 );
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<2>().second, 1 );

    // test diagonal indexing for N=3
    using IndexingType3 = CabanaPD::DiagonalIndexing<3>;
    IndexingType3 indexing3;

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

    // test inverse indexing
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<3>().first, 0 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<3>().second, 1 );

    ASSERT_EQ( IndexingType3::template getInverseIndexPair<4>().first, 1 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<4>().second, 2 );

    ASSERT_EQ( IndexingType3::template getInverseIndexPair<5>().first, 0 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<5>().second, 2 );
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
    // test binary indexing
    CabanaPD::BinaryIndexing indexing;

    // test binary indexing for N=3
    for ( unsigned i = 0; i < 3; ++i )
    {
        for ( unsigned j = 0; j < 3; ++j )
        {
            if ( i == j )
                ASSERT_EQ( indexing( i, j ), 0 );
            else
                ASSERT_EQ( indexing( i, j ), 1 );
        }
    }
}

TEST( TEST_CATEGORY, test_fullIndexing )
{
    // test full indexing for N=1
    using IndexingType1 = CabanaPD::FullIndexing<1>;
    IndexingType1 indexing1;

    ASSERT_EQ( indexing1( 0, 0 ), 0 );

    // test inverse indexing
    ASSERT_EQ( IndexingType1::template getInverseIndexPair<0>().first, 0 );
    ASSERT_EQ( IndexingType1::template getInverseIndexPair<0>().second, 0 );

    // test diagonal indexing for N=2
    using IndexingType2 = CabanaPD::FullIndexing<2>;
    IndexingType2 indexing2;

    ASSERT_EQ( indexing2( 0, 0 ), 0 );
    ASSERT_EQ( indexing2( 0, 1 ), 1 );
    ASSERT_EQ( indexing2( 1, 0 ), 2 );
    ASSERT_EQ( indexing2( 1, 1 ), 3 );

    // test inverse indexing
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<0>().first, 0 );
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<0>().second, 0 );

    ASSERT_EQ( IndexingType2::template getInverseIndexPair<1>().first, 0 );
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<1>().second, 1 );

    ASSERT_EQ( IndexingType2::template getInverseIndexPair<2>().first, 1 );
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<2>().second, 0 );

    ASSERT_EQ( IndexingType2::template getInverseIndexPair<3>().first, 1 );
    ASSERT_EQ( IndexingType2::template getInverseIndexPair<3>().second, 1 );

    // test diagonal indexing for N=3
    using IndexingType3 = CabanaPD::FullIndexing<3>;
    IndexingType3 indexing3;

    ASSERT_EQ( indexing3( 0, 0 ), 0 );
    ASSERT_EQ( indexing3( 0, 1 ), 1 );
    ASSERT_EQ( indexing3( 0, 2 ), 2 );
    ASSERT_EQ( indexing3( 1, 0 ), 3 );
    ASSERT_EQ( indexing3( 1, 1 ), 4 );
    ASSERT_EQ( indexing3( 1, 2 ), 5 );
    ASSERT_EQ( indexing3( 2, 0 ), 6 );
    ASSERT_EQ( indexing3( 2, 1 ), 7 );
    ASSERT_EQ( indexing3( 2, 2 ), 8 );

    // test inverse indexing
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<3>().first, 1 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<3>().second, 0 );

    ASSERT_EQ( IndexingType3::template getInverseIndexPair<4>().first, 1 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<4>().second, 1 );

    ASSERT_EQ( IndexingType3::template getInverseIndexPair<5>().first, 1 );
    ASSERT_EQ( IndexingType3::template getInverseIndexPair<5>().second, 2 );
}

TEST( TEST_CATEGORY, test_fullIndexing_death )
{
    // assert death for out of scope indexing
    ASSERT_DEATH(
        {
            CabanaPD::FullIndexing<2> indexing;
            auto i = indexing( 2, 0 );
            (void)i;
        },
        "Index out of range of FullIndexing" );
    ASSERT_DEATH(
        {
            CabanaPD::FullIndexing<2> indexing;
            auto i = indexing( 0, 2 );
            (void)i;
        },
        "Index out of range of FullIndexing" );
}
} // end namespace Test
