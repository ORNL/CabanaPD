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

#ifndef FORCE_MODELS_CORR_H
#define FORCE_MODELS_CORR_H

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <>
struct ForceModel<Correspondence, Elastic> : public BaseInfluenceForceModel
{
    using base_type = BaseInfluenceForceModel;
    using base_model = Correspondence;
    using fracture_type = Elastic;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;
    using base_type::influence_type;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    ForceModel( const double _delta, const double _K, const double _G,
                const int _influence = 0 )
        : base_type( _delta, _influence )
    {
        set_param( _delta, _K, _G );
    }

    void set_param( const double _delta, const double _K, const double _G )
    {
        delta = _delta;
        K = _K;
        G = _G;

        theta_coeff = 3.0 * K - 5.0 * G;
        s_coeff = 15.0 * G;
    }

    template <typename DefGradType>
    KOKKOS_INLINE_FUNCTION auto getStress( DefGradType F, const int i )
    {
        double epsilon[3][3];

        for ( std::size_t d = 0; d < 3; d++ )
        {
            epsilon[d][d] = F( i, d, d ) - 1.0;
            int d2 = getComponent( d );
            epsilon[d][d2] = 0.5 * ( F( i, d, d2 ) + F( i, d2, d ) );
            epsilon[d2][d] = epsilon[d][d2];
        }

        double sigma[3][3];
        double diag = ( K - 2.0 / 3.0 * G ) *
                      ( epsilon[0][0] + epsilon[1][1] + epsilon[2][2] );
        for ( std::size_t d = 0; d < 3; d++ )
        {
            sigma[d][d] = diag + 2.0 * G * epsilon[d][d];
            int d2 = getComponent( d );
            sigma[d][d2] = 2.0 * G * epsilon[d][d2];
            sigma[d2][d] = sigma[d][d2];
        }
        return sigma;
    }

    auto getComponent( const int d )
    {
        int d2;
        if ( d < 2 )
            return d + 1;
        else
            return 0;
    }
};

} // namespace CabanaPD

#endif
