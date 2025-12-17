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

#include <CabanaPD_config.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_ForceModelsMulti.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Particles.hpp>
#include <force/CabanaPD_LPS.hpp>
#include <force/CabanaPD_PMB.hpp>
#include <force_models/CabanaPD_LPS.hpp>
#include <force_models/CabanaPD_PMB.hpp>

#include <type_traits>

namespace Test
{
struct LinearTag
{
};
struct QuadraticTag
{
};

template <typename ModelType>
struct Inputs;

template <>
struct Inputs<CabanaPD::PMB>
{
    using base_model = CabanaPD::PMB;
    double force_horizon;
    double K;
    double coeff;
    double boundary_width;

    // This represents either stretch or displacement for the linear or
    // quadratic cases below.
    void update( const double new_coeff ) { coeff = new_coeff; }
};

template <>
struct Inputs<CabanaPD::LPS>
{
    using base_model = CabanaPD::LPS;
    double force_horizon;
    double K;
    double G;
    double coeff;
    double boundary_width;

    // This represents either stretch or displacement for the linear or
    // quadratic cases below.
    void update( const double new_coeff ) { coeff = new_coeff; }
};

//---------------------------------------------------------------------------//
// Reference particle summations.
//---------------------------------------------------------------------------//
// Note: all of these reference calculations assume uniform volume and a full
// particle neighborhood.
//---------------------------------------------------------------------------//

// Get the PMB stress tensor (at the center point).
// Linear case: simplified here because the stretch is constant.
template <class ModelType>
auto computeReferenceStress(
    LinearTag, ModelType model, const int m, const double s0,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double f_mag;
    double f_x, f_y, f_z;
    // Components xx, yy, zz
    std::array<double, 3> sigma = { 0.0, 0.0, 0.0 };
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;

    // Stress tensor.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi =
                    std::sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    // This i/j indexing has to be valid for the multi-material
                    // cases, but everything is the same type so it doesn't
                    // matter which particle we check.
                    f_mag = model( CabanaPD::ForceCoeffTag{}, 0, 0, s0, vol );
                    f_x = f_mag * xi_x / xi;
                    f_y = f_mag * xi_y / xi;
                    f_z = f_mag * xi_z / xi;
                    sigma[0] += 0.5 * f_x * xi_x;
                    sigma[1] += 0.5 * f_y * xi_y;
                    sigma[2] += 0.5 * f_z * xi_z;
                }
            }
    return sigma;
}

// Get the PMB strain energy density (at the center point).
// Linear case: Simplified here because the stretch is constant.
template <class ModelType>
double computeReferenceStrainEnergyDensity(
    LinearTag, ModelType model, const int m, const double s0, const double,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double W = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    W += model( CabanaPD::EnergyTag{}, 0, 0, s0, xi, vol );
                }
            }
    return W;
}

// Quadratic case: assumes zero y- and z-displacement components.
template <class ModelType>
double computeReferenceStrainEnergyDensity(
    QuadraticTag, ModelType model, const int m, const double u11,
    const double x,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double W = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    W += model( CabanaPD::EnergyTag{}, 0, 0, s, xi, vol );
                }
            }
    return W;
}

// Get the PMB internal force density (at one point).
// Linear case: always zero
template <class ModelType>
double computeReferenceForceX( LinearTag, ModelType, const int, const double,
                               const double )
{
    return 0.0;
}

// Quadratic case: assumes zero y- and z-displacement components.
template <class ModelType>
double computeReferenceForceX(
    QuadraticTag, ModelType model, const int m, const double u11,
    const double x,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double fx = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                    fx += model( CabanaPD::ForceCoeffTag{}, 0, 0, s, vol ) *
                          rx / r;
            }
    return fx;
}

// Get the weighted volume (at one point).
template <class ModelType>
double computeReferenceWeightedVolume( ModelType model, const int m,
                                       const double vol )
{
    double dx = model.force_horizon / m;
    double weighted_volume = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                    weighted_volume +=
                        model( CabanaPD::InfluenceFunctionTag{}, 0, 0, xi ) *
                        xi * xi * vol;
            }
    return weighted_volume;
}

// Get the dilatation (at one point).
// Linear case: Simplified here because the stretch is constant.
template <class ModelType>
double computeReferenceDilatation( ModelType model, const int m,
                                   const double s0, const double vol,
                                   const double weighted_volume )
{
    double dx = model.force_horizon / m;
    double theta = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                    theta +=
                        3.0 / weighted_volume *
                        model( CabanaPD::InfluenceFunctionTag{}, 0, 0, xi ) *
                        s0 * xi * xi * vol;
            }
    return theta;
}

// Quadratic case: assumes zero y- and z-displacement components.
template <class ModelType>
double computeReferenceDilatation( ModelType model, const int m,
                                   const double u11, const double vol,
                                   const double weighted_volume,
                                   const double x )
{
    double dx = model.force_horizon / m;
    double theta = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                    theta +=
                        3.0 / weighted_volume *
                        model( CabanaPD::InfluenceFunctionTag{}, 0, 0, xi ) *
                        s * xi * xi * vol;
            }
    return theta;
}

// Get the number of neighbors (at one point).
double computeReferenceNeighbors( const double horizon, const int m )
{
    double dx = horizon / m;
    double num_neighbors = 0.0;
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < horizon + 1e-14 )
                    num_neighbors += 1.0;
            }
    return num_neighbors;
}

// Get the LPS stress tensor (at one point).
// Linear case: simplified here because the stretch is constant.
template <class ModelType>
auto computeReferenceStress(
    LinearTag, ModelType model, const int m, const double s0,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double f_mag;
    double f_x, f_y, f_z;
    // Components xx, yy, zz
    std::array<double, 3> sigma = { 0.0, 0.0, 0.0 };
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;

    auto weighted_volume = computeReferenceWeightedVolume( model, m, vol );
    auto theta =
        computeReferenceDilatation( model, m, s0, vol, weighted_volume );

    // Stress tensor.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    // We assume the dilatation and weighted volume are constant
                    f_mag =
                        model( CabanaPD::ForceCoeffTag{},
                               CabanaPD::SingleMaterial{}, 0, 0, s0, xi, vol,
                               weighted_volume, weighted_volume, theta, theta );
                    f_x = f_mag * xi_x / xi;
                    f_y = f_mag * xi_y / xi;
                    f_z = f_mag * xi_z / xi;
                    sigma[0] += 0.5 * f_x * xi_x;
                    sigma[1] += 0.5 * f_y * xi_y;
                    sigma[2] += 0.5 * f_z * xi_z;
                }
            }
    return sigma;
}

// Quadratic case: not calculated.
template <class ModelType>
auto computeReferenceStress( QuadraticTag, ModelType, const int, const double )
{
    std::array<double, 3> sigma = { 0.0, 0.0, 0.0 };
    return sigma;
}

// Get the LPS strain energy density (at one point).
// Linear case: simplified here because the stretch is constant.
template <class ModelType>
double computeReferenceStrainEnergyDensity(
    LinearTag, ModelType model, const int m, const double s0, const double,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double W = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;

    auto weighted_volume = computeReferenceWeightedVolume( model, m, vol );
    auto theta =
        computeReferenceDilatation( model, m, s0, vol, weighted_volume );
    auto num_neighbors = computeReferenceNeighbors( model.force_horizon, m );

    // Strain energy density.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    W += model( CabanaPD::EnergyTag{},
                                CabanaPD::SingleMaterial{}, 0, 0, s0, xi, vol,
                                weighted_volume, theta, num_neighbors );
                }
            }
    return W;
}

// Quadratic case: Assumes zero y- and z-displacement components.
template <class ModelType>
double computeReferenceStrainEnergyDensity(
    QuadraticTag, ModelType model, const int m, const double u11,
    const double x,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double W = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;

    auto weighted_volume = computeReferenceWeightedVolume( model, m, vol );
    auto num_neighbors = computeReferenceNeighbors( model.force_horizon, m );
    auto theta_i =
        computeReferenceDilatation( model, m, u11, vol, weighted_volume, x );

    // Strain energy density.
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    W += model( CabanaPD::EnergyTag{},
                                CabanaPD::SingleMaterial{}, 0, 0, s, xi, vol,
                                weighted_volume, theta_i, num_neighbors );
                }
            }
    return W;
}

// Get the LPS internal force density (at one point).
// Quadratic case: assumes zero y- and z-displacement components.
template <class ModelType>
double computeReferenceForceX(
    QuadraticTag, ModelType model, const int m, const double u11,
    const double x,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double fx = 0.0;
    double dx = model.force_horizon / m;
    double vol = dx * dx * dx;

    auto weighted_volume = computeReferenceWeightedVolume( model, m, vol );
    auto theta_i =
        computeReferenceDilatation( model, m, u11, vol, weighted_volume, x );
    for ( int i = -m; i < m + 1; ++i )
        for ( int j = -m; j < m + 1; ++j )
            for ( int k = -m; k < m + 1; ++k )
            {
                double xi_x = dx * i;
                double xi_y = dx * j;
                double xi_z = dx * k;
                double eta_u = u11 * ( 2 * x * xi_x + xi_x * xi_x );
                double rx = xi_x + eta_u;
                double ry = xi_y;
                double rz = xi_z;
                double r = sqrt( rx * rx + ry * ry + rz * rz );
                double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
                double s = ( r - xi ) / xi;

                double x_j = x + xi_x;
                auto theta_j = computeReferenceDilatation(
                    model, m, u11, vol, weighted_volume, x_j );
                if ( xi > 0.0 && xi < model.force_horizon + 1e-14 )
                {
                    fx += model( CabanaPD::ForceCoeffTag{},
                                 CabanaPD::SingleMaterial{}, 0, 0, s, xi, vol,
                                 weighted_volume, weighted_volume, theta_i,
                                 theta_j ) *
                          rx / r;
                }
            }
    return fx;
}

//---------------------------------------------------------------------------//
// System creation.
//---------------------------------------------------------------------------//
namespace
{

template <class X, class U, class V>
struct InitFunctor
{
    X x;
    U u;
    V v;
    double s0;

    InitFunctor( X _x, U _u, V _v, double _s0 )
        : x( _x )
        , u( _u )
        , v( _v )
        , s0( _s0 )
    {
    }

    KOKKOS_FUNCTION void operator()( const int pid ) const
    {
        for ( int d = 0; d < 3; d++ )
        {
            u( pid, d ) = s0 * x( pid, d );
            v( pid, d ) = 0.0;
        }
    }
};

template <class T>
struct TempInitFunctor
{
    T t;
    double temp0;

    TempInitFunctor( T _t, double _temp0 )
        : t( _t )
        , temp0( _temp0 )
    {
    }

    KOKKOS_FUNCTION void operator()( const int pid ) const { t( pid ) = temp0; }
};

} // namespace
template <class ModelTag, class ThermalTag = CabanaPD::TemperatureIndependent>
auto createParticles( ModelTag tag, LinearTag, const double dx, const double s0,
                      ThermalTag thermal_tag = ThermalTag{},
                      const double temp0 = 0. )
{
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    int nc = ( box_max[0] - box_min[0] ) / dx;
    std::array<int, 3> num_cells = { nc, nc, nc };

    // Create particles based on the mesh
    auto particles = CabanaPD::Particles( TEST_MEMSPACE{}, tag, thermal_tag,
                                          CabanaPD::EnergyStressOutput{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( TEST_EXECSPACE{} );

    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto v = particles.sliceVelocity();
    particles.update( TEST_EXECSPACE{}, InitFunctor( x, u, v, s0 ) );
    if constexpr ( !std::is_same_v<ThermalTag,
                                   CabanaPD::TemperatureIndependent> )
    {
        auto t = particles.sliceTemperature();
        particles.update( TEST_EXECSPACE{}, TempInitFunctor( t, temp0 ) );
    }
    (void)temp0; // silence unused parameter warning if TemperatureIndependent
    return particles;
}

template <class ModelTag>
CabanaPD::Particles<TEST_MEMSPACE, ModelTag, CabanaPD::TemperatureIndependent,
                    CabanaPD::EnergyStressOutput>
createParticles( ModelTag tag, QuadraticTag, const double dx, const double s0 )
{
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    int nc = ( box_max[0] - box_min[0] ) / dx;
    std::array<int, 3> num_cells = { nc, nc, nc };

    // Create particles based on the mesh.
    CabanaPD::Particles particles( TEST_MEMSPACE{}, tag,
                                   CabanaPD::EnergyStressOutput{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( TEST_EXECSPACE{} );

    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto v = particles.sliceVelocity();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        for ( int d = 0; d < 3; d++ )
        {
            u( pid, d ) = 0.0;
            v( pid, d ) = 0.0;
        }
        u( pid, 0 ) = s0 * x( pid, 0 ) * x( pid, 0 );
    };
    particles.update( TEST_EXECSPACE{}, init_functor );
    return particles;
}

//---------------------------------------------------------------------------//
// Check all particles.
//---------------------------------------------------------------------------//
template <class HostParticleType, class TestType, class ModelType,
          class InputType>
void checkResults( HostParticleType aosoa_host, double local_min[3],
                   double local_max[3], TestType test_tag, ModelType model,
                   const int m, const InputType inputs, const double Phi )
{
    const double s0 = inputs.coeff;
    const double boundary_width = inputs.boundary_width;
    double force_horizon = inputs.force_horizon;
    double ref_Phi = 0.0;
    auto sigma_host = Cabana::slice<0>( aosoa_host );
    auto f_host = Cabana::slice<1>( aosoa_host );
    auto x_host = Cabana::slice<2>( aosoa_host );
    auto W_host = Cabana::slice<3>( aosoa_host );
    auto vol_host = Cabana::slice<4>( aosoa_host );
    auto theta_host = Cabana::slice<5>( aosoa_host );
    // Check the results: avoid the system boundary for per particle values.
    int particles_checked = 0;
    for ( std::size_t p = 0; p < aosoa_host.size(); ++p )
    {
        double x = x_host( p, 0 );
        double y = x_host( p, 1 );
        double z = x_host( p, 2 );
        if ( x > local_min[0] + force_horizon * boundary_width &&
             x < local_max[0] - force_horizon * boundary_width &&
             y > local_min[1] + force_horizon * boundary_width &&
             y < local_max[1] - force_horizon * boundary_width &&
             z > local_min[2] + force_horizon * boundary_width &&
             z < local_max[2] - force_horizon * boundary_width )
        {
            // These are constant for linear, but vary for quadratic.
            auto ref_sigma = computeReferenceStress( test_tag, model, m, s0 );
            double ref_W = computeReferenceStrainEnergyDensity( test_tag, model,
                                                                m, s0, x );
            double ref_f = computeReferenceForceX( test_tag, model, m, s0, x );

            std::array<double, 3> f = { f_host( p, 0 ), f_host( p, 1 ),
                                        f_host( p, 2 ) };
            // Components xx, yy, zz, xy, xz, yz, yx, zx, zy
            std::array<double, 9> sigma = {
                sigma_host( p, 0, 0 ), sigma_host( p, 1, 1 ),
                sigma_host( p, 2, 2 ), sigma_host( p, 0, 1 ),
                sigma_host( p, 0, 2 ), sigma_host( p, 1, 2 ),
                sigma_host( p, 1, 0 ), sigma_host( p, 2, 0 ),
                sigma_host( p, 2, 1 ) };
            checkParticle( test_tag, inputs, f, ref_f, W_host( p ), ref_W,
                           sigma, ref_sigma, x );
            particles_checked++;
        }
        checkAnalyticalDilatation( model, test_tag, inputs.coeff,
                                   theta_host( p ) );

        // Check total sum of strain energy matches per particle sum.
        ref_Phi += W_host( p ) * vol_host( p );
    }

    EXPECT_NEAR( Phi, ref_Phi, 1e-5 );
    std::cout << "Particles checked: " << particles_checked << std::endl;
}

//---------------------------------------------------------------------------//
// Individual checks per particle.
//---------------------------------------------------------------------------//
template <class InputType>
void checkParticle( LinearTag tag, const InputType inputs,
                    const std::array<double, 3> f, const double, const double W,
                    const double ref_W, const std::array<double, 9> sigma,
                    const std::array<double, 3> ref_sigma, const double )
{
    for ( std::size_t d = 0; d < 3; ++d )
        EXPECT_LE( f[d], 1e-13 );

    // Check strain energy density (all should be equal for fixed stretch).
    EXPECT_FLOAT_EQ( W, ref_W );

    // Check strain energy density with analytical value.
    checkAnalyticalStrainEnergy( tag, inputs, W, -1 );

    // Check stress (diagonal should be equal for fixed stretch).
    for ( std::size_t d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ( sigma[d], ref_sigma[d] );
    // Other components should be zero.
    for ( std::size_t d = 3; d < 9; ++d )
        EXPECT_LE( sigma[d], 1e-13 );
}

template <class InputType>
void checkParticle( QuadraticTag tag, const InputType inputs,
                    const std::array<double, 3> f, const double ref_f,
                    const double W, const double ref_W,
                    const std::array<double, 9>, const std::array<double, 3>,
                    const double x )
{
    // Check force in x with discretized result (reference currently incorrect).
    EXPECT_FLOAT_EQ( f[0], ref_f );

    // Check force in x with analytical value.
    checkAnalyticalForce( tag, inputs, f[0] );

    // Check force: other components should be zero.
    EXPECT_LE( f[1], 1e-13 );
    EXPECT_LE( f[2], 1e-13 );

    // Check energy. Not quite within the floating point tolerance.
    EXPECT_NEAR( W, ref_W, 1e-6 );

    // Check energy with analytical value.
    checkAnalyticalStrainEnergy( tag, inputs, W, x );
}

template <class InputType>
void checkAnalyticalStrainEnergy(
    LinearTag, const InputType inputs, const double W, const double,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    // Relatively large error for small m.
    double threshold = W * 0.15;
    double K = inputs.K;
    double s0 = inputs.coeff;
    double analytical_W = 9.0 / 2.0 * K * s0 * s0;
    EXPECT_NEAR( W, analytical_W, threshold );
}

template <class InputType>
void checkAnalyticalStrainEnergy(
    LinearTag, const InputType inputs, const double W, const double,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    // LPS is exact.
    double K = inputs.K;
    double s0 = inputs.coeff;
    double analytical_W = 9.0 / 2.0 * K * s0 * s0;
    EXPECT_FLOAT_EQ( W, analytical_W );
}

template <class InputType>
void checkAnalyticalStrainEnergy(
    QuadraticTag, const InputType inputs, const double W, const double x,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double threshold = W * 0.05;
    const double u11 = inputs.coeff;
    double K = inputs.K;
    double analytical_W =
        18.0 * K * u11 * u11 *
        ( 1.0 / 5.0 * x * x +
          inputs.force_horizon * inputs.force_horizon / 42.0 );
    EXPECT_NEAR( W, analytical_W, threshold );
}

template <class InputType>
void checkAnalyticalStrainEnergy(
    QuadraticTag, const InputType inputs, const double W, const double x,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double threshold = W * 0.20;
    const double u11 = inputs.coeff;
    double K = inputs.K;
    double G = inputs.G;
    double analytical_W =
        u11 * u11 *
        ( ( 2.0 * K + 8.0 / 3.0 * G ) * x * x +
          75.0 / 2.0 * G * inputs.force_horizon * inputs.force_horizon / 49.0 );
    EXPECT_NEAR( W, analytical_W, threshold );
}

template <class InputType>
void checkAnalyticalForce(
    QuadraticTag, const InputType inputs, const double fx,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    double threshold = fx * 0.10;
    double K = inputs.K;
    double s0 = inputs.coeff;
    double analytical_f = 18.0 / 5.0 * K * s0;
    EXPECT_NEAR( fx, analytical_f, threshold );
}

template <class InputType>
void checkAnalyticalForce(
    QuadraticTag, const InputType inputs, const double fx,
    typename std::enable_if<
        ( std::is_same<typename InputType::base_model, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    double threshold = fx * 0.10;
    double K = inputs.K;
    double G = inputs.G;
    double s0 = inputs.coeff;
    double analytical_f = 2.0 * ( K + 4.0 / 3.0 * G ) * s0;
    EXPECT_NEAR( fx, analytical_f, threshold );
}

template <class ModelType>
void checkAnalyticalDilatation(
    ModelType, LinearTag, const double, const double theta,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::PMB>::value ),
        int>::type* = 0 )
{
    EXPECT_FLOAT_EQ( 0.0, theta );
}

template <class ModelType>
void checkAnalyticalDilatation(
    ModelType, LinearTag, const double s0, const double theta,
    typename std::enable_if<
        ( std::is_same<typename ModelType::model_tag, CabanaPD::LPS>::value ),
        int>::type* = 0 )
{
    EXPECT_FLOAT_EQ( 3 * s0, theta );
}

template <class ModelType>
void checkAnalyticalDilatation( ModelType, QuadraticTag, const double,
                                const double )
{
}

template <class ModelType, class ForceType, class ParticleType,
          class NeighborType>
void initializeForce( const ModelType& model, ForceType& force,
                      ParticleType& particles, const NeighborType& neighbor )
{
    force.computeWeightedVolume( model, particles, neighbor );
    force.computeDilatation( model, particles, neighbor );
}

template <class ParticleType, class AoSoAType>
void copyTheta( CabanaPD::PMB, const ParticleType&, AoSoAType& aosoa_host )
{
    auto theta_host = Cabana::slice<5>( aosoa_host );
    Cabana::deep_copy( theta_host, 0.0 );
}

template <class ParticleType, class AoSoAType>
void copyTheta( CabanaPD::LPS, const ParticleType& particles,
                AoSoAType& aosoa_host )
{
    auto theta_host = Cabana::slice<5>( aosoa_host );
    auto theta = particles.sliceDilatation();
    Cabana::deep_copy( theta_host, theta );
}

//---------------------------------------------------------------------------//
// Main test function.
//---------------------------------------------------------------------------//
template <class ModelType, class TestType, class InputType>
void testForce( ModelType model, const double dx, const double m,
                const TestType test_tag, const InputType inputs )
{
    using force_tag = typename ModelType::force_tag;
    using model_tag = typename ModelType::model_tag;
    using fracture_tag = typename ModelType::fracture_type;
    auto particles = createParticles( model_tag{}, test_tag, dx, inputs.coeff );

    // This needs to exactly match the mesh spacing to compare with the single
    // particle calculation.
    CabanaPD::Force<TEST_MEMSPACE, force_tag, fracture_tag> force;
    CabanaPD::Neighbor<TEST_MEMSPACE, fracture_tag> neighbor( model,
                                                              particles );
    auto x = particles.sliceReferencePosition();
    auto f = particles.sliceForce();
    auto W = particles.sliceStrainEnergy();
    auto sigma = particles.sliceStress();
    auto vol = particles.sliceVolume();
    //  No communication needed (as in the main solver) since this test is only
    //  intended for one rank.
    initializeForce( model, force, particles, neighbor );

    computeForce( model, force, particles, neighbor );
    computeEnergy( model, force, particles, neighbor );
    computeStress( model, force, particles, neighbor );

    // Make a copy of final results on the host
    std::size_t num_particle = x.size();
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3][3], double[3], double[3],
                                          double, double, double>,
                      Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", num_particle );
    auto sigma_host = Cabana::slice<0>( aosoa_host );
    auto f_host = Cabana::slice<1>( aosoa_host );
    auto x_host = Cabana::slice<2>( aosoa_host );
    auto W_host = Cabana::slice<3>( aosoa_host );
    auto vol_host = Cabana::slice<4>( aosoa_host );
    Cabana::deep_copy( sigma_host, sigma );
    Cabana::deep_copy( f_host, f );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( W_host, W );
    Cabana::deep_copy( vol_host, vol );
    copyTheta( typename ModelType::model_tag{}, particles, aosoa_host );

    double local_min[3] = { particles.local_mesh_lo[0],
                            particles.local_mesh_lo[1],
                            particles.local_mesh_lo[2] };
    double local_max[3] = { particles.local_mesh_hi[0],
                            particles.local_mesh_hi[1],
                            particles.local_mesh_hi[2] };
    double Phi = force.totalStrainEnergy();
    checkResults( aosoa_host, local_min, local_max, test_tag, model, m, inputs,
                  Phi );
}

//---------------------------------------------------------------------------//
// Neighbor test function.
//---------------------------------------------------------------------------//
void testNeighbor( const double dx, const double m, const double force_horizon )
{
    CabanaPD::PMB model_tag;
    CabanaPD::ForceModel model( model_tag, CabanaPD::Elastic{},
                                CabanaPD::NoFracture{}, force_horizon, 1.0 );

    auto particles = createParticles( model_tag, LinearTag{}, dx, 0.0 );

    CabanaPD::Neighbor<TEST_MEMSPACE, CabanaPD::NoFracture> neighbor(
        model, particles );

    auto expected_num_neighbors =
        computeReferenceNeighbors( model.force_horizon, m );
    EXPECT_EQ( expected_num_neighbors, neighbor.getMaxLocal() );
}

//---------------------------------------------------------------------------//
// GTest tests.
//---------------------------------------------------------------------------//

TEST( TEST_CATEGORY, test_neighbor )
{
    // dx needs to be decreased for increased m: boundary particles are ignored.
    double m = 3;
    double dx = 2.0 / 11.0;
    double force_horizon = dx * m;
    testNeighbor( dx, m, force_horizon );

    m = 6;
    dx = 2.0 / 15.0;
    force_horizon = dx * m;
    testNeighbor( dx, m, force_horizon );
}

// Test construction of all PMB models.
TEST( TEST_CATEGORY, test_force_pmb_construct )
{
    double force_horizon = 5.0;
    double K = 1.0;
    double G0 = 100.0;

    {
        // With defaults.
        CabanaPD::ForceModel model( CabanaPD::PMB{}, force_horizon, K, G0 );
    }
    {
        // With mechanics input.
        CabanaPD::ForceModel model( CabanaPD::PMB{}, CabanaPD::Elastic{},
                                    force_horizon, K, G0 );
    }
    {
        // With all inputs.
        CabanaPD::ForceModel model( CabanaPD::PMB{}, CabanaPD::Elastic{},
                                    CabanaPD::Fracture{}, force_horizon, K,
                                    G0 );
    }
    // Without fracture.
    {
        CabanaPD::ForceModel model( CabanaPD::PMB{}, CabanaPD::Elastic{},
                                    CabanaPD::NoFracture{}, force_horizon, K );
    }
    {
        CabanaPD::ForceModel model( CabanaPD::PMB{}, CabanaPD::NoFracture{},
                                    force_horizon, K );
    }
    // With EPP (cannot be run without fracture).
    double sigma_y = 10.0;
    {
        CabanaPD::ForceModel force_model(
            CabanaPD::PMB{}, CabanaPD::ElasticPerfectlyPlastic{},
            TEST_MEMSPACE{}, force_horizon, K, G0, sigma_y );
    }

    // With thermomechanics.
    double alpha = 1.0;
    double temp0 = 0.0;
    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };
    CabanaPD::Particles particles( TEST_MEMSPACE{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureDependent{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( TEST_EXECSPACE{} );
    auto temp = particles.sliceTemperature();
    {
        //  With elastic, without fracture.
        CabanaPD::ForceModel force_model( CabanaPD::PMB{},
                                          CabanaPD::NoFracture{}, force_horizon,
                                          K, temp, alpha, temp0 );
    }
    {
        //  With elastic.
        CabanaPD::ForceModel force_model( CabanaPD::PMB{}, force_horizon, K, G0,
                                          temp, alpha, temp0 );
    }
    {
        //  With EPP.
        CabanaPD::ForceModel force_model(
            CabanaPD::PMB{}, CabanaPD::ElasticPerfectlyPlastic{}, force_horizon,
            K, G0, sigma_y, temp, alpha, temp0 );
    }

    // With heat transfer.
    double kappa = 1.0;
    double cp = 1.0;
    {
        CabanaPD::ForceModel force_model( CabanaPD::PMB{},
                                          CabanaPD::NoFracture{}, force_horizon,
                                          K, temp, kappa, cp, alpha, temp0 );
    }
    {
        CabanaPD::ForceModel force_model( CabanaPD::PMB{}, force_horizon, K, G0,
                                          temp, kappa, cp, alpha, temp0 );
    }
    {
        //  With EPP.
        CabanaPD::ForceModel force_model(
            CabanaPD::PMB{}, CabanaPD::ElasticPerfectlyPlastic{}, force_horizon,
            K, G0, sigma_y, temp, kappa, cp, alpha, temp0 );
    }
}

// Test construction of all LPS models.
TEST( TEST_CATEGORY, test_force_lps_construct )
{
    double force_horizon = 5.0;
    double K = 1.0;
    double G = 2.0;
    double G0 = 100.0;

    {
        // LPS with defaults.
        CabanaPD::ForceModel model( CabanaPD::LPS{}, force_horizon, K, G, G0,
                                    1 );
    }
    {
        // LPS with mechanics input.
        CabanaPD::ForceModel model( CabanaPD::LPS{}, CabanaPD::Elastic{},
                                    force_horizon, K, G, G0, 1 );
    }
    {
        // LPS with all inputs.
        CabanaPD::ForceModel model( CabanaPD::LPS{}, CabanaPD::Elastic{},
                                    CabanaPD::Fracture{}, force_horizon, K, G,
                                    G0, 1 );
    }
    // LPS without fracture.
    {
        CabanaPD::ForceModel model( CabanaPD::LPS{}, CabanaPD::Elastic{},
                                    CabanaPD::NoFracture{}, force_horizon, K, G,
                                    1 );
    }
    {
        CabanaPD::ForceModel model( CabanaPD::LPS{}, CabanaPD::NoFracture{},
                                    force_horizon, K, G, 1 );
    }
}

TEST( TEST_CATEGORY, test_force_pmb )
{
    // dx needs to be decreased for increased m: boundary particles are ignored.
    double m = 3;
    double dx = 2.0 / 11.0;
    double force_horizon = dx * m;
    double K = 1.0;
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel model( model_type{}, CabanaPD::Elastic{},
                                CabanaPD::NoFracture{}, force_horizon, K );

    Inputs<model_type> inputs{ force_horizon, K, 0.1, 1.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( model, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_linear_pmb )
{
    double m = 3;
    double dx = 2.0 / 11.0;
    double force_horizon = dx * m;
    double K = 1.0;
    CabanaPD::ForceModel model( CabanaPD::LinearPMB{}, CabanaPD::Elastic{},
                                CabanaPD::NoFracture{}, force_horizon, K );

    Inputs<CabanaPD::PMB> inputs{ force_horizon, K, 0.1, 1.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_lps )
{
    double m = 3;
    // Need a larger system than PMB because the boundary region is larger.
    double dx = 2.0 / 15.0;
    double force_horizon = dx * m;
    double K = 1.0;
    double G = 0.5;
    using model_type = CabanaPD::LPS;
    CabanaPD::ForceModel model( model_type{}, CabanaPD::Elastic{},
                                CabanaPD::NoFracture{}, force_horizon, K, G,
                                1 );

    Inputs<model_type> inputs{ force_horizon, K, G, 0.1, 2.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( model, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_linear_lps )
{
    double m = 3;
    double dx = 2.0 / 15.0;
    double force_horizon = dx * m;
    double K = 1.0;
    double G = 0.5;
    CabanaPD::ForceModel model( CabanaPD::LinearLPS{}, CabanaPD::NoFracture{},
                                force_horizon, K, G, 1 );
    Inputs<CabanaPD::LPS> inputs{ force_horizon, K, G, 0.1, 2.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
}

// Tests without damage, but using damage models.
TEST( TEST_CATEGORY, test_force_pmb_damage )
{
    double m = 3;
    double dx = 2.0 / 11.0;
    double force_horizon = dx * m;
    double K = 1.0;
    // Large value to make sure no bonds break.
    double G0 = 1000.0;
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel model( model_type{}, force_horizon, K, G0 );

    Inputs<model_type> inputs{ force_horizon, K, 0.1, 1.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( model, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_lps_damage )
{
    double m = 3;
    double dx = 2.0 / 15.0;
    double force_horizon = dx * m;
    double K = 1.0;
    double G = 0.5;
    double G0 = 1000.0;
    using model_type = CabanaPD::LPS;
    CabanaPD::ForceModel model( model_type{}, force_horizon, K, G, G0, 1 );

    Inputs<model_type> inputs{ force_horizon, K, G, 0.1, 2.1 };
    testForce( model, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( model, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_pmb_multi )
{
    // dx needs to be decreased for increased m: boundary particles are ignored.
    double m = 3;
    double dx = 2.0 / 11.0;
    double force_horizon = dx * m;
    double K = 1.0;
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel model1( model_type{}, CabanaPD::Elastic{},
                                 CabanaPD::NoFracture{}, force_horizon, K );
    CabanaPD::ForceModel model2( model1 );

    auto particles = createParticles( model_type{}, LinearTag{}, dx, 0.1 );
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, model1, model2 );

    Inputs<model_type> inputs{ force_horizon, K, 0.1, 1.1 };
    testForce( models, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( models, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_lps_multi )
{
    double m = 3;
    // Need a larger system than PMB because the boundary region is larger.
    double dx = 2.0 / 15.0;
    double force_horizon = dx * m;
    double K = 1.0;
    double G = 0.5;
    double G0 = 1000.0;
    using model_type = CabanaPD::LPS;
    CabanaPD::ForceModel model1( model_type{}, CabanaPD::Elastic{},
                                 CabanaPD::NoFracture{}, force_horizon, K, G,
                                 1 );
    CabanaPD::ForceModel model2( model_type{}, force_horizon, K, G, G0, 1 );

    auto particles = createParticles( model_type{}, LinearTag{}, dx, 0.1 );
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, model1, model2 );

    Inputs<model_type> inputs{ force_horizon, K, G, 0.1, 2.1 };
    testForce( models, dx, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( models, dx, m, QuadraticTag{}, inputs );
}
TEST( TEST_CATEGORY, test_force_thermal_pmb_multi )
{
    double m = 3;
    double dx = 2.0 / 11.0;
    double horizon = dx * m;
    double K = 1.0;
    double G0 = 1000.0;
    double kappa = 1.0;
    double cp = 1.0;
    double alpha = 1.0;
    double temp0 = 1.0;
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::DynamicTemperature;

    auto particles = createParticles( model_type{}, LinearTag{}, dx, 0.1,
                                      thermal_type{}, temp0 );
    auto temp = particles.sliceTemperature();

    CabanaPD::ForceModel model1( CabanaPD::PMB{}, horizon, K, G0, temp, kappa,
                                 cp, alpha, temp0 );
    CabanaPD::ForceModel model2( model1 );
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, model1, model2 );

    Inputs<model_type> inputs{ horizon, K, 0.1, 1.1 };
    testForce( models, horizon / m, m, LinearTag{}, inputs );
    inputs.update( 0.01 );
    testForce( models, horizon / m, m, QuadraticTag{}, inputs );
}

} // end namespace Test
