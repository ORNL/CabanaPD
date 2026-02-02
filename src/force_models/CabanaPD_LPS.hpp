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

#ifndef FORCE_MODELS_LPS_H
#define FORCE_MODELS_LPS_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <typename MechanicsModelType>
struct BaseForceModelLPS;

template <>
struct BaseForceModelLPS<Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = LPS;
    using force_tag = LPS;

    using base_type::force_horizon;

    int influence_type;
    InfluenceFunctionTag influence_tag;

    using base_type::K;
    double G;
    // Store coefficients for multi-material systems.
    // TODO: this currently only supports bi-material systems.
    Kokkos::Array<double, 2> theta_coeff;
    Kokkos::Array<double, 2> s_coeff;

    BaseForceModelLPS( LPS, NoFracture, const double _force_horizon,
                       const double _K, const double _G,
                       const int _influence = 0 )
        : base_type( _force_horizon, _K )
        , influence_type( _influence )
        , G( _G )
    {
        init();
    }

    BaseForceModelLPS( LPS, Elastic, NoFracture, const double _force_horizon,
                       const double _K, const double _G,
                       const int _influence = 0 )
        : base_type( _force_horizon, _K )
        , influence_type( _influence )
        , G( _G )
    {
        init();
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelLPS( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
        G = ( model1.G + model2.G ) / 2.0;
        theta_coeff[0] = 3.0 * model1.K - 5.0 * model1.G;
        s_coeff[0] = 15.0 * model1.G;
        theta_coeff[1] = 3.0 * model2.K - 5.0 * model2.G;
        s_coeff[1] = 15.0 * model2.G;

        influence_type = model1.influence_type;
        if ( model2.influence_type != model1.influence_type )
            log_err( std::cout,
                     "Influence function type for each model must match for "
                     "multi-material systems" );
    }

    void init()
    {
        theta_coeff[0] = 3.0 * K - 5.0 * G;
        s_coeff[0] = 15.0 * G;
        // Set extra coefficients for multi-material.
        theta_coeff[1] = theta_coeff[0];
        s_coeff[1] = s_coeff[0];

        if ( influence_type > 1 || influence_type < 0 )
            log_err( std::cout, "Influence function type must be 0 or 1." );
    }

    KOKKOS_INLINE_FUNCTION auto operator()( InfluenceFunctionTag,
                                            double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }

    KOKKOS_INLINE_FUNCTION auto operator()( InfluenceFunctionTag, const int,
                                            const int, double xi ) const
    {
        return ( *this )( influence_tag, xi );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( WeightedVolumeTag, const int, const int, const double xi,
                     const double vol ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        return influence * xi * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION auto operator()( DilatationTag, const int, const int,
                                            const double s, const double xi,
                                            const double vol,
                                            const double m_i ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        double theta_i = influence * s * xi * xi * vol;
        return 3.0 * theta_i / m_i;
    }

    // In this case, we know that we only have one material so we use the first.
    // This must be separate from the multi-material interface because no type
    // information was passed here.
    KOKKOS_INLINE_FUNCTION auto
    operator()( ForceCoeffTag, SingleMaterial, const int, const int,
                const double s, const double xi, const double vol,
                const double m_i, const double m_j, const double theta_i,
                const double theta_j ) const
    {
        auto influence = ( *this )( influence_tag, xi );

        return ( theta_coeff[0] * ( theta_i / m_i + theta_j / m_j ) +
                 s_coeff[0] * s * ( 1.0 / m_i + 1.0 / m_j ) ) *
               influence * xi * vol;
    }

    // CI failures for gcc-13 in release show an apparent false-positive warning
    // for array bounds of the coefficients.
#pragma GCC diagnostic warning "-Warray-bounds"
#pragma GCC diagnostic push
    // In this case we may have any combination of material types. These
    // coefficients may still be the same for some interaction pairs.
    KOKKOS_INLINE_FUNCTION auto
    operator()( ForceCoeffTag, MultiMaterial, const int type_i,
                const int type_j, const double s, const double xi,
                const double vol, const double m_i, const double m_j,
                const double theta_i, const double theta_j ) const
    {
        KOKKOS_ASSERT( type_i < 2 );
        KOKKOS_ASSERT( type_j < 2 );
        auto influence = ( *this )( influence_tag, xi );
        double theta_coeff_i = theta_coeff[type_i];
        double theta_coeff_j = theta_coeff[type_j];
        double s_coeff_i = s_coeff[type_i];
        double s_coeff_j = s_coeff[type_j];

        return ( theta_coeff_i * theta_i / m_i + theta_coeff_j * theta_j / m_j +
                 s * ( s_coeff_i / m_i + s_coeff_j / m_j ) ) *
               influence * xi * vol;
    }

    // -----------------------------------------------------------------------------
    // ForceCoeffTag wrappers to accept the extra int argument inserted by ForceModelsMulti
    // while preserving the existing ForceCoeffTag double-parameter order:
    //   (s, xi, vol, m_i, m_j, theta_i, theta_j)
    // -----------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, SingleMaterial,
                     const int type_i, const int type_j, const int /*extra*/,
                     const double s, const double xi, const double vol,
                     const double m_i, const double m_j,
                     const double theta_i, const double theta_j ) const
    {
        return ( *this )( ForceCoeffTag{}, SingleMaterial{}, type_i, type_j,
                          s, xi, vol, m_i, m_j, theta_i, theta_j );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, MultiMaterial,
                     const int type_i, const int type_j, const int /*extra*/,
                     const double s, const double xi, const double vol,
                     const double m_i, const double m_j,
                     const double theta_i, const double theta_j ) const
    {
        return ( *this )( ForceCoeffTag{}, MultiMaterial{}, type_i, type_j,
                          s, xi, vol, m_i, m_j, theta_i, theta_j );
    }

    // In this case, we know that we only have one material so we use the first.
    // This must be separate from the multi-material interface because no type
    // information was passed here.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, SingleMaterial, const int, const int,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        return 1.0 / num_bonds * 0.5 * theta_coeff[0] / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff[0] / m_i ) * influence * s * s * xi * xi * vol;
    }

    // In this case we may have any combination of material types. These
    // coefficients may still be the same for some interaction pairs.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, MultiMaterial, const int type_i, const int,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        auto influence = ( *this )( influence_tag, xi );

        KOKKOS_ASSERT( type_i < 2 );
        double theta_coeff_i = theta_coeff[type_i];
        double s_coeff_i = s_coeff[type_i];
        return 1.0 / num_bonds * 0.5 * theta_coeff_i / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff_i / m_i ) * influence * s * s * xi * xi * vol;
    }

    // -----------------------------------------------------------------------------
    // EnergyTag wrappers to accept the extra int argument inserted by ForceModelsMulti
    // while preserving the existing EnergyTag double-parameter order:
    //   (s, xi, vol, m_i, theta_i, num_bonds)
    // -----------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, SingleMaterial,
                     const int type_i, const int type_j, const int /*extra*/,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        // Forward to the existing 2-int EnergyTag overload.
        return ( *this )( EnergyTag{}, SingleMaterial{}, type_i, type_j,
                          s, xi, vol, m_i, theta_i, num_bonds );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, MultiMaterial,
                     const int type_i, const int type_j, const int /*extra*/,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        // Forward to the existing 2-int EnergyTag overload.
        return ( *this )( EnergyTag{}, MultiMaterial{}, type_i, type_j,
                          s, xi, vol, m_i, theta_i, num_bonds );
    }
#pragma GCC diagnostic pop
};

template <typename MemorySpace>
struct BaseForceModelLPSPlastic
  : public BaseForceModelLPS<Elastic>,
    public BasePlasticity<MemorySpace>,
    public BaseHardening<MemorySpace>
{
  using base_elastic_type = BaseForceModelLPS<Elastic>;
  using base_plasticity_type = BasePlasticity<MemorySpace>;
  using base_hardening_type = BaseHardening<MemorySpace>;

  using base_elastic_type::K;
  using base_elastic_type::G;
  using base_elastic_type::theta_coeff;
  using base_elastic_type::s_coeff;
  using base_elastic_type::influence_type;
  using base_elastic_type::influence_tag;
  using base_elastic_type::operator();

  using base_plasticity_type::_s_p;
  using base_hardening_type::_alpha;

  double s_Y0;
  double H_s;
  double alpha_softening_start;
  double alpha_softening_final;
  double softening_gmin_;
  double yield_softening_start_;
  double yield_softening_final_;
  double yield_softening_min_factor_;
  double yield_softening_floor_factor_;
  int yield_softening_shape_;
  bool yield_softening_relative_;
  
  // Track alpha at first yield per bond-slot (for relative mode)
  Kokkos::View<double**, MemorySpace> _alpha_yield_onset;

  using base_plasticity_type::updateBonds;
  // Do NOT add a second updateBonds name; rely on BasePlasticity/BaseHardening having same signature.

  BaseForceModelLPSPlastic( LPS, LPSPlastic, MemorySpace,
                            const double delta, const double K_in, const double G_in,
                            const double sigma_y, const double H,
                            const int influence = 0,
                            const double a0 = -1.0, const double af = -1.0,
                            const double gmin = 1e-8,
                            const double ys0 = -1.0, const double ysf = -1.0,
                            const double ymin_fac = 1.0,
                            const double yfloor_fac = -1.0,
                            const int yshape = 0,
                            const bool yrelative = false )
    : base_elastic_type( LPS{}, NoFracture{}, delta, K_in, G_in, influence ),
      base_plasticity_type(),
      base_hardening_type()
  {
    s_Y0 = sigma_y / ( 3.0 * K_in );
    H_s  = H / ( 3.0 * K_in );
    alpha_softening_start = a0;
    alpha_softening_final = af;
    softening_gmin_ = gmin;
    yield_softening_start_ = ys0;
    yield_softening_final_ = ysf;
    yield_softening_min_factor_ = ymin_fac;
    // If yfloor_fac not provided (default -1), use min_factor for backward compatibility
    yield_softening_floor_factor_ = ( yfloor_fac < 0.0 ) ? ymin_fac : yfloor_fac;
    yield_softening_shape_ = yshape;
    yield_softening_relative_ = yrelative;
  }

  KOKKOS_INLINE_FUNCTION
  double softening_factor(const double a) const
  {
    // Softening disabled unless:
    //   (alpha_softening_start >= 0) AND (alpha_softening_final > alpha_softening_start)
    if ( alpha_softening_start < 0.0 ||
         alpha_softening_final <= alpha_softening_start )
      return 1.0;

    // Define: d = clamp( (a - a0) / (af - a0), 0, 1 )
    double d = ( a - alpha_softening_start ) / ( alpha_softening_final - alpha_softening_start );
    d = d < 0.0 ? 0.0 : ( d > 1.0 ? 1.0 : d );

    // g = clamp( 1 - d, gmin, 1 )
    double g = 1.0 - d;
    const double gmin = softening_gmin_;
    return g < gmin ? gmin : g;
  }

  KOKKOS_INLINE_FUNCTION
  double yield_softening_factor( const double a, const double alpha_onset = -1.0 ) const
  {
    // Disabled if ys start < 0 or ysf <= ys0 or min_factor >= 1
    if ( yield_softening_start_ < 0.0 )
      return 1.0;
    if ( yield_softening_final_ <= yield_softening_start_ )
      return 1.0;
    if ( yield_softening_min_factor_ >= 1.0 )
      return 1.0;

    const double aa = Kokkos::fabs( a );
    
    // Compute effective ys0/ysf: use relative mode if enabled and onset is valid
    double ys0_eff = yield_softening_start_;
    double ysf_eff = yield_softening_final_;
    if ( yield_softening_relative_ && alpha_onset > 0.0 )
    {
      ys0_eff = yield_softening_start_ * alpha_onset;
      ysf_eff = yield_softening_final_ * alpha_onset;
      // Fallback if computed values are invalid
      if ( ysf_eff <= ys0_eff )
      {
        ys0_eff = yield_softening_start_;
        ysf_eff = yield_softening_final_;
      }
    }
    
    double d = ( aa - ys0_eff ) / ( ysf_eff - ys0_eff );

    // clamp d to [0,1]
    d = d < 0.0 ? 0.0 : ( d > 1.0 ? 1.0 : d );

    // Apply shape function: linear (shape==0) or smoothstep (shape==1)
    if ( yield_softening_shape_ == 1 )
    {
      // Smoothstep: d = d*d*(3-2*d) for C1 continuity
      d = d * d * ( 3.0 - 2.0 * d );
    }
    // else shape==0: keep linear d as-is

    // Compute y from shaped d
    double y = 1.0 - d * ( 1.0 - yield_softening_min_factor_ );

    // enforce lower bound
    if ( y < yield_softening_min_factor_ )
      y = yield_softening_min_factor_;

    return y;
  }

  bool is_yield_softening_enabled() const
  {
    return ( yield_softening_start_ >= 0.0 &&
             yield_softening_final_ > yield_softening_start_ &&
             yield_softening_min_factor_ < 1.0 );
  }

  void updateBonds(const int num_local, const int max_neighbors)
  {
    base_plasticity_type::updateBonds(num_local, max_neighbors);
    base_hardening_type::updateBonds(num_local, max_neighbors);
    // Allocate and initialize alpha_yield_onset to -1 (invalid) for relative mode
    if ( yield_softening_relative_ )
    {
      Kokkos::realloc( _alpha_yield_onset, num_local, max_neighbors );
      Kokkos::deep_copy( _alpha_yield_onset, -1.0 );
    }
  }

  auto getAlphaView() const { return _alpha; }
  auto getSpView() const { return _s_p; }
  double getSofteningStart() const { return alpha_softening_start; }
  double getSofteningFinal() const { return alpha_softening_final; }
  double getYieldSofteningStart() const { return yield_softening_start_; }
  double getYieldSofteningFinal() const { return yield_softening_final_; }
  double getYieldSofteningMinFactor() const { return yield_softening_min_factor_; }
  double getYieldSofteningFloorFactor() const { return yield_softening_floor_factor_; }
  int getYieldSofteningShape() const { return yield_softening_shape_; }

  KOKKOS_INLINE_FUNCTION
  double sign_double(const double x) const { return (x >= 0.0) ? 1.0 : -1.0; }

  // Plastic-enabled coefficient operator with neighbor index n:
  KOKKOS_INLINE_FUNCTION
  auto operator()( ForceCoeffTag, SingleMaterial, const int i, const int j,
                   const int n, const double s, const double xi, const double vol,
                   const double m_i, const double m_j, const double theta_i,
                   const double theta_j ) const
  {
    double sp = _s_p(i,n);
    double a  = _alpha(i,n);

    double se_trial = s - sp;
    double sY = s_Y0 + H_s * a;
    
    // Track first yield for relative mode
    double alpha_onset = -1.0;
    if ( yield_softening_relative_ && _alpha_yield_onset.extent(0) > 0 )
    {
      alpha_onset = _alpha_yield_onset(i, n);
    }
    
    // Apply yield softening (no-op when disabled)
    const double yfac = yield_softening_factor( a, alpha_onset );
    sY *= yfac;
    // Enforce a floor relative to initial yield (only when yield softening is enabled)
    if ( yield_softening_start_ >= 0.0 &&
         yield_softening_final_ > yield_softening_start_ &&
         yield_softening_min_factor_ < 1.0 )
    {
      sY = Kokkos::fmax( sY, s_Y0 * yield_softening_floor_factor_ );
    }
    double f  = Kokkos::fabs(se_trial) - sY;

    if ( f > 0.0 )
    {
      // Track first yield: store current alpha when bond first yields
      if ( yield_softening_relative_ && _alpha_yield_onset.extent(0) > 0 )
      {
        if ( _alpha_yield_onset(i, n) < 0.0 )
        {
          _alpha_yield_onset(i, n) = Kokkos::fabs( a );
        }
      }
      
      double dgamma = f / ( 1.0 + H_s );
      sp += sign_double(se_trial) * dgamma;
      a  += dgamma;
      _s_p(i,n)  = sp;
      _alpha(i,n)= a;
    }

    double se = s - sp;
    auto influence_val = (*this)( influence_tag, xi );

    // Compute softening factor based on current hardening variable (use updated a)
    double g = softening_factor( a );

    // Apply softening ONLY to the shear/deviatoric term, NOT to the volumetric (theta) term
    double theta_term = theta_coeff[0] * ( theta_i / m_i + theta_j / m_j );
    double shear_term = s_coeff[0] * se * ( 1.0 / m_i + 1.0 / m_j );

    return ( theta_term + g * shear_term ) * influence_val * xi * vol;
  }

  // Energy (recoverable) with se. Do not include dissipation yet.
  KOKKOS_INLINE_FUNCTION
  auto operator()( EnergyTag, SingleMaterial, const int i, const int j,
                   const int n, const double s, const double xi, const double vol,
                   const double m_i, const double theta_i,
                   const double num_bonds ) const
  {
    double sp = _s_p(i,n);
    double se = s - sp;
    auto influence_val = (*this)( influence_tag, xi );

    // Compute softening factor based on current hardening variable
    double a = _alpha(i,n);
    double g = softening_factor( a );

    // Apply softening ONLY to the shear energy portion, NOT to the volumetric energy
    double volumetric_term = 1.0 / num_bonds * 0.5 * theta_coeff[0] / 3.0 * ( theta_i * theta_i );
    double shear_energy_term = 0.5 * ( s_coeff[0] / m_i ) * influence_val * se * se * xi * xi * vol;

    return volumetric_term + g * shear_energy_term;
  }

  // -----------------------------------------------------------------------------
  // MultiMaterial wrappers for ForceModelsMulti compatibility
  // Forward to SingleMaterial (i,j,n,...) versions, ignoring type_i/type_j
  // -----------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  auto operator()( ForceCoeffTag, MultiMaterial,
                   const int type_i, const int type_j,
                   const int i, const int j, const int n,
                   const double s, const double xi, const double vol,
                   const double m_i, const double m_j,
                   const double theta_i, const double theta_j ) const
  {
    // Forward to SingleMaterial n-version, ignoring type_i/type_j
    (void)type_i; // Suppress unused parameter warning
    (void)type_j; // Suppress unused parameter warning
    return ( *this )( ForceCoeffTag{}, SingleMaterial{}, i, j, n,
                     s, xi, vol, m_i, m_j, theta_i, theta_j );
  }

  KOKKOS_INLINE_FUNCTION
  auto operator()( EnergyTag, MultiMaterial,
                   const int type_i, const int type_j,
                   const int i, const int j, const int n,
                   const double s, const double xi, const double vol,
                   const double m_i, const double theta_i,
                   const double num_bonds ) const
  {
    // Forward to SingleMaterial n-version, ignoring type_i/type_j
    (void)type_i; // Suppress unused parameter warning
    (void)type_j; // Suppress unused parameter warning
    return ( *this )( EnergyTag{}, SingleMaterial{}, i, j, n,
                     s, xi, vol, m_i, theta_i, num_bonds );
  }
};

template <>
struct ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModelLPS<Elastic>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureIndependent>

{
    using base_type = BaseForceModelLPS<Elastic>;
    using base_fracture_type = BaseNoFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = NoFracture;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::base_type;
    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    using base_type::influence_type;
};

template <typename MemorySpace>
struct ForceModel<LPS, LPSPlastic, NoFracture, TemperatureIndependent, MemorySpace>
    : public BaseForceModelLPSPlastic<MemorySpace>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelLPSPlastic<MemorySpace>;
    using base_fracture_type = BaseNoFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = NoFracture;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::operator();
    using base_type::updateBonds;
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( LPS model, LPSPlastic mechanics, MemorySpace space,
                const double delta, const double K, const double G,
                const double sigma_y, const double H,
                const int influence = 0,
                const double a0 = -1.0, const double af = -1.0,
                const double gmin = 1e-8,
                const double ys0 = -1.0, const double ysf = -1.0,
                const double ymin_fac = 1.0,
                const double yfloor_fac = -1.0,
                const int yshape = 0,
                const bool yrelative = false )
        : base_type( model, mechanics, space, delta, K, G, sigma_y, H, influence, a0, af, gmin,
                     ys0, ysf, ymin_fac, yfloor_fac, yshape, yrelative )
    {
    }
};

template <typename MemorySpace>
struct ForceModel<LPS, LPSPlastic, Fracture, TemperatureIndependent, MemorySpace>
    : public BaseForceModelLPSPlastic<MemorySpace>,
      public BaseFractureModel,
      public BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelLPSPlastic<MemorySpace>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = Fracture;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::operator();
    using base_type::updateBonds;
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( LPS model, LPSPlastic mechanics, MemorySpace space,
                const double delta, const double K, const double G,
                const double G0, const double sigma_y, const double H,
                const int influence = 0,
                const double a0 = -1.0, const double af = -1.0,
                const double gmin = 1e-8,
                const double ys0 = -1.0, const double ysf = -1.0,
                const double ymin_fac = 1.0,
                const double yfloor_fac = -1.0,
                const int yshape = 0,
                const bool yrelative = false )
        : base_type( model, mechanics, space, delta, K, G, sigma_y, H, influence, a0, af, gmin,
                     ys0, ysf, ymin_fac, yfloor_fac, yshape, yrelative )
        , base_fracture_type(
              G0,
              ( influence == 1 ) ? Kokkos::sqrt( 5.0 * G0 / 9.0 / K / delta )
                                 : Kokkos::sqrt( 8.0 * G0 / 15.0 / K / delta ) )
        , base_temperature_type()
    {
    }
};

template <>
struct ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
    : public BaseForceModelLPS<Elastic>,
      BaseFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelLPS<Elastic>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using fracture_type = Fracture;
    using thermal_type = base_temperature_type::thermal_type;

    using base_fracture_type::bond_break_coeff;
    using base_fracture_type::G0;
    using base_fracture_type::s0;
    using base_type::base_type;
    using base_type::force_horizon;
    using base_type::influence_type;
    using base_type::K;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( LPS model, const double _force_horizon, const double _K,
                const double _G, const double _G0, const int _influence = 0 )
        : base_type( model, NoFracture{}, _force_horizon, _K, _G, _influence )
        , base_fracture_type( _force_horizon, _K, _G0, _influence )
    {
        init();
    }

    ForceModel( LPS model, Fracture, const double _force_horizon,
                const double _K, const double _G, const double _G0,
                const int _influence = 0 )
        : base_type( model, NoFracture{}, _force_horizon, _K, _G, _influence )
        , base_fracture_type( _force_horizon, _K, _G0, _influence )
    {
        init();
    }

    ForceModel( LPS model, Elastic, const double _force_horizon,
                const double _K, const double _G, const double _G0,
                const int _influence = 0 )
        : base_type( model, NoFracture{}, _force_horizon, _K, _G, _influence )
        , base_fracture_type( _force_horizon, _K, _G0, _influence )
    {
        init();
    }

    ForceModel( LPS model, Elastic elastic, Fracture,
                const double _force_horizon, const double _K, const double _G,
                const double _G0, const int _influence = 0 )
        : base_type( model, elastic, NoFracture{}, _force_horizon, _K, _G,
                     _influence )
        , base_fracture_type( _force_horizon, _K, _G0, _influence )
    {
        init();
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_fracture_type( model1, model2 )
    {
    }

    void init()
    {
        if ( influence_type == 1 )
        {
            s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / K / force_horizon ); // 1/xi
        }
        else
        {
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / K / force_horizon ); // 1
        }
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <>
struct ForceModel<LinearLPS, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>;
    // Tag to dispatch to force iteration.
    using force_tag = LinearLPS;

    template <typename... Args>
    ForceModel( LinearLPS, Args&&... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }

    using base_type::base_type;
    using base_type::operator();
};

template <>
struct ForceModel<LinearLPS, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>;

    // Tag to dispatch to force iteration.
    using force_tag = LinearLPS;

    template <typename... Args>
    ForceModel( LinearLPS, Args&&... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }

    using base_type::base_type;
    using base_type::operator();
};

template <typename MemorySpace>
struct ForceModel<LinearLPS, LPSPlastic, NoFracture, TemperatureIndependent, MemorySpace>
    : public ForceModel<LPS, LPSPlastic, NoFracture, TemperatureIndependent, MemorySpace>
{
    using base_type = ForceModel<LPS, LPSPlastic, NoFracture, TemperatureIndependent, MemorySpace>;
    using force_tag = LinearLPS;
    using base_type::operator();
    using base_type::updateBonds;

    ForceModel( LinearLPS model, LPSPlastic mechanics, MemorySpace space,
                const double delta, const double K, const double G,
                const double sigma_y, const double H,
                const int influence = 0,
                const double a0 = -1.0, const double af = -1.0 )
        : base_type( LPS{}, mechanics, space, delta, K, G, sigma_y, H, influence, a0, af )
    {
    }
};

template <typename MemorySpace>
struct ForceModel<LinearLPS, LPSPlastic, Fracture, TemperatureIndependent, MemorySpace>
    : public ForceModel<LPS, LPSPlastic, Fracture, TemperatureIndependent, MemorySpace>
{
    using base_type = ForceModel<LPS, LPSPlastic, Fracture, TemperatureIndependent, MemorySpace>;
    using force_tag = LinearLPS;
    using base_type::operator();
    using base_type::updateBonds;

    ForceModel( LinearLPS model, LPSPlastic mechanics, MemorySpace space,
                const double delta, const double K, const double G,
                const double G0, const double sigma_y, const double H,
                const int influence = 0,
                const double a0 = -1.0, const double af = -1.0 )
        : base_type( LPS{}, mechanics, space, delta, K, G, G0, sigma_y, H, influence, a0, af )
    {
    }
};

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, const double force_horizon,
            const double K, const double G, const int influence = 0 )
    -> ForceModel<ModelType, Elastic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, const double force_horizon, const double K,
            const double G, const int influence = 0 )
    -> ForceModel<ModelType, Elastic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, Elastic, const double force_horizon, const double K,
            const double G, const double _G0, const int influence = 0,
            typename std::enable_if<( is_state_based<ModelType>::value ),
                                    int>::type* = 0 )
    -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel(
    ModelType, Elastic, Fracture, const double force_horizon, const double K,
    const double G, const double _G0, const int influence = 0,
    typename std::enable_if<( is_state_based<ModelType>::value ), int>::type* =
        0 ) -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel( ModelType, const double _force_horizon, const double _K,
            const double _G, const double _G0, const int _influence = 0,
            typename std::enable_if<( is_state_based<ModelType>::value ),
                                    int>::type* = 0 ) -> ForceModel<ModelType>;

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, LPSPlastic, NoFracture, MemorySpace,
            const double delta, const double K, const double G,
            const double sigma_y, const double H,
            const int influence = 0 )
    -> ForceModel<ModelType, LPSPlastic, NoFracture, TemperatureIndependent, MemorySpace>;

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, LPSPlastic, Fracture, MemorySpace,
            const double delta, const double K, const double G,
            const double G0, const double sigma_y, const double H,
            const int influence = 0 )
    -> ForceModel<ModelType, LPSPlastic, Fracture, TemperatureIndependent, MemorySpace>;

} // namespace CabanaPD

#endif
