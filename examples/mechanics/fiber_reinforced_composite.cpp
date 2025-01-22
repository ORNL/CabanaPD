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

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

#include <CabanaPD_Constants.hpp>

// Generate a unidirectional fiber-reinforced composite geometry
void fiberReinforcedCompositeExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 1.0 / 3.0;
    double K = E / ( 3.0 * ( 1.0 - 2.0 * nu ) );
    double G0 = inputs["fracture_energy"];
    // double G = E / ( 2.0 * ( 1.0 + nu ) ); // Only for LPS.
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type1 = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_type1 force_model1( delta, K, G0 );
    using model_type2 = CabanaPD::ForceModel<CabanaPD::LinearPMB>;
    model_type2 force_model2( delta, K / 10.0, G0 / 10.0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles = CabanaPD::createParticles<memory_space, model_type2>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================

    std::array<double, 3> system_size = inputs["system_size"];

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto type = particles->sliceType();

    // Fiber-reinforced composite geometry parameters
    double Vf = inputs["fiber_volume_fraction"];
    double Df = inputs["fiber_diameter"];
    std::vector<double> stacking_sequence = inputs["stacking_sequence"];

    // Fiber radius
    double Rf = 0.5 * Df;

    // System sizes
    double Lx = system_size[0];
    double Ly = system_size[1];
    double Lz = system_size[2];

    // Number of plies
    auto Nplies = stacking_sequence.size();
    // Ply thickness (in z-direction)
    double dzply = Lz / Nplies;

    // Single-fiber volume (assume a 0° fiber orientation)
    double Vfs = CabanaPD::pi * Rf * Rf * Lx;
    // Domain volume
    double Vd = Lx * Ly * Lz;
    // Total fiber volume
    double Vftotal = Vf * Vd;
    // Total number of fibers
    int Nf = std::floor( Vftotal / Vfs );
    // Cross section corresponding to a single fiber in the YZ-plane
    // (assume all plies have 0° fiber orientation)
    double Af = Ly * Lz / Nf;
    // Number of fibers in y-direction (assume Af is a square area)
    int Nfy = std::round( Ly / std::sqrt( Af ) );
    // Ensure Nfy is even
    if ( Nfy % 2 == 1 )
        Nfy = Nfy + 1;

    // Number of fibers in z-direction
    int Nfz = std::round( Nf / Nfy );
    // Ensure number of fibers in z-direction within each ply is even
    int nfz = std::round( Nfz / Nplies );
    if ( nfz % 2 == 0 )
    {
        Nfz = nfz * Nplies;
    }
    else
    {
        Nfz = ( nfz + 1 ) * Nplies;
    };

    // Fiber grid spacings (assume all plies have 0° fiber orientation)
    double dyf = Ly / Nfy;
    double dzf = Lz / Nfz;

    // Domain center
    double Xc = 0.5 * ( low_corner[0] + high_corner[0] );
    double Yc = 0.5 * ( low_corner[1] + high_corner[1] );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Particle position
        double xi = x( pid, 0 );
        double yi = x( pid, 1 );
        double zi = x( pid, 2 );

        // Find ply number of particle (counting from 0)
        double nply = std::floor( ( zi - low_corner[2] ) / dzply );

        // Ply fiber orientation (in radians)
        double theta = stacking_sequence[nply] * CabanaPD::pi / 180;

        // Translate then rotate y-coordinate of particle in XY-plane
        double yinew =
            -std::sin( theta ) * ( xi - Xc ) + std::cos( theta ) * ( yi - Yc );

        // Find center of ply in z-direction (recall first ply has nply = 0)
        double Zply_bot = low_corner[2] + nply * dzply;
        double Zply_top = Zply_bot + dzply;
        double Zcply = 0.5 * ( Zply_bot + Zply_top );

        // Translate point in z-direction
        double zinew = zi - Zcply;

        // Find nearest fiber grid center point in YZ plane
        double Iyf = std::floor( yinew / dyf );
        double Izf = std::floor( zinew / dzf );
        double YI = 0.5 * dyf + dyf * Iyf;
        double ZI = 0.5 * dzf + dzf * Izf;

        // Check if point belongs to fiber
        if ( ( yinew - YI ) * ( yinew - YI ) + ( zinew - ZI ) * ( zinew - ZI ) <
             Rf * Rf + 1e-8 )
            type( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        *particles, CabanaPD::AverageTag{}, force_model1, force_model2 );
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, models );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init();
    cabana_pd->run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fiberReinforcedCompositeExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
