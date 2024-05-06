/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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

void thermalDeformationExample( const std::string filename )
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    CabanaPD::Inputs inputs( filename );
    double E = inputs["elastic_modulus"];
    double rho0 = inputs["density"];
    double nu = 0.25; // unitless
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double delta = inputs["horizon"];

    double alpha = inputs["thermal_coefficient"];
    double temp0 = inputs["reference_temperature"];

    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // Choose force model type.
    using model_type = CabanaPD::PMB;

    // Create particles from mesh.
    auto particles =
        std::make_shared<CabanaPD::Particles<memory_space, model_type>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

    auto temp = particles->sliceTemperature();
    auto x = particles->sliceReferencePosition();
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        temp( pid ) = 5000.0 * ( x( pid, 1 ) - ( -0.15 ) ) * t;
    };
    auto body_term = CabanaPD::createBodyTerm( temp_func );

    auto rho = particles->sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { rho( pid ) = rho0; };
    particles->updateParticles( exec_space{}, init_functor );

    auto force_model =
        CabanaPD::createForceModel<model_type, CabanaPD::Elastic,
                                   CabanaPD::TemperatureDependent>(
            *particles, delta, K, alpha, temp0 );
    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, force_model, body_term );
    cabana_pd->init_force();
    cabana_pd->run();
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
