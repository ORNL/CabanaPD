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

// Simulate crack branching from an pre-crack.
void sphericalIndenterExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
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
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //                    Pre-notch
    // ====================================================
/*	
    double height = inputs["system_size"][0];
    double thickness = inputs["system_size"][2];
    double L_prenotch = height / 2.0;
    double y_prenotch = 0.0;
    Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch, low_corner[2] };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
    CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );
*/
    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model( model_type{}, horizon, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    CabanaPD::Particles particles( memory_space{}, model_type{} );

    // Note that individual inputs can be passed instead (see other examples).
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================

    double dz = particles.dx[2];
    CabanaPD::Region<CabanaPD::RectangularPrism> square_pressure(
        0.5 * low_corner[0], 0.5 * high_corner[0], 0.5 * low_corner[1],
        0.5 * high_corner[1], high_corner[2] - dz, high_corner[2] + dz );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();


    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
		
        // Density
        rho( pid ) = rho0;
/*		
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon + 1e-10 ||
             x( pid, 1 ) >= plane2.high[1] - horizon - 1e-10 )
            nofail( pid ) = 1;
*/
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = -sigma0 / dz;
	auto indent_force = 0;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
	v = solver.particles.sliceVelocity();
    // Create a symmetric force BC in the z-direction.
	
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {		
	  double xsq = x(pid,0) * x(pid,0);
	  double ysq = x(pid,1) * x(pid,1);
	  double v0 = 50.0; // m/s
	  double R = 3e-3; //max indenter radius
	  double t = 1e-4;
	  double r = std::pow(v0*t * (2.0*R - v0*t), 0.5);
	  
	    
	        if ( std::pow(xsq + ysq, 0.5) < r )
	        {
//	          f( pid, 2 ) += b0 * R*R - (xsq + ysq) / R*R;
             f( pid, 2 ) += b0 * std::pow(R*R - (xsq + ysq), 0.5) / R;
	        }  	 

		
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, square_pressure );
									   
//									   
//========================================
//            OUTPUTS
//========================================
//									   

auto dx = solver.particles.dx[0];
auto dy = solver.particles.dx[1];
//auto dz = solver.particles.dx[2];
//auto f = solver.particles.sliceForce();

auto force_func_x = KOKKOS_LAMBDA( const int p )
{
    return f( p, 0 ) * dx * dy * dz;
};
auto output_fx = CabanaPD::createOutputTimeSeries(
    "output_force_x.txt", inputs, exec_space{}, solver.particles,
    force_func_x, square_pressure );

// Output force on right grip in y-direction.
auto force_func_y = KOKKOS_LAMBDA( const int p )
{
    return f( p, 1 ) * dx * dy * dz;
};
auto output_fy = CabanaPD::createOutputTimeSeries(
    "output_force_y.txt", inputs, exec_space{}, solver.particles,
    force_func_y, square_pressure );

// Output force on right grip in z-direction.
auto force_func_z = KOKKOS_LAMBDA( const int p )
{
    return f( p, 2 ) * dx * dy * dz;
};
auto output_fz = CabanaPD::createOutputTimeSeries(
    "output_force_z.txt", inputs, exec_space{}, solver.particles,
    force_func_z, square_pressure );


    // ====================================================
    //                   Simulation run
    // ====================================================
    //solver.init( bc, prenotch );
	solver.init( bc );
    solver.run( bc, output_fx, output_fy, output_fz);
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    sphericalIndenterExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
