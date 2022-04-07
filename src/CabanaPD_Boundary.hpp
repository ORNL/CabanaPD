#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

namespace CabanaPD
{
namespace BoundaryCondition
{

struct Empty
{
    void operator()( const int ){};
};
/*
template <int SurfaceDim, int WidthDim>
struct ForceResetWidth
{
    double prenotch_low;
    double prenotch_high;
    double surface;

    void operator()( const int pid )
    {
        if ( x( pid, WidthDim ) > prenotch_low &&
             x( pid, WidthDim ) < prenotch_high &&
             x( pid, SurfaceDim ) > surface )
            for ( int d = 0; d < 3; d++ )
                f( pid, d ) = 0.0;
    };
};
*/
template <class ExecSpace, class ParticleType>   //, class FunctorType>
void apply( ExecSpace, ParticleType& particles ) //, FunctorType functor )
{
    // FIXME: this is all hardcoded
    double y_prenotch1 = -0.025; // [m] (-25 mm)
    double y_prenotch2 = 0.025;  // [m] ( 25 mm)
    double height = 0.1;         // [m] (100 mm)
    double dx = particles.dx;

    auto x = particles.slice_x();
    auto f = particles.slice_f();
    Kokkos::RangePolicy<ExecSpace> policy( 0, particles.n_local );
    Kokkos::parallel_for(
        "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int pid ) {
            // functor( pid );
            if ( x( pid, 1 ) > y_prenotch1 && x( pid, 1 ) < y_prenotch2 &&
                 x( pid, 0 ) < -0.5 * height + dx )
                for ( int d = 0; d < 3; d++ )
                    f( pid, d ) = 0.0;
        } );
}

} // namespace BoundaryCondition
} // namespace CabanaPD

#endif
