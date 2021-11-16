#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD_Output.hpp>
#include <CabanaPD_Solver.hpp>

int main( int argc, char *argv[] )
{

    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        CabanaPD::Inputs inputs;
        inputs.read_args( argc, argv );

        auto cabana_pd = CabanaPD::createSolver( inputs );
        cabana_pd->run();
    }

    MPI_Finalize();
}
