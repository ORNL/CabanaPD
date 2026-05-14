#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

constexpr std::size_t NUM_GRAINS = 4;
constexpr double PI = 3.141592653589793238462643383;

// Get flat index into ND array
template <std::size_t n>
int indexND( const std::array<int, n>& index, const std::array<int, n>& shape )
{
    int outIndex = 0;
    int stride = 1;

    for ( int axis = n - 1; axis >= 0; --axis )
    {
        outIndex += index[axis] * stride;
        stride *= shape[axis];
    }

    return outIndex;
}

// Check if ND array multi-index is valid
template <std::size_t n>
bool isValid( const std::array<int, n>& idx, const std::array<int, n>& shape )
{
    for ( int i = 0; i < n; ++i )
    {
        if ( idx[i] < 0 || idx[i] >= shape[i] )
        {
            return false;
        }
    }
    return true;
}

// Recursive helper function for makeNeighborRelativeIndices
template <std::size_t n>
void makeRelativeIndicesRecursive(
    int axis, std::array<int, n>& curIndex,
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    int maxDist = std::ceil( std::sqrt( static_cast<double>( n ) ) );
    for ( int i = -maxDist; i <= maxDist; ++i )
    {
        curIndex[axis] = i;
        if ( axis < n - 1 )
        {
            makeRelativeIndicesRecursive( axis + 1, curIndex,
                                          outRelativeIndices );
        }
        else
        {
            outRelativeIndices.push_back( curIndex );
        }
    }
}

// Get ND array neighbor relative multi-indices
template <std::size_t n>
void makeNeighborRelativeIndices(
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    std::array<int, n> curIndex = {};
    makeRelativeIndicesRecursive( 0, curIndex, outRelativeIndices );
}

template <std::size_t n>
double distSquared( const std::array<double, n>& a,
                    const std::array<double, n>& b )
{
    double r2 = 0.0;
    for ( int axis = 0; axis < n; ++axis )
    {
        r2 += ( a[axis] - b[axis] ) * ( a[axis] - b[axis] );
    }
    return r2;
}

// n-dimensional Poisson disc sampling in a rectangular prism domain
template <std::size_t n, class RNGType>
void poissonDiscSampling( const std::array<double, n>& extent, double r, int k,
                          std::vector<std::array<double, n>>& outPoints,
                          RNGType& gen )
{
    // Precompute reused values for sampling
    double rn = std::pow( r, static_cast<double>( n ) );
    double rn2 = std::pow( 2.0 * r, static_cast<double>( n ) );

    // Calculate shape of grid
    double cellSize = r / std::sqrt( static_cast<double>( n ) );
    std::array<int, n> gridShape = {};
    int totalCells = 1;
    for ( int axis = 0; axis < n; ++axis )
    {
        double axisCells = std::ceil( extent[axis] / cellSize );

        // Add an extra cell in each direction just in case
        gridShape[axis] = 1 + static_cast<int>( axisCells );
        totalCells *= gridShape[axis];
    }

    // Initialize empty grid (as flat array)
    std::vector<int> grid( totalCells, -1 );

    // Calculate grid search relative indices
    std::vector<std::array<int, n>> nbrIndicesRel;
    makeNeighborRelativeIndices( nbrIndicesRel );

    // Choose first point
    std::uniform_real_distribution<double> coordDist( 0.0, 1.0 );
    auto coordGen = std::bind( coordDist, gen );

    std::array<double, n> x0 = {};
    std::array<int, n> idx0 = {};
    for ( int axis = 0; axis < n; ++axis )
    {
        x0[axis] = coordGen() * extent[axis];
        idx0[axis] = static_cast<int>( std::floor( x0[axis] / cellSize ) );
    };

    grid[indexND( idx0, gridShape )] = 0; // Store x0 location in grid
    outPoints.push_back( x0 );

    // Initialize active set
    std::vector<int> activeSet = { 0 };

    while ( activeSet.size() > 0 )
    {
        std::uniform_int_distribution<int> pointDist( 0, activeSet.size() - 1 );
        int seedIndexInActive = pointDist( gen );
        int seedIndex = activeSet[seedIndexInActive];
        std::array<double, n> seedX = outPoints[seedIndex];

        bool addedPoint = false;

        for ( int i = 0; i < k; ++i )
        {
            std::array<double, n> x = {};
            std::array<int, n> idx = {};

            // Use inverse method to sample distance from seed
            double xR = std::pow( rn + coordGen() * ( rn2 - rn ),
                                  1.0 / static_cast<double>( n ) );

            // Sample direction cosine angles uniformly in [0, pi] to get
            // direction
            bool failed = false;
            for ( int axis = 0; axis < n; ++axis )
            {
                x[axis] = seedX[axis] + xR * std::cos( coordGen() * PI );
                idx[axis] =
                    static_cast<int>( std::floor( x[axis] / cellSize ) );
                if ( x[axis] < 0 || x[axis] >= extent[axis] )
                {
                    failed = true;
                    break;
                }
            }
            if ( failed )
            {
                continue;
            }

            // Check if point is too close to any existing points
            std::array<int, n> checkIdx = idx;
            bool isClose = false;

            for ( const std::array<int, n>& relIdx : nbrIndicesRel )
            {
                for ( int axis = 0; axis < n; ++axis )
                {
                    checkIdx[axis] = idx[axis] + relIdx[axis];
                }

                if ( !isValid( checkIdx, gridShape ) )
                {
                    continue;
                }

                int checkPoint = grid[indexND( checkIdx, gridShape )];
                if ( checkPoint == -1 )
                {
                    continue;
                }

                const std::array<double, n>& checkX = outPoints[checkPoint];

                if ( distSquared( checkX, x ) > r * r )
                {
                    continue;
                }

                isClose = true;
                break;
            }

            // New point is not too close to any existing, so add it
            if ( !isClose )
            {
                outPoints.push_back( x );
                activeSet.push_back( outPoints.size() - 1 );
                grid[indexND( idx, gridShape )] = outPoints.size() - 1;
                addedPoint = true;
                break;
            }
        }
        // If no point could be generated farther than r from existing points,
        // remove seed point from active set
        if ( !addedPoint )
        {
            activeSet.erase( activeSet.begin() + seedIndexInActive );
        }
    }
}

// Generates numGrains random polycrystal grain centers using
// Poisson disc sampling
template <std::size_t numGrains>
void getPolycrystalGrains(
    const std::array<double, 3>& extent,
    std::array<std::array<double, 3>, numGrains>& outLocations )
{
    // Initialize RNG
    std::random_device trueRng;
    std::seed_seq randomSeed{ trueRng(), trueRng(), trueRng(), trueRng(),
                              trueRng(), trueRng(), trueRng(), trueRng() };
    std::mt19937 gen( randomSeed );

    // Generate random, evenly-spaced candidate grain locations
    double volume = extent[0] * extent[1] * extent[2];
    double radius =
        2.0 * std::pow( 0.75 * volume / PI / static_cast<double>( numGrains ),
                        1.0 / 3.0 );
    std::vector<std::array<double, 3>> testPoints;
    do
    {
        testPoints.clear();
        poissonDiscSampling( extent, radius, 30, testPoints, gen );
        radius *= 0.95;
    } while ( testPoints.size() < numGrains );

    // Randomly choose grains locations from candidates
    std::vector<std::size_t> indices( testPoints.size(), 0 );
    std::iota( indices.begin(), indices.end(), 0 );
    std::shuffle( indices.begin(), indices.end(), gen );
    for ( std::size_t i = 0; i < numGrains; ++i )
    {
        outLocations[i] = testPoints[indices[i]];
    }
}

// Simulate a crack in a polycrystal
void crackPolycrystalExample( const std::string filename )
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
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = {
        inputs["low_corner"][0];
        inputs["low_corner"][1];
        inputs["low_corner"][2]
    };
    std::array<double, 3> high_corner = {
        inputs["high_corner"][0],
        inputs["high_corner"][1],
        inputs["high_corner"][2]
    };

    // ====================================================
    //                Material parameters
    // ====================================================
    std::array<double, NUM_GRAINS> grainRho;
    std::array<double, NUM_GRAINS> E;
    std::array<double, NUM_GRAINS> nu;
    std::array<double, NUM_GRAINS> G0;
    std::array<double, NUM_GRAINS> K;
    std::array<double, NUM_GRAINS> G;

    for ( int i = 0; i < NUM_GRAINS; ++i )
    {
        grainRho[i] = inputs["density"][i];
        E[i] = inputs["elastic_modulus"][i];
        nu[i] = inputs["Poisson's_ratio"][i];
        G0[i] = inputs["fracture_energy"][i];
        K[i] = E[i] / ( 3 * ( 1 - 2 * nu[i] ) );
        G[i] = E[i] / ( 2 * ( 1 + nu[i] ) );
    }

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                Polycrystal grains
    // ====================================================
    std::array<double, 3> extent = inputs["system_size"];
    std::array<std::array<double, 3>, NUM_GRAINS> grainPos;
    getPolycrystalGrains( extent, grainPos );

    // Shift grains relative to low_corner
    std::cout << "Generated " << NUM_GRAINS << " polycrystal grains" << std::endl;
    for ( int i = 0; i < NUM_GRAINS; ++i )
    {
        grainPos[i] = { grainPos[i][0] + low_corner[0],
                        grainPos[i][1] + low_corner[1],
                        grainPos[i][2] + low_corner[2] };
        std::cout << grainPos[i][0] << ", " << grainPos[i][1] << ", " grainPos[i][2] << std::endl;
    }

    // ====================================================
    //                    Pre-notch
    // ====================================================
    double height = inputs["system_size"][0];
    double thickness = inputs["system_size"][2];
    double L_prenotch = height / 2.0;
    double y_prenotch = 0.5 * ( low_corner[1] + high_corner[1] );
    Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch, low_corner[2] };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
    CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::LPS;

    // Grain materials
    CabanaPD::ForceModel force_models0( model_type{}, horizon, K[0], G[0],
                                        G0[0] );
    CabanaPD::ForceModel force_models1( model_type{}, horizon, K[1], G[1],
                                        G0[1] );
    CabanaPD::ForceModel force_models2( model_type{}, horizon, K[2], G[2],
                                        G0[2] );
    CabanaPD::ForceModel force_models3( model_type{}, horizon, K[3], G[3],
                                        G0[3] );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon + 1e-10 ||
             x( pid, 1 ) >= plane2.high[1] - horizon - 1e-10 )
            nofail( pid ) = 1;

        // Distance squared from nearest grain location
        double distSq = 0.0;
        int grainIndex = 0;
        for ( int i = 0; i < NUM_GRAINS; ++i )
        {
            const std::array<double, 3>& pos = grainPos[i];
            double dx = x( pid, 0 ) - pos[0];
            double dy = x( pid, 1 ) - pos[1];
            double dz = x( pid, 2 ) - pos[2];
            double check = dx * dx + dy * dy + dz * dz;
            if ( i == 0 || check < distSq )
            {
                distSq = check;
                grainIndex = i;
            }
        }

        // Density and material type
        type( pid ) = grainIndex;
        rho( pid ) = grainRho[grainIndex];
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, force_models0, force_models1,
        force_models2, force_models3 );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dy;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto ypos = x( pid, 1 );
        auto sign = std::abs( ypos ) / ypos;
        f( pid, 1 ) += b0 * sign;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, plane1, plane2 );

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output maximum y-extent of the crack.
    CabanaPD::Region<CabanaPD::RectangularPrism> box( low_corner, high_corner );
    auto d = solver.particles.sliceDamage();
    auto crack_y_func = KOKKOS_LAMBDA( const int p )
    {
        // Use a threshold of damage to only output damaged particles.
        if ( d( p ) > 0.3 )
            return x( p, 1 );
        else
            return 0.0;
    };
    auto output_yl = CabanaPD::createOutputTimeSeries<Kokkos::Max<double>>(
        "output_polycrystal_crack_y.txt", inputs, exec_space{},
        solver.particles, crack_y_func, box );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc, prenotch );
    solver.run( bc, output_yl );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    crackInclusionExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
