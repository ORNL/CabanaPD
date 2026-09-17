/**
 * Helper functions to support the polycrystal example by generating
 * Voronoi grain structures and providing a constant-time lookup
 * algorithm used to assign computational particles to grains.
 *
 * Grain nucleation site generation uses Bridson's algorithm [1]
 * for Poisson disc sampling. Comments also reference custom
 * documentation describing the mathematics in greater detail.
 *
 * References
 *   [1]: Robert Bridson. Fast Poisson disk sampling in arbitrary
 *          dimensions. In ACM SIGGRAPH 2007 Sketches, page 22,
 *          San Diego California, August, 2007. ACM.
 *
 */

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include <Kokkos_Core.hpp>

// Get flat index into n-dim array (equations (7) and (8))
//
// Parameters
// ----------
// index: multi-index into an n-dimensional array
// shape: number of entries along each axis of the array (ceil(E_i/c))
//
// Return
// ------
// outIndex: index into underlying flat array
template <std::size_t n>
int indexND( const std::array<int, n>& index, const std::array<int, n>& shape )
{
    int outIndex = 0;
    int stride = 1;

    // Calculate stride and contribution to index in the same loop
    for ( int axis = n - 1; axis >= 0; --axis )
    {
        outIndex += index[axis] * stride;
        stride *= shape[axis];
    }

    return outIndex;
}

// Check if N-dim array multi-index is valid
//
// Parameters
// ----------
// idx: a multi-index
// shape: number of entries along each axis of the array (ceil(E_i/c))
//
// Return
// ------
// isValid: whether the given multi-index belongs to the set A of grid cell
//          multi-indices that overlap the domain
template <std::size_t n>
bool isValid( const std::array<int, n>& idx, const std::array<int, n>& shape )
{
    for ( std::size_t i = 0; i < n; ++i )
    {
        if ( idx[i] < 0 || idx[i] >= shape[i] )
        {
            return false;
        }
    }
    return true;
}

// Recursive function to generate the set of *relative* multi-indices of
// grid cells that may contain points within a distance r of a generated
// point in C (see equation (12)). Each recursion step handles one axis.
//
// Parameters
// ----------
// n: dimension of the domain (and depth of the recursion)
// axis: which axis of the recursion we are on
// curIndex: current starting multi-index in the axes that have already been
//           generated
//
// Return
// ------
// outRelativeIndices: list of relative multi-indices of cells that can
//                     contain points within a distance r
template <std::size_t n>
void makeRelativeIndicesRecursive(
    std::size_t axis, std::array<int, n>& curIndex,
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    // Get maximum distance
    int maxDist = std::ceil( std::sqrt( static_cast<double>( n ) ) );

    // Go over all indices along the current axis within maximum distance
    for ( int i = -maxDist; i <= maxDist; ++i )
    {
        curIndex[axis] = i;
        // If there are axes left, go to next axis
        if ( axis < n - 1 )
        {
            makeRelativeIndicesRecursive( axis + 1, curIndex,
                                          outRelativeIndices );
        }
        // Otherwise, all axes have been set and we output the current
        // relative multi-index
        else
        {
            outRelativeIndices.push_back( curIndex );
        }
    }
}

// Generate the set of *relative* multi-indices of grid cells that may
// contain points within a distance r of a generated point in C
// (see equation (12)).
//
// Parameters
// ----------
// n: dimension of the domain
//
// Return
// ------
// outRelativeIndices: list of relative multi-indices of cells that can
//                     contain points within a distance r
template <std::size_t n>
void makeNeighborRelativeIndices(
    std::vector<std::array<int, n>>& outRelativeIndices )
{
    // Start the recursion
    std::array<int, n> curIndex = {}; // = (0, 0, ..., 0)
    makeRelativeIndicesRecursive( 0, curIndex, outRelativeIndices );
}

// Compute distance squared between two points
//
// Parameters
// ----------
// a: first point
// b: second point
//
// Return
// ------
// distSquared: distance squared between a and b
template <std::size_t n>
double distSquared( const std::array<double, n>& a,
                    const std::array<double, n>& b )
{
    double r2 = 0.0;
    for ( std::size_t axis = 0; axis < n; ++axis )
    {
        r2 += ( a[axis] - b[axis] ) * ( a[axis] - b[axis] );
    }
    return r2;
}

// n-dimensional Poisson disc sampling in a rectangular prism domain
//
// Parameters
// ----------
// n: dimension of the domain
// extent: Extents of the domain (E_1, E_2, ..., E_n)
// r: minimum distance between sampled points
// k: maximum number of attempts to generate new points
// gen: STL random number generator
//
//
// Return
// ------
// outPoints: list of sampled points satisfying PD1, PD2, and PD3
//            (approximately)
// outGridShape: size of the lookup grid in each dimension
//               (ceil(E_i/c))
// outCellSize: size c of cells in lookup grid
template <std::size_t n, class RNGType>
void poissonDiscSampling( const std::array<double, n>& extent, double r, int k,
                          std::vector<std::array<double, n>>& outPoints,
                          std::array<int, n>& outGridShape, double& outCellSize,
                          RNGType& gen )
{
    // Precompute reused values for sampling (equation (9))
    double rn = std::pow( r, static_cast<double>( n ) );
    double rn2 = std::pow( 2.0 * r, static_cast<double>( n ) );

    // Calculate shape of grid (i.e., c = cellSize and ceil(E_i/c))
    double cellSize = r / std::sqrt( static_cast<double>( n ) );
    std::array<int, n> gridShape = {};
    int totalCells = 1;
    for ( std::size_t axis = 0; axis < n; ++axis )
    {
        double axisCells = std::ceil( extent[axis] / cellSize );

        // Add an extra cell in each direction just in case
        gridShape[axis] = 1 + static_cast<int>( axisCells );
        totalCells *= gridShape[axis];
    }

    // Initialize empty grid (as flat array)
    std::vector<int> grid( totalCells, -1 );

    // Calculate grid search relative indices (see equation (12))
    std::vector<std::array<int, n>> nbrIndicesRel;
    makeNeighborRelativeIndices( nbrIndicesRel );

    // Create random number generator to get uniform numbers on [0, 1]
    std::uniform_real_distribution<double> coordDist( 0.0, 1.0 );
    auto coordGen = std::bind( coordDist, gen );

    // And a standard normal random number generator
    std::normal_distribution<double> normalDist;
    auto normalGen = std::bind( normalDist, gen );

    // Generate first point x0 by sampling uniformly on the domain
    std::array<double, n> x0 = {};
    std::array<int, n> idx0 = {};
    for ( std::size_t axis = 0; axis < n; ++axis )
    {
        x0[axis] = coordGen() * extent[axis];
        idx0[axis] = static_cast<int>( std::floor( x0[axis] / cellSize ) );
    };

    // Store x0 location in grid and add to output list
    grid[indexND( idx0, gridShape )] = 0;
    outPoints.push_back( x0 );

    // Initialize active set
    std::vector<int> activeSet = { 0 };

    // Main sampling loop
    while ( activeSet.size() > 0 )
    {
        // Create uniform distribution on active set indices to sample a
        // random point from the active set
        std::uniform_int_distribution<int> pointDist( 0, activeSet.size() - 1 );
        int seedIndexInActive = pointDist( gen );

        // Select a random seed point x_j from the active set
        int seedIndex = activeSet[seedIndexInActive];
        std::array<double, n> seedX = outPoints[seedIndex];

        // Whether a new point was added near the
        // seed (see line 15 of Algorithm 1)
        bool addedPoint = false;

        // Run k attempts to generate a new point
        for ( int i = 0; i < k; ++i )
        {
            // Initialize storage for test point x and its grid multi-index
            // alpha = idx.
            std::array<double, n> x = {};
            std::array<int, n> idx = {};

            // Use inverse CDF method to sample distance from seed (equation
            // (9))
            double xR = std::pow( rn + coordGen() * ( rn2 - rn ),
                                  1.0 / static_cast<double>( n ) );

            // Sample direction uniformly on unit sphere to get x
            // (equations (10) and (11))

            // First generate standard normal point (z, but we can
            // store temporarily in x)
            double normXSquared = 0.0;
            for ( std::size_t axis = 0; axis < n; ++axis )
            {
                x[axis] = normalGen();
                normXSquared += x[axis] * x[axis];
            }

            // Normalize and calculate final x
            double normX = std::sqrt( normXSquared );
            bool failed = false;
            for ( std::size_t axis = 0; axis < n; ++axis )
            {
                x[axis] = seedX[axis] + xR * x[axis] / normX;
                idx[axis] =
                    static_cast<int>( std::floor( x[axis] / cellSize ) );

                // We can terminate early if the current component
                // is out of the domain
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
            // (line 9 in Algorithm 1)
            std::array<int, n> checkIdx = idx;

            // This flag is set if point in C too close to x is found
            bool isClose = false;

            // Loop over precomputed relative indices
            for ( const std::array<int, n>& relIdx : nbrIndicesRel )
            {
                // Shift relative index to current test point's grid cell
                for ( std::size_t axis = 0; axis < n; ++axis )
                {
                    checkIdx[axis] = idx[axis] + relIdx[axis];
                }

                // Skip the current check cell (leave isClose = false) if
                // it is outside of the domain
                if ( !isValid( checkIdx, gridShape ) )
                {
                    continue;
                }

                // Look up which point, if any, is in the current check cell
                int checkPoint = grid[indexND( checkIdx, gridShape )];
                if ( checkPoint == -1 )
                {
                    // No point in this cell, so leave isClose = false
                    continue;
                }

                // We found an existing point in this check cell, so
                // now check precisely if it is too close to x
                const std::array<double, n>& checkX = outPoints[checkPoint];

                if ( distSquared( checkX, x ) > r * r )
                {
                    continue;
                }

                isClose = true;
                break;
            }

            // If new point is not too close to any existing, add it
            // (append to output list, add to active set, add to grid)
            if ( !isClose )
            {
                outPoints.push_back( x );
                activeSet.push_back( outPoints.size() - 1 );
                grid[indexND( idx, gridShape )] = outPoints.size() - 1;

                // Mark the added point flag and terminate the loop to
                // try adding a point k times
                addedPoint = true;
                break;
            }
        }

        // If no point could be generated (in k tries) farther than r
        // from existing points, then remove seed point from active set
        if ( !addedPoint )
        {
            activeSet.erase( activeSet.begin() + seedIndexInActive );
        }
    }
    // Loop terminates when the domain is mostly filled (PD2 almost satisfied)
    // and output points are stored in outPoints

    // Set grid shape and cell size return values
    outGridShape = gridShape;
    outCellSize = cellSize;
}

// Generates random, uniformly-sized polycrystal grains
// using Poisson disc sampling
//
// Parameters
// ----------
// extent: size of the rectangular domain (E_1, E_2, E_3),
//         bottom left is always (0, 0, 0)
// grainSize: typical *radius* of grains
//
// Return
// ------
// outLocations: list of grain center locations
// outGridShape: size of lookup grid used internally and by
//               FindClosestGrainFunctor
// outCellSize: size of lookup grid cells used internally and by
//              FindClosestGrainFunctor
void getPolycrystalGrains( const std::array<double, 3>& extent,
                           double grainSize,
                           std::vector<std::array<double, 3>>& outLocations,
                           std::array<int, 3>& outGridShape,
                           double& outCellSize )
{
    // Initialize RNG. We want to use Mersenne twister, but we need to
    // seed it using a simpler RNG first.
    std::minstd_rand baseRng;
    baseRng.seed( 12345 );

    // Get Mersenne twister seed from baseRng
    std::seed_seq randomSeed{ baseRng(), baseRng(), baseRng(), baseRng(),
                              baseRng(), baseRng(), baseRng(), baseRng() };
    std::mt19937 gen( randomSeed );

    // Generate random, evenly-spaced candidate grain locations with k = 30
    poissonDiscSampling( extent, grainSize, 30, outLocations, outGridShape,
                         outCellSize, gen );
}

// Functor that looks up the nearest grain center to a given
// query location
template <typename GrainGridType, typename GrainPosType>
struct FindClosestGrainFunctor
{
    // Kokkos view storing the lookup grid,
    // size = (E_1, E_2, E_3) = (sizeX, sizeY, sizeZ)
    GrainGridType grainGrid;

    // Kokkos view storing the list of physical grain locations
    // size = (numGrains, 3)
    GrainPosType grainPos;

    // Coordinates of physical domain minimum corner
    double minX;
    double minY;
    double minZ;

    // Size of lookup grid in each direction
    int sizeX;
    int sizeY;
    int sizeZ;

    // Physical size of lookup grid cells
    double grainGridCellSize;

    FindClosestGrainFunctor( GrainGridType grainGrid, GrainPosType grainPos,
                             const std::array<double, 3>& low_corner,
                             const std::array<int, 3>& grainGridShape,
                             double grainGridCellSize )
        : grainGrid( grainGrid )
        , grainPos( grainPos )
        , minX( low_corner[0] )
        , minY( low_corner[1] )
        , minZ( low_corner[2] )
        , sizeX( grainGridShape[0] )
        , sizeY( grainGridShape[1] )
        , sizeZ( grainGridShape[2] )
        , grainGridCellSize( grainGridCellSize )
    {
    }

    // Get index of nearest point in grainPos to (x, y, z)
    KOKKOS_FUNCTION int operator()( double x, double y, double z ) const
    {
        // Get grid cell multi-index of (x, y, z) (see equation (3))
        int gx = static_cast<int>(
            Kokkos::floor( ( x - minX ) / grainGridCellSize ) );
        int gy = static_cast<int>(
            Kokkos::floor( ( y - minY ) / grainGridCellSize ) );
        int gz = static_cast<int>(
            Kokkos::floor( ( z - minZ ) / grainGridCellSize ) );

        // Search for *any* grain in shells of increasing size (see Figure 1)
        int shellRadius = 0;
        int guessIndex = -1;
        while ( guessIndex == -1 )
        {
            // The cubic shell consists of 6 planes. We search the 2 planes
            // perpindicular to each axis in separate loops.

            // X planes -- loop over multi-index components of cells
            // in the 2 planes perpendicular to the X axis. Note the Kokkos:min
            // and Kokkos::max guards to prevent from searching cells outside
            // of the lookup grid
            for ( int sy = Kokkos::max( gy - shellRadius, 0 );
                  sy <= Kokkos::min( gy + shellRadius, sizeY - 1 ); ++sy )
            {
                for ( int sz = Kokkos::max( gz - shellRadius, 0 );
                      sz <= Kokkos::min( gz + shellRadius, sizeZ - 1 ); ++sz )
                {
                    // Search plane in negative X direction
                    int sx = Kokkos::max( gx - shellRadius, 0 );
                    guessIndex = grainGrid( sx, sy, sz );
                    // Any nonnegative index in the lookup grid indicates a
                    // point in the cell with multi-index (sx, sy, sz)
                    if ( guessIndex >= 0 )
                    {
                        sy = Kokkos::min( gy + shellRadius, sizeY - 1 ) + 1;
                        break;
                    }

                    // Search plane in positive X direction
                    sx = Kokkos::min( gx + shellRadius, sizeX - 1 );
                    guessIndex = grainGrid( sx, sy, sz );
                    if ( guessIndex >= 0 )
                    {
                        sy = Kokkos::min( gy + shellRadius, sizeY - 1 ) + 1;
                        break;
                    }
                }
            }

            if ( guessIndex >= 0 )
                break;

            // Y planes
            for ( int sx = Kokkos::max( gx - shellRadius, 0 );
                  sx <= Kokkos::min( gx + shellRadius, sizeX - 1 ); ++sx )
            {
                for ( int sz = Kokkos::max( gz - shellRadius, 0 );
                      sz <= Kokkos::min( gz + shellRadius, sizeZ - 1 ); ++sz )
                {
                    int sy = Kokkos::max( gy - shellRadius, 0 );
                    guessIndex = grainGrid( sx, sy, sz );
                    if ( guessIndex >= 0 )
                    {
                        sx = Kokkos::min( gx + shellRadius, sizeX - 1 ) + 1;
                        break;
                    }

                    sy = Kokkos::min( gy + shellRadius, sizeY - 1 );
                    guessIndex = grainGrid( sx, sy, sz );
                    if ( guessIndex >= 0 )
                    {
                        sx = Kokkos::min( gx + shellRadius, sizeX - 1 ) + 1;
                        break;
                    }
                }
            }

            if ( guessIndex >= 0 )
                break;

            // Z planes
            for ( int sy = Kokkos::max( gy - shellRadius, 0 );
                  sy <= Kokkos::min( gy + shellRadius, sizeY - 1 ); ++sy )
            {
                for ( int sx = Kokkos::max( gx - shellRadius, 0 );
                      sx <= Kokkos::min( gx + shellRadius, sizeX - 1 ); ++sx )
                {
                    int sz = Kokkos::max( gz - shellRadius, 0 );
                    guessIndex = grainGrid( sx, sy, sz );
                    if ( guessIndex >= 0 )
                    {
                        sy = Kokkos::min( gy + shellRadius, sizeY - 1 ) + 1;
                        break;
                    }

                    sz = Kokkos::min( gz + shellRadius, sizeZ - 1 );
                    guessIndex = grainGrid( sx, sy, sz );
                    if ( guessIndex >= 0 )
                    {
                        sy = Kokkos::min( gy + shellRadius, sizeY - 1 ) + 1;
                        break;
                    }
                }
            }

            ++shellRadius;
        }

        // Now find the nearest grain center -- the first point we found
        // might not be the closest one

        // Calculate the distance to the guess point found above
        double guessDX = grainPos( guessIndex, 0 ) - x;
        double guessDY = grainPos( guessIndex, 1 ) - y;
        double guessDZ = grainPos( guessIndex, 2 ) - z;
        double dist = Kokkos::sqrt( guessDX * guessDX + guessDY * guessDY +
                                    guessDZ * guessDZ );

        // Now calculate the size of a box of cells around (x, y, z)
        // that contains the guess point and any points that might be
        // closer
        int gridSearchDist =
            static_cast<int>( Kokkos::ceil( dist / grainGridCellSize ) );

        // Now search the box around (x, y, z) of size gridSearchDist
        // for the closest point
        double closestDist = dist;
        int closestIndex = guessIndex;
        for ( int sx = Kokkos::max( gx - gridSearchDist, 0 );
              sx <= Kokkos::min( gx + gridSearchDist, sizeX - 1 ); ++sx )
        {
            for ( int sy = Kokkos::max( gy - gridSearchDist, 0 );
                  sy <= Kokkos::min( gy + gridSearchDist, sizeY - 1 ); ++sy )
            {
                for ( int sz = Kokkos::max( gz - gridSearchDist, 0 );
                      sz <= Kokkos::min( gz + gridSearchDist, sizeZ - 1 );
                      ++sz )
                {
                    int testIndex = grainGrid( sx, sy, sz );
                    // If this cell contains a point, then check if it is
                    // closer to (x, y, z) than the current closest
                    if ( testIndex >= 0 )
                    {
                        double testDX = grainPos( testIndex, 0 ) - x;
                        double testDY = grainPos( testIndex, 1 ) - y;
                        double testDZ = grainPos( testIndex, 2 ) - z;
                        double testDist =
                            Kokkos::sqrt( testDX * testDX + testDY * testDY +
                                          testDZ * testDZ );
                        if ( testDist < closestDist )
                        {
                            closestDist = testDist;
                            closestIndex = testIndex;
                        }
                    }
                }
            }
        }

        return closestIndex;
    }
};
