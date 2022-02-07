#ifndef INPUTS_H
#define INPUTS_H

#include <string>

namespace CabanaPD
{

class Inputs
{
  public:
    std::string output_file = "cabanaPD.out";
    std::string error_file = "cabanaPD.err";
    std::string device_type = "SERIAL";
    int output_frequency = 1;

    std::array<int, 3> num_cells;
    std::array<double, 3> low_corner;
    std::array<double, 3> high_corner;

    std::size_t num_steps;
    double final_time;
    double timestep;

    std::string force_type;
    bool half_neigh;
    double K;
    double delta;

    Inputs( const int nc, const double lc, const double hc, const double K,
            const double d, const double t_f, const double dt );
    ~Inputs();
    void read_args( int argc, char* argv[] );
};

} // namespace CabanaPD

#endif
