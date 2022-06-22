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
    int output_frequency = 10;

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

    Inputs( const std::array<int, 3> nc, std::array<double, 3> lc,
            std::array<double, 3> hc, const double K, const double d,
            const double t_f, const double dt );
    ~Inputs();
    void read_args( int argc, char* argv[] );
};

class InputsFracture : public Inputs
{
  public:
    using Inputs::Inputs;
    double G0;

    InputsFracture( const std::array<int, 3> nc, std::array<double, 3> lc,
                    std::array<double, 3> hc, const double K, const double d,
                    const double t_f, const double dt, const double G0 );
    ~InputsFracture();
};

} // namespace CabanaPD

#endif
