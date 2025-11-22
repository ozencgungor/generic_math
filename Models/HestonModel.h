#ifndef HESTON_MODEL_H
#define HESTON_MODEL_H

#include <vector>

// Heston model parameters
struct HestonParams {
    double s0;       // Initial spot price
    double v0;       // Initial variance
    double mu;       // Drift (risk-free rate)
    double kappa;    // Mean reversion speed
    double theta;    // Long-term variance
    double sigma;    // Volatility of variance
    double rho;      // Correlation between spot and variance Brownian motions

    HestonParams(double s0, double v0, double mu, double kappa,
                 double theta, double sigma, double rho);
};

// Heston model state (single time point)
struct HestonState {
    double spot;
    double variance;

    HestonState(double spot = 0.0, double variance = 0.0);
};

// Namespace for Heston model functions
namespace Heston {
    // Update function for Heston model
    void updateHeston(HestonState& current,
                     const HestonState& previous,
                     size_t stepIndex,
                     double dt,
                     const std::vector<double>& dW,
                     const HestonParams& params);
}

#endif // HESTON_MODEL_H
