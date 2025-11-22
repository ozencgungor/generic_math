#ifndef CIR_MODEL_H
#define CIR_MODEL_H

#include <vector>

// CIR (Cox-Ingersoll-Ross) model parameters
struct CIRParams {
    double r0;       // Initial value
    double kappa;    // Mean reversion speed
    double theta;    // Long-term mean
    double sigma;    // Volatility

    CIRParams(double r0, double kappa, double theta, double sigma);
};

// CIR model state (single time point)
struct CIRState {
    double value;

    CIRState(double value = 0.0);
};

// Namespace for CIR model functions
namespace CIR {
    // Update function for CIR model
    void updateCIR(CIRState& current,
                  const CIRState& previous,
                  size_t stepIndex,
                  double dt,
                  const std::vector<double>& dW,
                  const CIRParams& params);
}

#endif // CIR_MODEL_H
