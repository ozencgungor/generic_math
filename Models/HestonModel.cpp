#include "HestonModel.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// HestonParams Implementation
// ============================================================================

HestonParams::HestonParams(double s0, double v0, double mu, double kappa,
                          double theta, double sigma, double rho)
    : s0(s0), v0(v0), mu(mu), kappa(kappa), theta(theta), sigma(sigma), rho(rho) {

    if (s0 <= 0.0) throw std::invalid_argument("Initial spot must be positive");
    if (v0 < 0.0) throw std::invalid_argument("Initial variance must be non-negative");
    if (kappa <= 0.0) throw std::invalid_argument("Mean reversion speed must be positive");
    if (theta < 0.0) throw std::invalid_argument("Long-term variance must be non-negative");
    if (sigma < 0.0) throw std::invalid_argument("Vol of variance must be non-negative");
    if (rho < -1.0 || rho > 1.0) throw std::invalid_argument("Correlation must be in [-1, 1]");
}

// ============================================================================
// HestonState Implementation
// ============================================================================

HestonState::HestonState(double spot, double variance)
    : spot(spot), variance(variance) {}

// ============================================================================
// Heston Update Function
// ============================================================================

namespace Heston {

void updateHeston(HestonState& current,
                 const HestonState& previous,
                 size_t stepIndex,
                 double dt,
                 const std::vector<double>& dW,
                 const HestonParams& params) {

    if (dW.size() < 2) {
        throw std::invalid_argument("Heston model requires 2 Brownian motions");
    }

    // Initial state
    if (stepIndex == 0) {
        current.spot = params.s0;
        current.variance = params.v0;
        return;
    }

    // Update variance (CIR process with truncation to ensure non-negativity)
    double v = std::max(previous.variance, 0.0);
    double dV = params.kappa * (params.theta - v) * dt
                + params.sigma * std::sqrt(v) * dW[1];
    current.variance = std::max(v + dV, 0.0);

    // Update spot
    double dS = params.mu * previous.spot * dt
                + previous.spot * std::sqrt(v) * dW[0];
    current.spot = previous.spot + dS;
}

} // namespace Heston
