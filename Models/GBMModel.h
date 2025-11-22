#ifndef GBM_MODEL_H
#define GBM_MODEL_H

#include <vector>
#include <type_traits>

// Geometric Brownian Motion (GBM) model - templated on MarketObject type
// MarketObject can be a scalar (double), vector, matrix, or custom class
// All elements evolve with the SAME Brownian motion

template<typename MarketObject>
struct GBMParams {
    MarketObject initialValue;  // Can be any type
    double drift;               // mu (scalar)
    double volatility;          // sigma (scalar)

    GBMParams(const MarketObject& initialValue, double drift, double volatility)
        : initialValue(initialValue), drift(drift), volatility(volatility) {}
};

template<typename MarketObject>
struct GBMState {
    MarketObject value;

    GBMState() = default;
    GBMState(const MarketObject& value) : value(value) {}
};

// Namespace for GBM model functions
namespace GBM {

    // Generic update function for any MarketObject type
    // Evolution: S_new = S_old * (1 + mu*dt + sigma*dW)
    template<typename MarketObject>
    inline void updateGBM(GBMState<MarketObject>& current,
                         const GBMState<MarketObject>& previous,
                         size_t stepIndex,
                         double dt,
                         const std::vector<double>& dW,
                         const GBMParams<MarketObject>& params) {

        if (dW.empty()) {
            throw std::invalid_argument("GBM model requires at least 1 Brownian motion");
        }

        // Initial state
        if (stepIndex == 0) {
            current.value = params.initialValue;
            return;
        }

        // GBM evolution: S_new = S_old * (1 + mu*dt + sigma*dW)
        // All elements use the SAME Brownian motion dW[0]
        double factor = 1.0 + params.drift * dt + params.volatility * dW[0];
        current.value = previous.value * factor;
    }

} // namespace GBM

#endif // GBM_MODEL_H
