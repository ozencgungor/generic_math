# Monte Carlo Simulation Framework - Architecture

## Overview

A flexible, extensible C++ framework for Monte Carlo simulation of stochastic processes with proper separation of concerns and type safety.

## Design Philosophy

### Core Principles

1. **Separation of Concerns**: Brownian motion generation, simulation logic, and result access are cleanly separated
2. **Flexibility**: States can be simple types or complex classes
3. **Extensibility**: Easy to add new asset classes and models
4. **Type Safety**: Compile-time type checking via templates
5. **Efficiency**: Paths stored as `std::map<int, State>` for O(log n) day-based queries
6. **Correlation Encapsulation**: Correlation handling is encapsulated in BrownianMotion, not in the simulator

## Architecture Layers

### Layer 1: Foundation Classes

#### `BrownianMotion`
- Generates a single Brownian motion path on a day-based schedule
- Stores path as `std::map<int, double>` (day -> value)
- Supports seeded and unseeded construction
- Validates schedule properties (non-decreasing, non-negative)
- Provides `getValue(day)` and `getIncrement(day1, day2)` methods

**Correlation Support**:
- Free functions `generateCorrelatedBrownianMotions()` create correlated BMs
- Uses Cholesky decomposition internally
- Returns vector of already-correlated BrownianMotion objects

#### `MonteCarloSimulator`
- **Takes pre-generated Brownian motions as input** (already correlated if needed)
- **Does NOT handle correlation** - receives already-correlated BMs
- Handles time stepping and extracts Brownian increments from BM maps
- Calls user-provided update callback at each time step with increments `dW`
- Purely functional - no state management or correlation logic

### Layer 2: Model Definitions

Models define three components:

#### Params Struct
```cpp
struct HestonParams {
    double s0, v0, mu, kappa, theta, sigma, rho;
    // Constructor and validation
};
```

#### State Struct (Single Time Point)
```cpp
struct HestonState {
    double spot;
    double variance;
};
```

#### Update Function
```cpp
void updateHeston(HestonState& current, const HestonState& previous,
                  size_t stepIndex, double dt, const std::vector<double>& dW,
                  const HestonParams& params);
```

### Layer 3: Base Classes

#### `ModelSimulator` (Abstract)
- Manages schedule and Brownian motions
- Defines interface: `generateBrownianMotions()`, `simulate()`, `regenerate()`
- Stores Brownian motions internally

#### `ModelGenerator` (Abstract)
- Provides query interface to simulated paths
- Base class for asset-specific generators

### Layer 4: Asset-Specific Implementations

#### Equity (Heston Model)

**EquitySimulator**:
- Manages multiple equities with different Heston parameters
- Each equity gets 2 correlated Brownian motions (spot, variance)
- Correlation generated via `generateCorrelatedBrownianMotions()` with Heston's rho parameter
- Stores paths as `map<string, map<int, HestonState>>` (keyed by day)

Flow:
```cpp
1. EquitySimulator::addEquity(name, params)
2. EquitySimulator::simulateEquity(name)  // Triggered on-demand or explicit
   - Generates 2 correlated BMs using generateCorrelatedBrownianMotions()
     with correlation matrix [[1, rho], [rho, 1]]
   - Creates MonteCarloSimulator with pre-correlated BMs
   - Runs simulation with Heston::updateHeston
   - Caches path as map<int, HestonState>
```

**EquityGenerator**:
- Query interface for equity paths
- Methods: `getSpot()`, `getVariance()`, `getState()`, `getPath()`

#### Credit (CIR Model)

**CreditSimulator**:
- Manages multiple credit/rate processes with different CIR parameters
- Each credit gets 1 Brownian motion (no correlation needed)
- Stores paths as `map<string, map<int, CIRState>>` (keyed by day)

Flow:
```cpp
1. CreditSimulator::addCredit(name, params)
2. CreditSimulator::simulateCredit(name)  // Triggered on-demand or explicit
   - Gets 1 independent BM from the pool
   - Creates MonteCarloSimulator with the BM
   - Runs simulation with CIR::updateCIR
   - Caches path as map<int, CIRState>
```

**CreditGenerator**:
- Query interface for credit/rate paths
- Methods: `getValue()`, `getState()`, `getPath()`

## Complete Simulation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code                                 │
│  1. Create EquitySimulator(schedule, seed)                  │
│  2. Add equities with parameters                            │
│  3. Create EquityGenerator(simulator)                       │
│  4. Query values (triggers lazy simulation)                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              EquitySimulator                                 │
│  - For each equity (on-demand):                             │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ generateCorrelatedBrownianMotions(schedule, 2, corrMatrix) │ │
│    │ Creates MonteCarloSimulator with pre-correlated BMs │ │
│    │  Calls simulate(updateHeston)                        │ │
│    └─────────────┬───────────────────────────────────────┘ │
└──────────────────┼─────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           MonteCarloSimulator                                │
│  - Receives pre-correlated Brownian motions as input       │
│  - For each time step:                                       │
│    - Extracts Brownian increments dW from BM maps          │
│    - Calls updateHeston(state, prevState, i, dt, dW, params)│
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            Heston::updateHeston                              │
│  - Updates state.spot and state.variance                    │
│  - Uses dW[0] for spot, dW[1] for variance (already corr.) │
│  - Returns updated state                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           EquitySimulator                                    │
│  - Stores path: map<int, HestonState> (keyed by day)       │
│  - Caches in: m_equityPaths[equityName]                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         EquityGenerator                                      │
│  - Provides query access to cached paths                    │
│  - getSpot(name, day), getVariance(name, day), etc.        │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. External Brownian Motion Management
**Why**: Allows simulators to control Brownian motion creation, enabling:
- Shared random numbers across different asset classes
- Custom seeding strategies
- Better memory management

### 2. State as Single Time Point
**Why**: Cleaner separation between state data and path storage
- States are lightweight
- Easy to copy and pass around
- Path storage strategy is flexible

### 3. Base Classes for Simulators/Generators
**Why**:
- Asset-specific simulators can manage multiple instances (e.g., many equities)
- Common interface for all asset classes
- Easy to add new asset classes (FX, Rates, etc.)

### 4. Update Function Signature
```cpp
void update(State& current, const State& previous,
            size_t stepIndex, double dt,
            const std::vector<double>& dW,
            const Params& params);
```
**Why**: Provides all necessary information:
- Previous state for computing increments
- Time step for drift terms
- Brownian increments for diffusion
- Parameters for model dynamics

### 5. Correlation Encapsulation in BrownianMotion
**Why**: MonteCarloSimulator should only handle time stepping and increments
- **Separation of concerns**: Correlation is a property of Brownian motions, not the simulation loop
- **Simplicity**: MonteCarloSimulator becomes purely functional - just extracts increments and calls update
- **Flexibility**: Different correlation strategies can be implemented without touching MonteCarloSimulator
- **Reusability**: Correlated BMs can be generated once and reused
- **API**: Helper functions `generateCorrelatedBrownianMotions()` provide clean interface

### 6. Path Storage as `std::map<int, State>` (Day-Based)
**Why**:
- O(log n) lookups by day
- Natural representation of day → state mapping
- Easy iteration in time order
- Schedules specified in days (integers), internally converted to years for simulation

## Adding a New Model

To add a new model (e.g., SABR for FX):

1. **Define components** (FXModel.h/cpp):
```cpp
struct SABRParams { double f0, alpha, beta, rho, nu; };
struct SABRState { double forward, volatility; };
void updateSABR(SABRState& current, const SABRState& previous, ...);
```

2. **Create FXSimulator** (inherits from ModelSimulator):
```cpp
class FXSimulator : public ModelSimulator {
    map<string, SABRParams> m_fxParams;
    map<string, map<double, SABRState>> m_fxPaths;
    // Implement generateBrownianMotions(), simulate(), etc.
};
```

3. **Create FXGenerator** (inherits from ModelGenerator):
```cpp
class FXGenerator : public ModelGenerator {
    // Query methods for FX-specific data
};
```

## Example Usage

See `example_usage.cpp` for complete examples:

```cpp
// Equity
EquitySimulator eqSim(schedule, seed);
eqSim.addEquity("AAPL", hestonParams);
eqSim.generateBrownianMotions();
eqSim.simulate();
EquityGenerator gen(eqSim.getEquityPaths(), schedule);
double spot = gen.getSpot("AAPL", 1.0);

// Credit
CreditSimulator crSim(schedule, seed);
crSim.addCredit("Corp_AAA", cirParams);
crSim.generateBrownianMotions();
crSim.simulate();
CreditGenerator gen(crSim.getCreditPaths(), schedule);
double rate = gen.getValue("Corp_AAA", 3.0);
```

## Naming Conventions

- **Classes**: PascalCase (e.g., `EquitySimulator`)
- **Member variables**: `m_` prefix, camelCase (e.g., `m_equityPaths`)
- **Function parameters**: camelCase (e.g., `timeSchedule`)
- **Local variables**: camelCase (e.g., `updateStep`)

## File Structure

```
BrownianMotion.h/cpp          - Brownian motion generator
MonteCarloSimulator.h/cpp     - Core simulation engine

ModelSimulator.h              - Simulator base class
ModelGenerator.h              - Generator base class

HestonModel.h/cpp            - Heston state, params, update function
CIRModel.h/cpp               - CIR state, params, update function

EquitySimulator.h/cpp        - Equity-specific simulator
EquityGenerator.h/cpp        - Equity-specific generator
CreditSimulator.h/cpp        - Credit-specific simulator
CreditGenerator.h/cpp        - Credit-specific generator

example_usage.cpp            - Usage examples
```
