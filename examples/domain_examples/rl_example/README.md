
# Building Temperature Control with PD-DDPG and DPC

This project implements a **temperature control environment** for a single-zone building, where the objective is to regulate indoor temperature efficiently. The solution includes reinforcement learning (**PD-DDPG**) and differentiable predictive control (**DPC**) approaches.

---

## Table of Contents

1. [Overview](#overview)  
2. [Environment](#environment)  
3. [PD-DDPG Algorithm](#pd-ddpg-algorithm)  
4. [DPC Solution](#dpc-solution)  
5. [Training Process](#training-process)  
6. [Evaluation](#evaluation)  
7. [How to Run](#how-to-run)  
8. [Requirements](#requirements)  

---

## Overview

The environment models indoor temperature as a **linear state-space system**, where:
- **States** represent the indoor temperature.
- **Actions** represent the mass flow rate of the heating system.
- **External disturbances** include outdoor temperature, solar radiation, and occupant heat load.

The agent optimizes its control policy to minimize **energy consumption** while adhering to temperature reference constraints.

---

## Environment

### State Transition Dynamics

The temperature dynamics are modeled as a **linear state-space system**:

```math
x_{t+1} = A x_t + B u_t + E d_t
```

- \( x_t \): System state (indoor temperature).  
- \( u_t \): Control action (heating system mass flow rate).  
- \( d_t \): External disturbances (e.g., outdoor temperature, solar radiation).  
- \( A, B, E \): System matrices.

The output temperature \( y_t \) is measured as:

```math
y_t = C x_t
```

---

### Action Space

The control action \( u_t \) is normalized within the range:

```math
u_t \in [0, 1]
```

It is scaled to the maximum allowable mass flow rate \( U_{\text{max}} \).

---

### Observation Space

The state observed by the agent includes:
1. System states \( x_t \).  
2. Current disturbance \( d_t \).  
3. Reference temperature limits \( [T_{\text{min}}, T_{\text{max}}] \).

---

### Reward Function

The reward penalizes energy consumption:

```math
r_t = - (u_t \cdot 0.01 \cdot U_{\text{max}})
```

- \( u_t \): Normalized control action.  
- \( U_{\text{max}} \): Maximum allowable mass flow rate.

---

### Cost Function

The cost penalizes deviations from temperature limits:

```math
c_t = 50 \cdot \left( \max(T_{\text{min}} - y_t, 0) + \max(y_t - T_{\text{max}}, 0) \right)
```

- \( T_{\text{min}}, T_{\text{max}} \): Reference temperature bounds.  
- \( y_t \): Current indoor temperature.

---

## PD-DDPG Algorithm

The **Primal-Dual Deep Deterministic Policy Gradient (PD-DDPG)** handles the optimization of both rewards and constraints in a **Constrained Markov Decision Process (CMDP)**.

### Dual Objective (Lagrangian Function)

The algorithm optimizes the following Lagrangian:

```math
\mathcal{L}(\pi, \lambda) = \mathbb{E}_{\pi} \left[ r(s_t, a_t) - \lambda \cdot c(s_t, a_t) \right]
```

- \( r(s_t, a_t) \): Immediate reward.  
- \( c(s_t, a_t) \): Immediate cost.  
- \( \lambda \): Dual variable balancing rewards and costs.

---

### Critic and Cost Networks

1. **Critic Network**: Estimates immediate rewards:

```math
Q(s_t, a_t) = r(s_t, a_t)
```

2. **Cost Network**: Estimates immediate costs:

```math
C(s_t, a_t) = c(s_t, a_t)
```

The loss functions are:

```math
\mathcal{L}_{\text{critic}} = \mathbb{E} \left[ \left( Q(s_t, a_t) - r(s_t, a_t) \right)^2 \right]
```

```math
\mathcal{L}_{\text{cost}} = \mathbb{E} \left[ \left( C(s_t, a_t) - c(s_t, a_t) \right)^2 \right]
```

---

### Actor Network Update

The actor optimizes:

1. **Maximize reward** \( Q(s_t, a_t) \).  
2. **Minimize cost** \( C(s_t, a_t) \).  
3. **Ensure smooth actions**:

```math
\mathcal{L}_{\text{actor}} = -\mathbb{E} \left[ Q(s_t, a_t) - \lambda \cdot C(s_t, a_t) \right] + \beta \cdot \| a_t - a_{t+1} \|^2
```

---

## DPC Solution

The **DPC** solution is implemented using the Neuromancer framework and focuses on minimizing energy while satisfying constraints.

### Control Policy

A **neural network policy** generates control actions based on normalized inputs:

```math
u = \text{policy}(x, d, T_{\text{min}}, T_{\text{max}})
```

### Optimization Objectives

1. **Minimize energy usage**:

```math
\text{action loss} = 0.01 \cdot (u == 0.0)
```

2. **Enforce smooth control actions**:

```math
\text{delta u loss} = 0.1 \cdot (u_{t} - u_{t-1})^2
```

3. **Ensure state constraints**:

```math
\text{penalty} = 50 \cdot \left( y < T_{\text{min}} \right) + 50 \cdot \left( y > T_{\text{max}} \right)
```

---

## Training Process

1. **RL Training**:
   - Environment initialized using `SimpleBuildingEnv`.
   - Actions selected using exploration noise during training.
   - Rewards and costs logged for policy updates.
   - Actor-Critic and dual variable \( \lambda \) are updated.

2. **DPC Training**:
   - Simulated trajectories are generated.
   - Neuromancer optimizes the policy for energy efficiency and constraint satisfaction.

---

## Evaluation

- Evaluate the **PD-DDPG** policy and **DPC** solution.
- Compare:
  - Controlled temperature vs. reference limits.
  - Energy usage (actions).
  - Disturbance impacts.

---

## How to Run

### Install Dependencies

Run the following command:

```bash
pip install torch numpy matplotlib gym neuromancer
```

### Run RL Training

Open and execute the RL notebook:

```bash
jupyter notebook SimpleBuildingControl- RL.ipynb
```

### Run DPC Training

Open and execute the DPC notebook:

```bash
jupyter notebook DPC.ipynb
```

---

## Requirements

- Python 3.8+  
- PyTorch  
- Gym  
- Neuromancer  
- NumPy  
- Matplotlib  
