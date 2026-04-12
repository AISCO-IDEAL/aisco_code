#eschaerer's **Solar v3B_2 loss update**
## Context
v3B returned a surrogate-scale value (often near zero) with useful gradients; v3B_2 now returns meaningful solar watts while preserving surrogate gradients for mesh_sdf backward; manual objective mode in total_loss sign-flips solar so minimizing total still maximizes power.
## Goal
Use solar as a physically meaningful multi-objective signal (target range roughly 0..5000 W depending on geometry/config) with stable optimization gradients.
## Formula
For selected faces: $$P=\sum_i A_i\cdot\max(\cos\theta_i,0)\cdot I_{solar}\cdot\eta$$ with optional output scaling: $$P_{out}=\text{power_output_scale}\cdot P$$ and manual objective weighting term: $$\mathcal{L}^{weighted}_{solar}=-w_{solar}\,P_{out}$$
## Surrogate-vs-true link (mesh_sdf)
Straight-through merge: $$y=s+\operatorname{detach}(p-s)$$ where $p=P_{out}$ (forward value shown/logged) and $s$ is surrogate tensor (backward gradient path); forward gives meaningful watts, backward uses surrogate gradients.
## All changed parts
- subprojects/wp4_aeroshell_optimizer/v1/src/losses/total_loss.py: manual objective branch now applies negative sign only to solar term.
- subprojects/wp4_aeroshell_optimizer/v1/src/losses/solar_loss/solar_v3B_2.py: removed legacy wrapper-side multiplication by power_conversion_factor; output stays in watts from estimator.
- subprojects/wp4_aeroshell_optimizer/v1/configs/solar_configs/solar_loss_config.yaml: added SolarLossConfigV3.power_output_scale (default 1.0) for editable output calibration.
- subprojects/wp3_solar_estimator/v3B_2/solar_loss_v3B_2.py: added power_output_scale plumbing (config->estimator->solar_mesh_loss_v3), applied scaling in final watts output, and added logging fields solar_power_watts_unscaled + solar_power_output_scale.
- subprojects/wp3_solar_estimator/v3B_2/Solar v3B_2 Loss Update.md: revised to compact style and full change trace.
## Config notes
Use SolarLossConfigV3.power_output_scale to tune range without changing physics internals; keep solar_irradiance_W_m2 and efficiency as physical parameters; wrapper-side scaling remains disabled for v3B_2 to avoid double-scaling.
## Status
Implemented; v3B_2 now supports editable config scaling, manual sign convention, and explicit surrogate/true linkage.
