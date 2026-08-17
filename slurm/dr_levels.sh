#!/bin/bash
# Domain-randomization fidelity ladder: three rungs, expressed purely as Hydra
# overrides of upstream's EventsCfg. No code fork is needed because BHL's
# train.py is decorated with @hydra_task_config.
#
# "DR" here means randomization of *physics parameters* only. Initial-state
# randomization (reset_base pose/velocity, reset_robot_joints) is held constant
# across all three rungs, so the ladder isolates one variable.
#
# Rung definitions:
#   off        - every physics parameter pinned to its nominal value
#   default    - upstream's shipped ranges, unmodified
#   aggressive - roughly 2x the default half-width on every axis
#
# Usage: dr_overrides <off|default|aggressive>

dr_overrides() {
    case "$1" in
    off)
        echo "env.events.physics_material.params.static_friction_range=[0.8,0.8] \
              env.events.physics_material.params.dynamic_friction_range=[0.8,0.8] \
              env.events.add_base_mass.params.mass_distribution_params=[0.0,0.0] \
              env.events.add_all_joint_default_pos.params.pos_distribution_params=[0.0,0.0] \
              env.events.scale_all_actuator_torque_constant.params.stiffness_distribution_params=[1.0,1.0] \
              env.events.scale_all_actuator_torque_constant.params.damping_distribution_params=[1.0,1.0] \
              env.events.base_external_force_torque.params.force_range=[0.0,0.0] \
              env.events.base_external_force_torque.params.torque_range=[0.0,0.0]"
        ;;
    default)
        # Upstream values, stated explicitly rather than omitted, so the run is
        # self-documenting and a change upstream cannot silently move the rung.
        echo "env.events.physics_material.params.static_friction_range=[0.4,1.2] \
              env.events.physics_material.params.dynamic_friction_range=[0.4,1.2] \
              env.events.add_base_mass.params.mass_distribution_params=[-1.0,2.0] \
              env.events.add_all_joint_default_pos.params.pos_distribution_params=[-0.05,0.05] \
              env.events.scale_all_actuator_torque_constant.params.stiffness_distribution_params=[0.8,1.2] \
              env.events.scale_all_actuator_torque_constant.params.damping_distribution_params=[0.8,1.2] \
              env.events.base_external_force_torque.params.force_range=[-2.0,2.0] \
              env.events.base_external_force_torque.params.torque_range=[-2.0,2.0]"
        ;;
    aggressive)
        echo "env.events.physics_material.params.static_friction_range=[0.2,1.8] \
              env.events.physics_material.params.dynamic_friction_range=[0.2,1.8] \
              env.events.add_base_mass.params.mass_distribution_params=[-2.0,4.0] \
              env.events.add_all_joint_default_pos.params.pos_distribution_params=[-0.10,0.10] \
              env.events.scale_all_actuator_torque_constant.params.stiffness_distribution_params=[0.6,1.4] \
              env.events.scale_all_actuator_torque_constant.params.damping_distribution_params=[0.6,1.4] \
              env.events.base_external_force_torque.params.force_range=[-4.0,4.0] \
              env.events.base_external_force_torque.params.torque_range=[-4.0,4.0]"
        ;;
    *)
        echo "unknown DR level: $1" >&2
        return 1
        ;;
    esac
}
