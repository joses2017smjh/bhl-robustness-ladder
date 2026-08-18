#!/bin/bash
# Domain-randomization fidelity ladder, parameterised by a scale factor.
#
# Every DR range in upstream's EventsCfg is (center +/- half_width). A rung of
# scale s uses (center +/- s*half_width), so:
#     s=0.0  no randomization at all
#     s=1.0  exactly upstream's shipped values
#     s=2.0  double-width
#
# Expressing rungs on a continuous axis rather than as three named presets is
# what turns this from three points into a retention curve, and it lets the
# cliff be located rather than merely hit. The first pass used s=0/1/2 and
# found s=2 unlearnable (reward 4.6 vs 48 at s=0, 69% falling), so the
# interesting region is between 1 and 2.
#
# Initial-state randomization (reset_base, reset_robot_joints) is deliberately
# held constant across rungs so the ladder isolates one variable.
#
# Usage: dr_overrides_scale <scale>

dr_overrides_scale() {
    awk -v s="$1" 'BEGIN {
        # Two kinds of parameter, scaled differently:
        #  - absolute physical quantities (friction, gain multipliers) scale
        #    around their CENTER, so s=0 pins them to the nominal value;
        #  - additive offsets (mass delta, joint offset, external wrench) scale
        #    their ENDPOINTS from zero, so s=0 means "no offset applied".
        # Scaling mass around its center would silently add a constant +0.5 kg
        # at s=0 and break agreement with the already-trained s=0 runs.
        printf "env.events.physics_material.params.static_friction_range=[%.4f,%.4f] ",  0.8 - s*0.4, 0.8 + s*0.4
        printf "env.events.physics_material.params.dynamic_friction_range=[%.4f,%.4f] ", 0.8 - s*0.4, 0.8 + s*0.4
        printf "env.events.add_base_mass.params.mass_distribution_params=[%.4f,%.4f] ",  -1.0*s, 2.0*s
        printf "env.events.add_all_joint_default_pos.params.pos_distribution_params=[%.4f,%.4f] ", -s*0.05, s*0.05
        printf "env.events.scale_all_actuator_torque_constant.params.stiffness_distribution_params=[%.4f,%.4f] ", 1.0 - s*0.2, 1.0 + s*0.2
        printf "env.events.scale_all_actuator_torque_constant.params.damping_distribution_params=[%.4f,%.4f] ",   1.0 - s*0.2, 1.0 + s*0.2
        printf "env.events.base_external_force_torque.params.force_range=[%.4f,%.4f] ",  -s*2.0, s*2.0
        printf "env.events.base_external_force_torque.params.torque_range=[%.4f,%.4f]\n", -s*2.0, s*2.0
    }'
}

# Back-compat names used by the first pass.
dr_overrides() {
    case "$1" in
        off)        dr_overrides_scale 0.0 ;;
        default)    dr_overrides_scale 1.0 ;;
        aggressive) dr_overrides_scale 2.0 ;;
        *) echo "unknown DR level: $1" >&2; return 1 ;;
    esac
}
