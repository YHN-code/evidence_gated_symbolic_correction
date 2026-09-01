from __future__ import annotations

from typing import Any, Iterable, Mapping

from asrc.constitutive.flac3d_uj_validation import elastic_moduli_pa
from asrc.constitutive.structural_softening_application import (
    StructuralSofteningCase,
    StructuralSofteningLaw,
    path_shear_increments,
    structural_softening_cases,
)
from asrc.constitutive.weak_plane import weak_plane_normal


def structural_softening_output_name(
    law: StructuralSofteningLaw, case: StructuralSofteningCase
) -> str:
    return f"{law.law_id}_{case.case_id}_flac3d.csv"


def _fish_recorders(
    law: StructuralSofteningLaw,
    case: StructuralSofteningCase,
    row_count: int,
) -> list[str]:
    filename = structural_softening_output_name(law, case)
    return [
        "fish define asrc_softening_initialize",
        f"    global asrc_output = array.create({row_count + 1})",
        "    global asrc_step = 0",
        "    asrc_output(1) = 'step,strain_inc_xx,strain_inc_yy,strain_inc_zz,strain_inc_xy,strain_inc_xz,strain_inc_yz,stress_xx_pa,stress_yy_pa,stress_zz_pa,stress_xy_pa,stress_xz_pa,stress_yz_pa,state_bits,plastic_shear_joint,joint_cohesion_pa'",
        "end",
        "[asrc_softening_initialize]",
        "fish define asrc_softening_record",
        "    asrc_step += 1",
        "    local current_zone = zone.near(0.5,0.5,0.5)",
        "    local strain_increment = zone.strain.inc(current_zone)",
        "    local stress = zone.stress(current_zone)",
        "    local state_bits = zone.state(current_zone,1)",
        "    local plastic_shear = zone.prop(current_zone,'strain-shear-plastic-joint')",
        "    local joint_cohesion = zone.prop(current_zone,'joint-cohesion')",
        "    asrc_output(asrc_step + 1) = string(asrc_step) + ',' ...",
        "        + string(comp.xx(strain_increment)) + ',' + string(comp.yy(strain_increment)) + ',' ...",
        "        + string(comp.zz(strain_increment)) + ',' + string(comp.xy(strain_increment)) + ',' ...",
        "        + string(comp.xz(strain_increment)) + ',' + string(comp.yz(strain_increment)) + ',' ...",
        "        + string(comp.xx(stress)) + ',' + string(comp.yy(stress)) + ',' ...",
        "        + string(comp.zz(stress)) + ',' + string(comp.xy(stress)) + ',' ...",
        "        + string(comp.xz(stress)) + ',' + string(comp.yz(stress)) + ',' ...",
        "        + string(state_bits) + ',' + string(plastic_shear) + ',' + string(joint_cohesion)",
        "end",
        "fish define asrc_softening_finalize",
        f"    local status = file.open('{filename}',1,1)",
        f"    status = file.write(asrc_output,{row_count + 1})",
        "    status = file.close",
        "    asrc_output = array.delete(asrc_output)",
        f"    io.out('ASRC_CASE_COMPLETE law={law.law_id} case={case.case_id} rows={row_count}')",
        "end",
    ]


def render_flac3d_structural_softening_data_file(
    config: Mapping[str, Any],
    laws: Iterable[StructuralSofteningLaw],
) -> str:
    elastic = config["material"]["elastic"]
    matrix = config["material"]["intact_matrix"]
    joint = config["material"]["joint"]
    flac = config["flac3d"]
    table = config["table"]
    timestep = float(flac["dynamic_timestep"])
    bulk_pa, shear_pa = elastic_moduli_pa(elastic["young_mpa"], elastic["poisson"])
    cases = structural_softening_cases(config)
    law_values = tuple(laws)
    lines = [
        "; Frozen SR03 structural-softening FLAC3D validation.",
        "model new",
        "model precision 10",
        "model title 'Frozen structural softening validation'",
        "model large-strain off",
        "model deterministic on",
        "model configure dynamic",
        "fish automatic-create off",
        f"model dynamic timestep fix {timestep:.12g}",
    ]
    for law in law_values:
        kappa, cohesion = law.table(
            maximum_damage=float(table["maximum_damage"]),
            point_count=int(table["point_count"]),
        )
        table_name = f"asrc_{law.law_id}"
        lines.append(f"table '{table_name}' delete")
        for x_value, y_value in zip(kappa, cohesion):
            lines.append(
                f"table '{table_name}' add ({x_value:.16g},{y_value * 1.0e6:.16g})"
            )
        for case in cases:
            schedule = path_shear_increments(config["paths"][case.path])
            velocities = [value / timestep for value in schedule]
            normal = weak_plane_normal(case.beta_deg)
            pressure_pa = case.pressure_mpa * 1.0e6
            lines.extend(
                [
                    f"; Law {law.law_id}, case {case.case_id}",
                    "zone delete",
                    "zone create brick size 1 1 1",
                    "zone cmodel assign softening-ubiquitous",
                    f"zone property density {float(flac['density_kg_m3']):.12g} bulk {bulk_pa:.12g} shear {shear_pa:.12g}",
                    f"zone property cohesion {float(matrix['cohesion_mpa']) * 1.0e6:.12g} friction {float(matrix['friction_deg']):.12g} dilation {float(matrix['dilation_deg']):.12g} tension {float(matrix['tension_mpa']) * 1.0e6:.12g}",
                    f"zone property joint-cohesion {law.amplitude_mpa * 1.0e6:.12g} joint-friction {float(joint['friction_deg']):.12g} joint-dilation {float(joint['dilation_deg']):.12g} joint-tension {float(joint['tension_mpa']) * 1.0e6:.12g}",
                    f"zone property table-joint-cohesion '{table_name}'",
                    f"zone property normal ({normal[0]:.16g},{normal[1]:.16g},{normal[2]:.16g})",
                    f"zone initialize stress xx {-pressure_pa:.12g} yy {-pressure_pa:.12g} zz {-pressure_pa:.12g}",
                    "zone gridpoint fix velocity",
                    *_fish_recorders(law, case, len(schedule)),
                ]
            )
            previous: float | None = None
            for velocity in velocities:
                if velocity != previous:
                    lines.extend(
                        [
                            f"zone gridpoint initialize velocity-x {velocity:.16g} range position-z 1",
                            "zone gridpoint initialize velocity-z 0 range position-z 1",
                        ]
                    )
                    previous = velocity
                lines.extend(["model cycle 1", "[asrc_softening_record]"])
            lines.extend(["[asrc_softening_finalize]", ""])
    lines.extend(
        [
            "fish define asrc_softening_complete",
            f"    io.out('ASRC_RESULT laws={len(law_values)} cases={len(cases)} status=complete')",
            "end",
            "[asrc_softening_complete]",
            "program return",
            "",
        ]
    )
    return "\n".join(lines)
