from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


VOIGT_COMPONENTS = ("11", "22", "12")
CORRECTION_COEFFICIENTS = ("delta_c11", "delta_c12", "delta_c22", "delta_c66")


@dataclass(frozen=True)
class ConstitutiveFit:
    """A fitted correction to a local two-dimensional stiffness matrix."""

    method: str
    correction_matrix_local_mpa: np.ndarray
    corrected_matrix_local_mpa: np.ndarray
    coefficients: dict[str, float]
    formulae: dict[str, str]


def isotropic_plane_stress_stiffness(young_mpa: float, poisson: float) -> np.ndarray:
    """Return the engineering-Voigt plane-stress stiffness matrix."""
    if young_mpa <= 0.0:
        raise ValueError("young_mpa must be positive.")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5.")
    scale = float(young_mpa) / (1.0 - float(poisson) ** 2)
    return scale * np.asarray(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1.0 - poisson) / 2.0],
        ],
        dtype=float,
    )


def orthotropic_plane_stress_stiffness(
    young_1_mpa: float,
    young_2_mpa: float,
    poisson_12: float,
    shear_12_mpa: float,
) -> np.ndarray:
    """Return a reciprocal orthotropic plane-stress stiffness matrix."""
    values = [young_1_mpa, young_2_mpa, shear_12_mpa]
    if any(float(value) <= 0.0 for value in values):
        raise ValueError("Orthotropic Young and shear moduli must be positive.")
    compliance = np.asarray(
        [
            [1.0 / young_1_mpa, -poisson_12 / young_1_mpa, 0.0],
            [-poisson_12 / young_1_mpa, 1.0 / young_2_mpa, 0.0],
            [0.0, 0.0, 1.0 / shear_12_mpa],
        ],
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(compliance)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("The orthotropic compliance matrix is not positive definite.")
    return np.linalg.inv(compliance)


def _rotation(beta_deg: float) -> np.ndarray:
    beta = np.deg2rad(float(beta_deg))
    return np.asarray(
        [[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]],
        dtype=float,
    )


def _strain_vector_to_tensor(strain: np.ndarray) -> np.ndarray:
    e11, e22, gamma12 = np.asarray(strain, dtype=float)
    return np.asarray([[e11, gamma12 / 2.0], [gamma12 / 2.0, e22]], dtype=float)


def _strain_tensor_to_vector(strain: np.ndarray) -> np.ndarray:
    return np.asarray([strain[0, 0], strain[1, 1], 2.0 * strain[0, 1]], dtype=float)


def _stress_vector_to_tensor(stress: np.ndarray) -> np.ndarray:
    s11, s22, tau12 = np.asarray(stress, dtype=float)
    return np.asarray([[s11, tau12], [tau12, s22]], dtype=float)


def _stress_tensor_to_vector(stress: np.ndarray) -> np.ndarray:
    return np.asarray([stress[0, 0], stress[1, 1], stress[0, 1]], dtype=float)


def strain_global_to_local(strain_global: np.ndarray, beta_deg: float) -> np.ndarray:
    rotation = _rotation(beta_deg)
    tensor = _strain_vector_to_tensor(strain_global)
    return _strain_tensor_to_vector(rotation.T @ tensor @ rotation)


def stress_local_to_global(stress_local: np.ndarray, beta_deg: float) -> np.ndarray:
    rotation = _rotation(beta_deg)
    tensor = _stress_vector_to_tensor(stress_local)
    return _stress_tensor_to_vector(rotation @ tensor @ rotation.T)


def stress_global_to_local(stress_global: np.ndarray, beta_deg: float) -> np.ndarray:
    rotation = _rotation(beta_deg)
    tensor = _stress_vector_to_tensor(stress_global)
    return _stress_tensor_to_vector(rotation.T @ tensor @ rotation)


def predict_stress(
    strain_global: np.ndarray,
    beta_deg: float,
    stiffness_local_mpa: np.ndarray,
) -> np.ndarray:
    """Apply a local constitutive matrix and rotate its stress to global axes."""
    strain_local = strain_global_to_local(strain_global, beta_deg)
    stress_local = np.asarray(stiffness_local_mpa, dtype=float) @ strain_local
    return stress_local_to_global(stress_local, beta_deg)


def _path_vector(name: str) -> np.ndarray:
    paths = {
        "uniaxial_x": np.asarray([-1.0, 0.0, 0.0]),
        "uniaxial_y": np.asarray([0.0, -1.0, 0.0]),
        "pure_shear": np.asarray([0.0, 0.0, 1.0]),
        "biaxial_2_to_1": np.asarray([-1.0, -0.5, 0.0]),
        "mixed_compression_shear": np.asarray([-0.8, -0.2, 0.65]),
        "reverse_mixed": np.asarray([-0.25, -0.9, -0.55]),
    }
    try:
        return paths[str(name)].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown material-point loading path: {name}") from exc


def _dataset_block(
    *,
    orientations: list[float],
    paths: list[str],
    amplitudes: list[float],
    partition: str,
    strain_scale: float,
    base_stiffness: np.ndarray,
    reference_stiffness: np.ndarray,
    noise_std_mpa: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta_deg in orientations:
        for path_name in paths:
            direction = _path_vector(path_name)
            for step, amplitude in enumerate(amplitudes, start=1):
                strain_global = direction * float(strain_scale) * float(amplitude)
                strain_local = strain_global_to_local(strain_global, beta_deg)
                stress_base_global = predict_stress(strain_global, beta_deg, base_stiffness)
                stress_reference_global = predict_stress(
                    strain_global, beta_deg, reference_stiffness
                )
                stress_base_local = base_stiffness @ strain_local
                stress_reference_local = reference_stiffness @ strain_local
                if noise_std_mpa > 0.0:
                    local_noise = rng.normal(0.0, noise_std_mpa, size=3)
                    stress_target_local = stress_reference_local + local_noise
                    stress_target_global = stress_local_to_global(
                        stress_target_local, beta_deg
                    )
                else:
                    stress_target_local = stress_reference_local
                    stress_target_global = stress_reference_global
                row: dict[str, Any] = {
                    "partition": partition,
                    "beta_deg": float(beta_deg),
                    "path": path_name,
                    "step": step,
                    "amplitude": float(amplitude),
                }
                for index, component in enumerate(VOIGT_COMPONENTS):
                    row[f"strain_global_{component}"] = float(strain_global[index])
                    row[f"strain_local_{component}"] = float(strain_local[index])
                    row[f"stress_base_global_{component}_MPa"] = float(
                        stress_base_global[index]
                    )
                    row[f"stress_reference_global_{component}_MPa"] = float(
                        stress_reference_global[index]
                    )
                    row[f"stress_target_global_{component}_MPa"] = float(
                        stress_target_global[index]
                    )
                    row[f"stress_base_local_{component}_MPa"] = float(
                        stress_base_local[index]
                    )
                    row[f"stress_reference_local_{component}_MPa"] = float(
                        stress_reference_local[index]
                    )
                    row[f"stress_target_local_{component}_MPa"] = float(
                        stress_target_local[index]
                    )
                    row[f"residual_local_{component}_MPa"] = float(
                        stress_target_local[index] - stress_base_local[index]
                    )
                rows.append(row)
    return rows


def generate_material_point_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate calibration and independently structured material-point partitions."""
    material = config["material"]
    base = isotropic_plane_stress_stiffness(
        float(material["baseline"]["young_mpa"]),
        float(material["baseline"]["poisson"]),
    )
    reference = orthotropic_plane_stress_stiffness(
        float(material["reference"]["young_1_mpa"]),
        float(material["reference"]["young_2_mpa"]),
        float(material["reference"]["poisson_12"]),
        float(material["reference"]["shear_12_mpa"]),
    )
    sampling = config["sampling"]
    calibration_angles = [float(value) for value in sampling["calibration_beta_deg"]]
    locked_angles = [float(value) for value in sampling["locked_beta_deg"]]
    calibration_paths = [str(value) for value in sampling["calibration_paths"]]
    locked_paths = [str(value) for value in sampling["locked_paths"]]
    amplitudes = [float(value) for value in sampling["amplitudes"]]
    strain_scale = float(sampling["strain_scale"])
    rng = np.random.default_rng(int(config.get("seed", 20260730)))
    noise_std = float(sampling.get("calibration_noise_std_mpa", 0.0))

    blocks = [
        (calibration_angles, calibration_paths, "calibration", noise_std),
        (locked_angles, calibration_paths, "locked_angle", 0.0),
        (calibration_angles, locked_paths, "locked_path", 0.0),
        (locked_angles, locked_paths, "locked_joint", 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for angles, paths, partition, block_noise in blocks:
        rows.extend(
            _dataset_block(
                orientations=angles,
                paths=paths,
                amplitudes=amplitudes,
                partition=partition,
                strain_scale=strain_scale,
                base_stiffness=base,
                reference_stiffness=reference,
                noise_std_mpa=block_noise,
                rng=rng,
            )
        )
    return pd.DataFrame(rows), base, reference


def _fit_ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_local_correction(
    calibration: pd.DataFrame,
    base_stiffness_local_mpa: np.ndarray,
    *,
    method: str,
    alpha: float = 0.0,
) -> ConstitutiveFit:
    """Fit either a generic component map or a symmetric orthotropic correction."""
    strain = calibration[
        [f"strain_local_{component}" for component in VOIGT_COMPONENTS]
    ].to_numpy(float)
    residual = calibration[
        [f"residual_local_{component}_MPa" for component in VOIGT_COMPONENTS]
    ].to_numpy(float)

    if method == "baseline":
        correction = np.zeros((3, 3), dtype=float)
    elif method == "generic_componentwise":
        # Y = X B, while constitutive notation uses Y = DeltaC X.
        correction = _fit_ridge(strain, residual, alpha).T
    elif method == "constitutive_asrc":
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for eps, delta_sigma in zip(strain, residual):
            e11, e22, gamma12 = eps
            rows.extend(
                [
                    np.asarray([e11, e22, 0.0, 0.0]),
                    np.asarray([0.0, e11, e22, 0.0]),
                    np.asarray([0.0, 0.0, 0.0, gamma12]),
                ]
            )
            targets.extend(delta_sigma.tolist())
        fitted = _fit_ridge(np.vstack(rows), np.asarray(targets), alpha)
        d11, d12, d22, d66 = fitted
        correction = np.asarray(
            [[d11, d12, 0.0], [d12, d22, 0.0], [0.0, 0.0, d66]],
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown constitutive correction method: {method}")

    corrected = np.asarray(base_stiffness_local_mpa, dtype=float) + correction
    coefficients = {
        "delta_c11": float(correction[0, 0]),
        "delta_c12": float(0.5 * (correction[0, 1] + correction[1, 0])),
        "delta_c22": float(correction[1, 1]),
        "delta_c66": float(correction[2, 2]),
    }
    formulae = {
        "delta_sigma_11": "delta_c11 * epsilon_11 + delta_c12 * epsilon_22",
        "delta_sigma_22": "delta_c12 * epsilon_11 + delta_c22 * epsilon_22",
        "delta_tau_12": "delta_c66 * gamma_12",
    }
    if method == "generic_componentwise":
        strain_names = ("epsilon_11", "epsilon_22", "gamma_12")
        formulae = {
            f"delta_sigma_{VOIGT_COMPONENTS[row]}": " + ".join(
                f"delta_c{row + 1}{column + 1} * {strain_names[column]}"
                for column in range(3)
            )
            for row in range(3)
        }
        coefficients = {
            f"delta_c{row + 1}{column + 1}": float(correction[row, column])
            for row in range(3)
            for column in range(3)
        }
    return ConstitutiveFit(
        method=method,
        correction_matrix_local_mpa=correction,
        corrected_matrix_local_mpa=corrected,
        coefficients=coefficients,
        formulae=formulae,
    )


def predict_dataset(frame: pd.DataFrame, fit: ConstitutiveFit) -> pd.DataFrame:
    predictions = frame.copy()
    for component in VOIGT_COMPONENTS:
        predictions[f"stress_predicted_global_{component}_MPa"] = np.nan
    for row_index, row in predictions.iterrows():
        strain_global = np.asarray(
            [row[f"strain_global_{component}"] for component in VOIGT_COMPONENTS],
            dtype=float,
        )
        stress = predict_stress(
            strain_global,
            float(row["beta_deg"]),
            fit.corrected_matrix_local_mpa,
        )
        for component_index, component in enumerate(VOIGT_COMPONENTS):
            predictions.at[
                row_index, f"stress_predicted_global_{component}_MPa"
            ] = float(stress[component_index])
    predictions["method"] = fit.method
    return predictions


def stress_rmse(frame: pd.DataFrame) -> float:
    errors: list[np.ndarray] = []
    for component in VOIGT_COMPONENTS:
        errors.append(
            frame[f"stress_reference_global_{component}_MPa"].to_numpy(float)
            - frame[f"stress_predicted_global_{component}_MPa"].to_numpy(float)
        )
    values = np.concatenate(errors)
    return float(np.sqrt(np.mean(values**2)))


def constitutive_audit(
    fit: ConstitutiveFit,
    reference_stiffness_local_mpa: np.ndarray,
    *,
    random_seed: int = 20260730,
    samples: int = 512,
) -> dict[str, float | int | bool]:
    """Audit symmetry, positive energy, stiffness recovery, and frame covariance."""
    stiffness = np.asarray(fit.corrected_matrix_local_mpa, dtype=float)
    reference = np.asarray(reference_stiffness_local_mpa, dtype=float)
    symmetry_error = float(np.max(np.abs(stiffness - stiffness.T)))
    symmetric_part = 0.5 * (stiffness + stiffness.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_part)
    min_eigenvalue = float(np.min(eigenvalues))
    rng = np.random.default_rng(int(random_seed))
    strains = rng.normal(0.0, 1e-3, size=(int(samples), 3))
    energies = 0.5 * np.einsum("ni,ij,nj->n", strains, symmetric_part, strains)

    covariance_errors: list[float] = []
    for _ in range(min(64, int(samples))):
        strain = rng.normal(0.0, 1e-3, size=3)
        beta = float(rng.uniform(0.0, 90.0))
        frame_rotation = float(rng.uniform(-90.0, 90.0))
        rotation = _rotation(frame_rotation)
        rotated_strain_tensor = (
            rotation @ _strain_vector_to_tensor(strain) @ rotation.T
        )
        rotated_strain = _strain_tensor_to_vector(rotated_strain_tensor)
        original_stress = predict_stress(strain, beta, stiffness)
        expected_rotated_stress = _stress_tensor_to_vector(
            rotation @ _stress_vector_to_tensor(original_stress) @ rotation.T
        )
        actual_rotated_stress = predict_stress(
            rotated_strain, beta + frame_rotation, stiffness
        )
        covariance_errors.append(
            float(np.max(np.abs(expected_rotated_stress - actual_rotated_stress)))
        )

    relative_stiffness_error = float(
        np.linalg.norm(stiffness - reference) / np.linalg.norm(reference)
    )
    tolerance = 1e-8
    violations = (
        int(symmetry_error > tolerance)
        + int(min_eigenvalue <= 0.0)
        + int(float(np.min(energies)) < -tolerance)
        + int(max(covariance_errors, default=0.0) > tolerance)
    )
    return {
        "physical_violations": violations,
        "major_symmetry_error_MPa": symmetry_error,
        "minimum_stiffness_eigenvalue_MPa": min_eigenvalue,
        "minimum_sampled_energy_MPa": float(np.min(energies)),
        "frame_covariance_max_error_MPa": max(covariance_errors, default=0.0),
        "relative_stiffness_error": relative_stiffness_error,
        "positive_definite": min_eigenvalue > 0.0,
    }
