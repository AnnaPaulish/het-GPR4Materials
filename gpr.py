from typing import Dict, List, Union

import ase
import numpy as np
import scipy
import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap, TensorBlock, Labels
from metatensor.torch.atomistic import ModelOutput
from metatensor.torch.learn.data import Dataset

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.additive import CompositionModel, remove_additive

from featomic.torch import SoapPowerSpectrum, systems_to_torch


class SoapCalculator(torch.nn.Module):
    """
    Computes a SOAP descriptor vector. Constructor accepts hypers.
    """

    def __init__(self, soap_hypers: dict) -> None:
        super().__init__()
        self.soap_calculator = SoapPowerSpectrum(**soap_hypers)

    def forward(self, systems, gradients: List[str] = None) -> TensorMap:
        soap_vector = self.soap_calculator.compute(systems, gradients)
        soap_vector = soap_vector.keys_to_samples("center_type")
        soap_vector = soap_vector.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
        return soap_vector


class PolynomialKernel(torch.nn.Module):

    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree

    def forward(self, tensor_1: TensorMap, tensor_2: TensorMap) -> TensorMap:
        return mts.pow(mts.dot(tensor_1, tensor_2), self.degree)


class KernelCalculator(torch.nn.Module):

    def __init__(self, kernel_fn: callable, sparse_points: TensorMap) -> None:
        super().__init__()
        self.kernel_fn = kernel_fn
        self.sparse_points = sparse_points

        # Compute k_mm from the sparse points
        self.k_mm = self.kernel_fn(self.sparse_points, self.sparse_points)

        # Initialize the Subset of Regressors solvers
        self.solver = _SorKernelSolver(self.k_mm[0].values)

    def forward(self, descriptor_vector: TensorMap) -> TensorMap:
        return self.kernel_fn(descriptor_vector, self.sparse_points)


class GPR(torch.nn.Module):

    def __init__(
        self,
        atomic_types: List[int],
        kernel_calculator: callable,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Store attributes
        self.atomic_types = atomic_types
        self.kernel_calculator = kernel_calculator
        self.dtype = dtype
        self.device = device

        # Initialize a composition model
        self._set_composition_model()

    def _set_composition_model(self) -> None:
        # Initialize a CompositionModel. As this is imported from metatrain, this
        # requires creating some information on the dataset
        self.dataset_info = DatasetInfo(
            length_unit="",
            atomic_types=self.atomic_types,
            targets={
                "energy": TargetInfo(
                    quantity="energy",
                    unit="eV",
                    layout=TensorMap(
                        Labels.single(),
                        [
                            TensorBlock(
                                samples=Labels(
                                    ["system"],
                                    torch.tensor([], dtype=torch.int64).reshape(0, 1),
                                ),
                                components=[],
                                properties=Labels(
                                    ["energy"],
                                    torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
                                ),
                                values=torch.zeros((0, 1), dtype=self.dtype),
                            )
                        ],
                    ),
                )
            },
        )
        self.composition_model = CompositionModel({}, dataset_info=self.dataset_info)

    def fit(
        self,
        systems,
        system_id: List[int],
        k_nm: TensorMap,
        targets: Dict[str, TensorMap],
        alpha_energy: Union[float, torch.Tensor],
        alpha_energy_grad: Union[float, torch.Tensor],
    ) -> None:
        """
        Fits the GPR model.
        """

        # Copy the targets to avoid overwriting data
        targets = {k: v.copy() for k, v in targets.items()}

        # Train on energies. Gradients of energies (negative forces) are handled
        # automatically.
        assert "energy" in targets
        assert len(targets) == 1

        # Homoscedastic noise if passed as a float, heteroscedastic implied otherwise.
        if isinstance(alpha_energy, float):  # homoscedastic
            alpha_energy = torch.ones(len(systems)) * alpha_energy
        assert alpha_energy.shape == torch.Size([len(systems)]), (
            "If passing ``alpha_energy`` as a tensor, it should have shape"
            " (N_sys,), where N_sys are the number of systems in ``systems``"
        )

        if isinstance(alpha_energy_grad, float):  # homoscedastic
            alpha_energy_grad = (
                torch.ones(sum([len(sys) for sys in systems]), 3) * alpha_energy_grad
            )
        assert alpha_energy_grad.shape == torch.Size(
            [sum([len(sys) for sys in systems]), 3]
        ), (
            "If passing ``alpha_energy_grad`` as a tensor, it should have shape"
            " (N_at, 3), where N_at are the number of atomic environments in"
            " ``systems``"
        )

        # Fit the composition model. As this is imported from metatrain, this requires
        # creating a temporary dataset of the systems and targets
        self.composition_model.train_model(
            Dataset(
                system=systems,
                energy=[
                    mts.slice(
                        targets["energy"],
                        "samples",
                        Labels(["system"], torch.tensor([A]).reshape(-1, 1)),
                    )
                    for A in system_id
                ],
            ),
            [],
        )

        # Remove additives from training data
        targets_baselined = remove_additive(
            systems,
            {"energy": mts.remove_gradients(targets["energy"].copy())},
            self.composition_model,
            self.dataset_info.targets,
        )

        # Extract the kernel and energy values from the TensorMap
        k_nm_vals = k_nm[0].values[:]
        energy_vals = targets_baselined["energy"][0].values[:]

        # Normalize kernel and targets per structure
        atoms_per_structure = torch.sqrt(torch.tensor([len(s) for s in systems]))
        k_nm_vals[:] *= atoms_per_structure[:, None]
        energy_vals[:] *= atoms_per_structure[:, None]

        # Regularize kernel and targets per structure
        k_nm_vals[:] /= alpha_energy[:, None]
        energy_vals[:] /= alpha_energy[:, None]

        # Handle energy gradients, if present
        if len(k_nm[0].gradients_list()) > 0:

            # Extract the gradient values
            k_nm_grad_vals = k_nm[0].gradient("positions").values
            energy_grad_vals = targets["energy"][0].gradient("positions").values

            # Regularize
            k_nm_grad_vals[:] /= alpha_energy_grad[:, :, None]
            energy_grad_vals[:] /= alpha_energy_grad[:, :, None]

            # Stack the kernel gradients to the kernel, and the energy gradients
            # (negative of the forces) to the energies
            k_nm_vals = torch.vstack(
                [
                    k_nm_vals,
                    k_nm_grad_vals.reshape(
                        k_nm_grad_vals.shape[0] * 3, k_nm_grad_vals.shape[2]
                    ),
                ]
            )
            energy_vals = torch.vstack(
                [
                    energy_vals,
                    energy_grad_vals.reshape(
                        energy_grad_vals.shape[0] * 3, energy_grad_vals.shape[2]
                    ),
                ]
            )

        # Now fit the GPR model
        self.kernel_calculator.solver.fit(
            k_nm_vals, energy_vals.reshape(-1),
        )

        # Get the weights
        weight_block = TensorBlock(
            values=self.kernel_calculator.solver.weights.T,
            samples=targets["energy"][0].properties,
            components=k_nm[0].components,
            properties=k_nm[0].properties,
        )

        self.weights = {"energy": TensorMap(targets["energy"].keys, [weight_block])}

    def forward(
        self,
        systems,
        system_id: List[int],
        k_tm: TensorMap,
        predict_gradients: bool,
        predict_std_energy: bool,
        predict_std_energy_grad: bool,
        k_tt: TensorMap = None,
    ) -> Dict[str, TensorMap]:
        """"""

        # Predict the energy
        if len(k_tm[0].gradients_list()) > 0:
            k_tm_ = mts.remove_gradients(k_tm)
        predictions = {
            "energy": mts.sum_over_samples(
                mts.dot(k_tm_, self.weights["energy"]), ["atom", "center_type"]
            )
        }

        # Predict the additive contributions
        outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=False,
            )
            for key, value in self.dataset_info.targets.items()
        }
        outputs_for_additive_model = {}
        for name, output in outputs.items():
            if name in self.composition_model.outputs:
                outputs_for_additive_model[name] = output

        additive_contributions = self.composition_model(
            systems,
            outputs,
            selected_atoms=None,
        )

        # Sum the predictions and the additive contributions
        predictions["energy"] = mts.add(
            # mts.remove_gradients(predictions["energy"]),
            predictions["energy"],
            additive_contributions["energy"],
        )

        # Reindex the "system" dimension
        predictions["energy"] = reindex_tensormap(predictions["energy"], system_id)

        # Compute the gradient of the energy by backpropagation and store in a TensorBlock
        if predict_gradients:
            e_grad_vals = []
            for system, energy in zip(systems, predictions["energy"][0].values):
                energy.backward(retain_graph=True)
                e_grad_vals.append(system.positions.grad)
            e_grad_vals = torch.vstack(e_grad_vals)
            e_grad_block = build_energy_gradient_block(systems, system_id, e_grad_vals)

            # Create a new TensorMap with the energy gradients stored as a gradient
            # TensorBlock
            energy_block = predictions["energy"][0].copy()
            energy_block.add_gradient("positions", e_grad_block)
            predictions["energy"] = TensorMap(Labels.single(), [energy_block])

        if predict_std_energy:
            # Compute energy variances
            if k_tt is None:
                var_energy = self.kernel_calculator.solver.predict_variance(k_tm[0].values, None)
            else:
                var_energy = self.kernel_calculator.solver.predict_variance(
                    k_tm[0].values, k_tt[0].values
                )

            # Create a TensorMap for the per-atom variances
            var_energy_per_atom = TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        samples=k_tm[0].samples,
                        components=[],
                        properties=Labels(["variance"], torch.arange(1).reshape(-1, 1)),
                        values=var_energy.reshape(-1, 1),
                    )
                ]
            )

            # Reduce over atomic envs and sqrt
            var_energy_per_structure = mts.sum_over_samples(
                var_energy_per_atom, ["atom", "center_type"]
            )
            std_energy_per_structure = mts.pow(var_energy_per_structure, 0.5)
            predictions["std_energy"] = std_energy_per_structure

        else:
            predictions["std_energy"] = None

        # Predictive variance for gradients
        if predict_gradients and predict_std_energy_grad:
            kernel_grad_vals = k_tm[0].gradient("positions").values
            n_samples, n_components, n_properties = kernel_grad_vals.shape
            assert n_components == 3
            kernel_grad_vals = kernel_grad_vals.reshape(
                n_samples * 3, n_properties
            )
            var_energy_grad = self.kernel_calculator.solver.predict_variance(
                kernel_grad_vals
            )
            print(var_energy_grad.shape)

            # Create a TensorMap for the per-atom variances
            var_energy_grad_per_atom = TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        samples=k_tm[0].gradient("positions").samples,
                        components=k_tm[0].gradient("positions").components,
                        properties=Labels(["variance"], torch.arange(1).reshape(-1, 1)),
                        values=var_energy_grad.reshape(n_samples, 3, 1),
                    )
                ]
            )

            # Reduce over atomic envs and sqrt
            var_energy_grad_per_structure = mts.sum_over_samples(
                var_energy_grad_per_atom, ["sample"]
            )
            std_energy_grad_per_structure = mts.pow(var_energy_grad_per_structure, 0.5)
            predictions["std_energy_grad"] = std_energy_grad_per_structure

        else:
            predictions["std_energy_grad"] = None

        return predictions


class _SorKernelSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the subset of regressors (SoR) problem::

    .. math::

        w = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM

    :param KMM:
        KNM matrix

    The function solve the linear problem with
    the RKHS-QR method.

    RKHS: Compute first the reproducing kernel features by diagonalizing K_MM and
          computing `P_NM = K_NM @ U_MM @ Lam_MM^(-1.2)` and then solves the linear
          problem for those (which is usually better conditioned)::

              (P_NM.T@P_NM + 1)^(-1) P_NM.T@Y
    Reference
    ---------
    Foster, L., Waagen, A., Aijaz, N., Hurley, M., Luis, A., Rinsky, J., ... &
    Srivastava, A. (2009). Stable and Efficient Gaussian Process Calculations. Journal
    of Machine Learning Research, 10(4).
    """

    def __init__(
        self,
        KMM: np.ndarray,
    ):
        self._KMM = KMM
        self._nM = len(KMM)
        self._vk, self._Uk = scipy.linalg.eigh(KMM)
        self._vk = self._vk[::-1]
        self._Uk = self._Uk[:, ::-1]

        self._nM = len(np.where(self._vk > 0)[0])
        self._PKPhi = self._Uk[:, : self._nM] * 1 / np.sqrt(self._vk[: self._nM])

    def fit(
        self,
        KNM: Union[torch.Tensor, np.ndarray],
        Y: Union[torch.Tensor, np.ndarray],
    ) -> None:
        # Convert to numpy arrays if passed as torch tensors for the solver
        if isinstance(KNM, torch.Tensor):
            weights_to_torch = True
            dtype = KNM.dtype
            device = KNM.device
            KNM = KNM.detach().numpy()
            assert isinstance(Y, torch.Tensor), "must pass `KNM` and `Y` as same type."
            Y = Y.detach().numpy()
        else:
            weights_to_torch = False

        # Broadcast Y for shape
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]

        # Solve with the RKHS-QR method
        A = np.vstack([KNM @ self._PKPhi, np.eye(self._nM)])  # \tilde{\Phi}_nm \oplus I
        Q, R = np.linalg.qr(A)

        weights = self._PKPhi @ scipy.linalg.solve_triangular(
            R, Q.T @ np.vstack([Y, np.zeros((self._nM, Y.shape[1]))])
        )

        # Store weights as torch tensors
        if weights_to_torch:
            weights = torch.tensor(weights, dtype=dtype, device=device)
        self._weights = weights

        # Store the inverse of the kernel matrix, used for predictive stdev
        # self._A_inv = torch.tensor(
        #     np.linalg.inv(np.add((KNM.T @ KNM), self._KMM)),
        #     dtype=dtype,
        #     device=device,
        # )
        # Compute A_inv using EVD
        A = np.add(self._KMM, KNM.T @ KNM)

        # Eigenvalue decomposition: A = U @ Λ @ U.T
        eigvals, eigvecs = scipy.linalg.eigh(A)

        # Sort eigenvalues in descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Identify positive eigenvalues for inversion
        positive_mask = eigvals > 0
        eigvals_filtered = eigvals[positive_mask]
        eigvecs_filtered = eigvecs[:, positive_mask]

        jitter = 1e-10

        # # Regularize and invert eigenvalues
        eigvals_inv = 1.0 / (eigvals_filtered + jitter)  # Add jitter for numerical stability

        # # Compute A_inv using reconstructed eigenvalues
        self._A_inv  = torch.tensor(
            eigvecs_filtered @ np.diag(eigvals_inv) @ eigvecs_filtered.T,
            dtype=dtype,
            device=device,
        )

    @property
    def weights(self):
        return self._weights

    def predict(self, KTM):
        return KTM @ self._weights
    
    def predict_variance(self, KTM: torch.Tensor, KTT: torch.Tensor) -> torch.Tensor:
        """
        Computes the predictive variance of the GPR model.
        """
        # Compute variances
        if KTT is None:
            return torch.diag(torch.sqrt(KTM @ self._A_inv @ KTM.T))
        else:
            return torch.diag(torch.sqrt(KTT - KTM @ self._A_inv @ KTM.T))



def reindex_tensormap(
    tensor: TensorMap,
    system_ids: List[int],
) -> TensorMap:
    """
    Takes a single TensorMap `tensor` containing data on multiple systems and re-indexes
    the "system" dimension of the samples. Assumes input has numeric system indices from
    {0, ..., N_system - 1} (inclusive), and maps these indices one-to-one with those
    passed in ``system_ids``.
    """
    assert tensor.sample_names[0] == "system"

    index_mapping = {i: A for i, A in enumerate(system_ids)}

    def new_row(row):
        return [index_mapping[row[0].item()]] + [i for i in row[1:]]

    new_blocks = []
    for block in tensor.blocks():
        new_samples = mts.Labels(
            names=block.samples.names,
            values=torch.tensor(
                [new_row(row) for row in block.samples.values],
                dtype=torch.int32,
            ),
        )
        new_block = mts.TensorBlock(
            values=block.values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return mts.TensorMap(tensor.keys, new_blocks)


# ===== Functions for parsing energies and energy gradients from xyz into metatensor format ===== #


def get_targets_from_xyz(
    frames: List[ase.Atoms],
    frame_idxs: List[int],
    dtype: torch.dtype,
    device: str,
    use_gradients: bool = True,
    energy_key: str = "energy",
    force_key: str = "forces",
) -> TensorMap:
    """
    Parses the energy (under the xyz key ``energy_key``) for each frame in ``frames``
    and returns it in a TensorMap, with each system indexed by its respective index in
    ``frame_idxs``.

    If forces (under the xyz key ``force_key``) are present in the `arrays` attribute of
    the passed ``frames``, the energy gradients (i.e. negative forces) are stored as
    gradient TensorBlocks associated to the energy TensorBlock, and returned in the
    final targets TensorMap.
    """

    # First parse the energy into a TensorBlock
    energy = TensorBlock(
        samples=Labels(["system"], torch.tensor(frame_idxs).reshape(-1, 1)),
        components=[],
        properties=Labels(["energy"], torch.arange(1).reshape(-1, 1)),
        values=torch.tensor(
            [frame.info[energy_key] for frame in frames],
            dtype=torch.float64,
            device=device,
        ).reshape(-1, 1),
    )

    # Next add the energy gradients as TensorBlock gradients if present
    if use_gradients:
        assert frames[0].arrays.get(force_key) is not None
        assert all([frame.arrays.get(force_key) is not None for frame in frames])

        # Parse the gradients of the energy into a TensorBlock. These are just the
        # negative forces.
        e_grad_vals = torch.vstack(
            [
                torch.tensor(-1 * frame.arrays[force_key], dtype=dtype, device=device)
                for frame in frames
            ],
        )
        e_grad_block = build_energy_gradient_block(frames, frame_idxs, e_grad_vals)
        energy.add_gradient("positions", e_grad_block)

    # Now store the energy TensorBlock (with associated energy gradients if present) in
    # a TensorMap
    targets = TensorMap(Labels.single(), [energy])

    return targets


def build_energy_gradient_block(
    systems, system_id: List[int], e_grad_vals: torch.Tensor
) -> TensorBlock:
    """
    Builds a TensorBlock for storing energy gradient values.

    Expects ``e_grad_vals`` to be of shape (n_systems, n_atoms, 3).
    """
    # Create the indices that map the per-atom gradient sample back to the per-structure
    # energy sample
    sample_mapping = {A: i for i, A in enumerate(system_id)}
    samples_values = torch.tensor(
        [
            [sample_mapping[A], A, i]
            for A, sys in zip(system_id, systems) for i in range(len(sys))
        ]
    )

    return TensorBlock(
        samples=Labels(["sample", "system", "atom"], samples_values),
        components=[
            Labels(["xyz"], torch.tensor([0, 1, 2]).reshape(-1, 1)),
        ],
        properties=Labels(["energy"], torch.arange(1).reshape(-1, 1)),
        values=e_grad_vals.reshape(-1, 3, 1),
    )
