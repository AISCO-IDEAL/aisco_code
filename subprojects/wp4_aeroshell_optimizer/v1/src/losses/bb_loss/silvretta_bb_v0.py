import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.s_d_ffrom_mesh import SDFfromMesh
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import fetch_bounding_box_ply


class SilvrettaBbV0(ParentLoss):
    def _setup(self) -> None:
        self.__config = self.losses_config["bb_config"]
        self.__inner_ply_location = self.losses_config["bb_config"]["inner_ply_location"]
        self.__outer_ply_location = self.losses_config["bb_config"]["outer_ply_location"]
        self.__device = self.losses_config["bb_config"].get("sdf_device", "cuda")  # Default to 'cuda' if not specified
        # Create SDFs for inner and outer boxes
        inner_mesh, outer_mesh = fetch_bounding_box_ply(
            inner_ply_location=self.__inner_ply_location, outer_ply_location=self.__outer_ply_location
        )
        self.__inner_SDF: SDFfromMesh = SDFfromMesh(
            mesh=inner_mesh, pointgrid_config=self.__config["pointgrid_params"], sdf_device=self.__device
        )
        self.__outer_SDF: SDFfromMesh = SDFfromMesh(
            mesh=outer_mesh, pointgrid_config=self.__config["pointgrid_params"], sdf_device=self.__device
        )
        self.__pointgrid: list[list[float]] = self.__inner_SDF.pointgrid()
        self.__inner_sdf_values: list[float] = self.__inner_SDF.sdf_at_pointgrid()
        self.__outer_sdf_values: list[float] = self.__outer_SDF.sdf_at_pointgrid()

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network                                              NO Default!!
        Returns:
            Bounding boxes loss (to be minimized)

        Constrains model SDF to lie between outer and inner SDFs:
            sdf_outer(p) <= sdf_model(p) <= sdf_inner(p)
        Loss = mean( relu(sdf_model - sdf_inner) + relu(sdf_outer - sdf_model) )
        """

        with torch.enable_grad():
            # Automatically find where the model is (CPU or GPU)
            target_device = next(model.parameters()).device
            self.device = target_device

            P = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.device)  # [N,3]
            sdf_inner_t = torch.as_tensor(
                self.__inner_sdf_values,
                dtype=torch.float32,
                device=self.device,
            )  # [N]
            sdf_outer_t = torch.as_tensor(self.__outer_sdf_values, dtype=torch.float32, device=self.device)  # [N]

            # Batch forward
            output = model(P)
            sdf_pred = output[0] if isinstance(output, tuple) else output

            sdf_pred = sdf_pred.squeeze(-1)  # [N]

            # Box constraints: clamp violations with ReLU
            upper_violation = torch.relu(sdf_pred - sdf_inner_t)
            lower_violation = torch.relu(sdf_outer_t - sdf_pred)

            n_points = sdf_pred.shape[0]
            total_violation = upper_violation + lower_violation
            loss = total_violation.mean() if n_points > 0 else torch.tensor(0.0, device=self.device)

            # Store diagnostics for DebugManager
            with torch.no_grad():
                self._last_diagnostics = {
                    "n_points": n_points,
                    "sdf_pred_min": sdf_pred.min().item(),
                    "sdf_pred_max": sdf_pred.max().item(),
                    "sdf_pred_mean": sdf_pred.mean().item(),
                    "sdf_inner_min": sdf_inner_t.min().item(),
                    "sdf_inner_max": sdf_inner_t.max().item(),
                    "sdf_outer_min": sdf_outer_t.min().item(),
                    "sdf_outer_max": sdf_outer_t.max().item(),
                    "upper_violation_count": int((upper_violation > 0).sum().item()),
                    "lower_violation_count": int((lower_violation > 0).sum().item()),
                    "upper_violation_max": upper_violation.max().item(),
                    "lower_violation_max": lower_violation.max().item(),
                    "upper_violation_mean": upper_violation.mean().item() if n_points > 0 else 0.0,
                    "lower_violation_mean": lower_violation.mean().item() if n_points > 0 else 0.0,
                    "violation_frac": float((total_violation > 0).sum().item()) / max(n_points, 1),
                }

            return loss
