import contextlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter


############################### 3D plotting   ###############################
def plot_3d_comparison_pyvista(
    coords: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    save_path: str,
    variable_name: str = "Value",
    point_size: float = 5.0,
    cmap: str = "jet",
    window_size: tuple[int, int] = (1800, 600),
):
    """
    Generates a 3-panel 3D point cloud visualization comparing ground truth,
    prediction, and their absolute difference using PyVista.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the points, shape (N, 3).
    u_gtr : np.ndarray
        Ground truth scalar values at each point, shape (N,).
    u_prd : np.ndarray
        Predicted scalar values at each point, shape (N,).
    save_path : str
        Path to save the visualization (e.g., 'comparison.png').
    variable_name : str, optional
        Name of the variable being plotted, used for titles and color bars.
        Defaults to "Value".
    point_size : float, optional
        Size of the points in the plot. Defaults to 5.0.
    cmap : str, optional
        Matplotlib colormap name to use. Defaults to "jet".
    window_size : Tuple[int, int], optional
        Window size (width, height) for the saved image. Defaults to (1800, 600).
    """
    try:
        # Ensure data is flat
        if u_gtr.ndim > 1:
            u_gtr = u_gtr.squeeze()
        if u_prd.ndim > 1:
            u_prd = u_prd.squeeze()
        assert u_gtr.ndim == 1, "Ground truth data must be 1D"
        assert u_prd.ndim == 1, "Predicted data must be 1D"
        assert coords.shape[0] == u_gtr.shape[0] == u_prd.shape[0], "Shapes mismatch"
        assert coords.shape[1] == 3, "Coordinates must be 3D"

        # Calculate absolute difference
        u_diff = np.abs(u_gtr - u_prd)

        # Create PyVista PolyData objects
        cloud_gtr = pv.PolyData(coords)
        cloud_gtr.point_data[f"Ground Truth {variable_name}"] = u_gtr

        cloud_prd = pv.PolyData(coords)
        cloud_prd.point_data[f"Prediction {variable_name}"] = u_prd

        cloud_diff = pv.PolyData(coords)
        cloud_diff.point_data["Absolute Difference"] = u_diff

        # Determine shared color limits for GTR and PRD for better comparison
        common_min = min(np.min(u_gtr), np.min(u_prd))
        common_max = max(np.max(u_gtr), np.max(u_prd))
        clim_common = [common_min, common_max]
        clim_diff = [np.min(u_diff), np.max(u_diff)]

        # Set up the plotter with 3 subplots
        plotter = pv.Plotter(
            shape=(1, 3), window_size=window_size, off_screen=True, border=False
        )  # off_screen=True for saving without showing

        # Subplot 1: Ground Truth
        plotter.subplot(0, 0)
        plotter.add_text(f"Ground Truth ({variable_name})", position="upper_edge", font_size=10)
        plotter.add_mesh(
            cloud_gtr,
            scalars=f"Ground Truth {variable_name}",
            render_points_as_spheres=True,
            point_size=point_size,
            cmap=cmap,
            clim=clim_common,
            scalar_bar_args={"title": f"GT {variable_name}"},
        )
        plotter.view_isometric()  # Set a standard view

        # Subplot 2: Prediction
        plotter.subplot(0, 1)
        plotter.add_text(f"Prediction ({variable_name})", position="upper_edge", font_size=10)
        plotter.add_mesh(
            cloud_prd,
            scalars=f"Prediction {variable_name}",
            render_points_as_spheres=True,
            point_size=point_size,
            cmap=cmap,
            clim=clim_common,
            scalar_bar_args={"title": f"Pred {variable_name}"},
        )
        plotter.view_isometric()

        # Subplot 3: Difference
        plotter.subplot(0, 2)
        plotter.add_text("Absolute Difference", position="upper_edge", font_size=10)
        plotter.add_mesh(
            cloud_diff,
            scalars="Absolute Difference",
            render_points_as_spheres=True,
            point_size=point_size,
            cmap="Reds",
            clim=clim_diff,  # Use a different cmap for difference
            scalar_bar_args={"title": "Abs. Diff."},
        )
        plotter.view_isometric()

        # Link cameras
        plotter.link_views()

        # Save the screenshot
        plotter.screenshot(save_path, transparent_background=False)
        print(f"Saved 3D comparison plot to {save_path}")

    except ImportError:
        print("PyVista is not installed. Skipping 3D visualization.")
        print("Install it using: pip install pyvista")
    except Exception as e:
        print(f"An error occurred during 3D plotting: {e}")
    finally:
        if "plotter" in locals():
            plotter.close()  # Ensure plotter is closed to free memory


def plot_3d_comparison_matplotlib(  # noqa: PLR0915, PLR0912
    coords: np.ndarray,
    u_gtr: np.ndarray | None,
    u_prd: np.ndarray,
    save_path: str,
    variable_name: str = "Value",
    point_size: float = 3.0,
    cmap: str = "jet",
    dpi: int = 300,
    view_angle: tuple[float, float] = (20, -120),
    hide_grid: bool = False,
    equal_aspect: bool = True,  # <-- Control aspect ratio setting
    show_abs_diff: bool = False,  # absolute difference
    show_input_shape: bool = True,
    use_ground_truth_range: bool = False,
    fixed_clim_common: tuple[float, float] | None = None,
    fixed_clim_diff: tuple[float, float] | None = None,
    fixed_axis_limits: dict[str, tuple[float, float]] | None = None,
):
    """
    Generates a 3/4-panel 3D scatter plot comparison using Matplotlib.
    When ``u_gtr`` is ``None``, only the prediction panel is shown.

    Parameters:
    coords : np.ndarray
        Coordinates of the points, shape (N, 3).
    u_gtr : np.ndarray
        Ground truth scalar values at each point, shape (N,).
    u_prd : np.ndarray
        Predicted scalar values at each point, shape (N,).
    save_path : str
        Path to save the visualization (e.g., 'comparison.png').
    variable_name : str, optional
        Name of the variable being plotted, used for titles and color bars.
        Defaults to "Value".
    point_size : float, optional
        Size of the points in the scatter plot (marker size). Defaults to 5.0.
    cmap : str, optional
        Matplotlib colormap name to use for GTR and PRD. Defaults to "jet".
    dpi : int, optional
        Dots per inch for the saved image resolution. Defaults to 150.
    view_angle : Tuple[float, float], optional
        The elevation and azimuth angles for the 3D view. Defaults to (30, -60).
    hide_grid: bool, optional
        Whether to show the background grid and axis.
    equal_aspect: bool, optional
        If True, attempts to set axis limits to achieve visually equal scaling.
        Defaults to True.  # Set to False to control the visualised grid
    show_abs_diff: bool, optional
        If True, show absolute difference subplot. Defaults to True.
    show_input_shape: bool, optional
        If True, include the input-shape subplot. Defaults to True.
    use_ground_truth_range: bool, optional
        If True, use only the GT min/max as shared color limits for GT and Pred.
        Defaults to False.
    fixed_clim_common: Tuple[float, float] | None, optional
        If set, force a fixed [vmin, vmax] for GT and prediction panels.
        Overrides use_ground_truth_range.
    fixed_clim_diff: Tuple[float, float] | None, optional
        If set, force a fixed [vmin, vmax] for the absolute-difference panel.
    fixed_axis_limits: Dict[str, Tuple[float, float]] | None, optional
        If set, enforce fixed axis limits with keys: "x", "y", "z".
    """
    try:
        coords = np.asarray(coords)
        u_prd = np.asarray(u_prd).squeeze()
        has_ground_truth = u_gtr is not None
        if has_ground_truth:
            u_gtr = np.asarray(u_gtr).squeeze()
            assert u_gtr.ndim == 1, "Ground truth data must be 1D"
        assert u_prd.ndim == 1, "Predicted data must be 1D"
        if has_ground_truth:
            assert u_gtr is not None
            assert coords.shape[0] == u_gtr.shape[0] == u_prd.shape[0], "Shapes mismatch"
        else:
            assert coords.shape[0] == u_prd.shape[0], "Shapes mismatch"
        assert coords.shape[1] == 3, "Coordinates must be 3D"

        u_diff = np.abs(u_gtr - u_prd) if u_gtr is not None else None
        if fixed_clim_common is not None:
            common_min, common_max = float(fixed_clim_common[0]), float(fixed_clim_common[1])
        elif has_ground_truth and use_ground_truth_range:
            assert u_gtr is not None
            common_min = float(np.min(u_gtr))
            common_max = float(np.max(u_gtr))
        elif has_ground_truth:
            assert u_gtr is not None
            common_min = min(np.min(u_gtr), np.min(u_prd))
            common_max = max(np.max(u_gtr), np.max(u_prd))
        else:
            common_min = float(np.min(u_prd))
            common_max = float(np.max(u_prd))
        clim_common = [common_min, common_max]
        if fixed_clim_diff is not None:
            clim_diff = [float(fixed_clim_diff[0]), float(fixed_clim_diff[1])]
        elif has_ground_truth and u_diff is not None:
            clim_diff = [np.min(u_diff), np.max(u_diff)]
        else:
            clim_diff = [0.0, 0.0]

        # Use one shared Normalize object so GT/Pred have exactly identical color mapping.
        shared_norm = mcolors.Normalize(vmin=clim_common[0], vmax=clim_common[1])
        diff_norm = mcolors.Normalize(vmin=clim_diff[0], vmax=clim_diff[1]) if has_ground_truth else None

        ncols = (
            (1 if show_input_shape else 0) + 1 + (1 if has_ground_truth else 0) + (1 if has_ground_truth and show_abs_diff else 0)
        )
        fig = plt.figure(figsize=(6 * ncols, 6))
        axes = []
        ax_idx = 1

        # Optional subplot 0: Input Shape
        if show_input_shape:
            ax0 = fig.add_subplot(1, ncols, ax_idx, projection="3d")
            ax0.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="gray", s=point_size, depthshade=False)
            ax0.set_title("Input Shape", fontsize=14)
            axes.append(ax0)
            ax_idx += 1

        ax_gtr = None
        if has_ground_truth:
            # Subplot 1: Ground Truth
            ax_gtr = fig.add_subplot(1, ncols, ax_idx, projection="3d")
            sc1 = ax_gtr.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=u_gtr,
                cmap=cmap,
                s=point_size,
                norm=shared_norm,
                depthshade=False,
            )
            ax_gtr.set_title("Ground-truth", fontsize=14)
            axes.append(ax_gtr)
            ax_idx += 1

        # Prediction panel
        ax2 = fig.add_subplot(1, ncols, ax_idx, projection="3d")
        sc2 = ax2.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=u_prd,
            cmap=cmap,
            s=point_size,
            norm=shared_norm,
            depthshade=False,
        )
        ax2.set_title("Model Estimate" if has_ground_truth else "Prediction", fontsize=14)
        axes.append(ax2)
        ax_idx += 1

        # Subplot 3: Difference (optional)
        if has_ground_truth and show_abs_diff:
            ax3 = fig.add_subplot(1, ncols, ax_idx, projection="3d")
            sc3 = ax3.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=u_diff,
                cmap="Reds",
                s=point_size,
                norm=diff_norm,
                depthshade=False,
            )
            ax3.set_title("Absolute Difference")
            axes.append(ax3)

        # Per-panel vertical colorbars (matching trainer screenshot style)
        # Build colorbars from explicit ScalarMappables so ranges stay fixed.
        common_mappable = cm.ScalarMappable(norm=shared_norm, cmap=cmap)
        common_mappable.set_array([])
        cb2 = fig.colorbar(
            common_mappable,
            ax=ax2,
            orientation="vertical",
            fraction=0.025,
            pad=0.04,
            ticks=np.linspace(clim_common[0], clim_common[1], 5),
        )
        cb2.ax.tick_params(labelsize=11, length=3)
        cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        cb2.ax.yaxis.get_offset_text().set_visible(False)
        plt.setp(cb2.outline, linewidth=0.8)

        if has_ground_truth:
            cb1 = fig.colorbar(
                common_mappable,
                ax=ax_gtr,
                orientation="vertical",
                fraction=0.025,
                pad=0.04,
                ticks=np.linspace(clim_common[0], clim_common[1], 5),
            )
            cb1.ax.tick_params(labelsize=11, length=3)
            cb1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            cb1.ax.yaxis.get_offset_text().set_visible(False)
            plt.setp(cb1.outline, linewidth=0.8)

        if has_ground_truth and show_abs_diff:
            diff_mappable = cm.ScalarMappable(norm=diff_norm, cmap="Reds")
            diff_mappable.set_array([])
            cb3 = fig.colorbar(
                diff_mappable,
                ax=ax3,
                orientation="vertical",
                fraction=0.025,
                pad=0.04,
                ticks=np.linspace(clim_diff[0], clim_diff[1], 5),
            )
            cb3.ax.tick_params(labelsize=11, length=3)
            cb3.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            cb3.ax.yaxis.get_offset_text().set_visible(False)
            plt.setp(cb3.outline, linewidth=0.8)

        # 统一设置
        for ax in axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # Show plain numeric ticks (no scientific offset text) to avoid
            # confusion when large configured axis limits are used.
            plain_fmt = ScalarFormatter(useOffset=False)
            plain_fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(plain_fmt)
            ax.yaxis.set_major_formatter(plain_fmt)
            ax.zaxis.set_major_formatter(plain_fmt)
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
            if fixed_axis_limits is not None:
                xlims = fixed_axis_limits["x"]
                ylims = fixed_axis_limits["y"]
                zlims = fixed_axis_limits["z"]
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.set_zlim(zlims)
                try:
                    ax.set_box_aspect(
                        (1, (ylims[1] - ylims[0]) / (xlims[1] - xlims[0]), (zlims[1] - zlims[0]) / (xlims[1] - xlims[0]))
                    )
                except AttributeError:
                    ax.set_aspect("auto")
            elif equal_aspect:
                x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                x_mid = (x_max + x_min) / 2.0
                y_mid = (y_max + y_min) / 2.0
                z_mid = (z_max + z_min) / 2.0
                max_range = max(x_range, y_range, z_range)
                plot_radius = max_range / 2.0 * 1.1
                xlims = (x_mid - plot_radius, x_mid + plot_radius)
                ylims = (y_mid - plot_radius, y_mid + plot_radius)
                zlims = (z_mid - plot_radius, z_mid + plot_radius)
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.set_zlim(zlims)
                try:
                    ax.set_box_aspect(
                        (1, (ylims[1] - ylims[0]) / (xlims[1] - xlims[0]), (zlims[1] - zlims[0]) / (xlims[1] - xlims[0]))
                    )
                except AttributeError:
                    ax.set_aspect("auto")
            if hide_grid:
                ax.grid(False)
                # 隐藏坐标轴刻度、标签、轴线
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_zlabel("")
                # 兼容不同matplotlib版本
                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    with contextlib.suppress(Exception):
                        axis.line.set_color((0, 0, 0, 0))
                # 隐藏3D坐标轴面
                try:
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor("w")
                    ax.yaxis.pane.set_edgecolor("w")
                    ax.zaxis.pane.set_edgecolor("w")
                except Exception:
                    pass

        fig.subplots_adjust(wspace=0.2)
        plt.tight_layout()
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved 3D comparison plot (matplotlib) to {save_path}")

    except Exception as e:
        print(f"An error occurred during Matplotlib 3D plotting: {e}")
    finally:
        if "fig" in locals():
            plt.close(fig)
