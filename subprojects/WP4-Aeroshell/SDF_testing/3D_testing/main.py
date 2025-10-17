
import torch
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Import our custom modules
from sdf import SDF3DNet
from losses import SDF3DLossCalculator
from visualization import SDF3DVisualizer

class SDF3DTrainer:
    """
    Main trainer class for 3D SDF shape optimization.

    This class handles the complete training pipeline including:
    - Network initialization
    - Loss computation and optimization
    - Progress monitoring
    - Visualization and results saving
    """

    def __init__(self, device=None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize network
        self.sdf_net = SDF3DNet(
            in_dim=3, 
            hidden=128, 
            layers=6, 
            use_skip_connection=True
        ).to(self.device)

        # Initialize loss calculator and visualizer
        self.loss_calculator = SDF3DLossCalculator(device=self.device)
        self.visualizer = SDF3DVisualizer(device=self.device)

        # Define optimization bounds
        # Outer bounds: vehicle cannot exceed these limits
        self.outer_bounds = (
            torch.tensor([-1.8, -1.8, -0.8], device=self.device),  # min (x, y, z)
            torch.tensor([1.8, 1.8, 0.8], device=self.device)       # max (x, y, z)
        )

        # Inner bounds: vehicle must fill at least this much space  
        self.inner_bounds = (
            torch.tensor([-0.5, -0.5, -0.2], device=self.device),   # min (x, y, z)
            torch.tensor([0.5, 0.5, 0.2], device=self.device)        # max (x, y, z)
        )

        # Training parameters
        self.learning_rate = 0.001
        self.num_iterations = 1000
        self.sample_points_per_iter = 2048

        # Initialize empty loss history
        self.loss_history = {}

        print("SDF3D Trainer initialized successfully!")

    def generate_sample_points(self, n_points=2048):
        """
        Generate random sample points for training.

        Args:
            n_points: Number of points to generate

        Returns:
            Random 3D points tensor
        """
        # Sample points in an extended region around the shape
        points = torch.rand(n_points, 3, device=self.device) * 4.0 - 2.0  # Range [-2, 2]
        return points

    def train(self, save_checkpoints=True, checkpoint_interval=500):
        """
        Main training loop for SDF optimization.

        Args:
            save_checkpoints: Whether to save model checkpoints
            checkpoint_interval: Frequency of checkpoint saving
        """
        print("\nStarting SDF3D Shape Optimization...")
        print(f"Target: Minimize frontal area, maximize top area")
        print(f"Constraints: Outer bounds {self.outer_bounds}, Inner bounds {self.inner_bounds}")
        print(f"Training for {self.num_iterations} iterations...")

        # Initialize optimizer
        optimizer = optim.Adam(self.sdf_net.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iterations)

        start_time = time.time()
        best_loss = float('inf')

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Generate sample points for this iteration
            sample_points = self.generate_sample_points(self.sample_points_per_iter)

            # Compute all loss components
            losses = self.loss_calculator.compute_total_loss(
                self.sdf_net, 
                sample_points, 
                self.outer_bounds, 
                self.inner_bounds
            )

            # Backward pass and optimization
            losses['total'].backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.sdf_net.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Store loss history
            for loss_name, loss_value in losses.items():
                if loss_name not in self.loss_history:
                    self.loss_history[loss_name] = []
                self.loss_history[loss_name].append(loss_value.item() if torch.is_tensor(loss_value) else loss_value)

            # Progress reporting
            if iteration % 100 == 0 or iteration == self.num_iterations - 1:
                elapsed_time = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]

                print(f"\nIteration {iteration:4d}/{self.num_iterations}")
                print(f"Time: {elapsed_time:.1f}s, LR: {current_lr:.6f}")
                print(f"Total Loss: {losses['total'].item():.6f}")
                print(f"  Eikonal: {losses['eikonal'].item():.6f}")
                print(f"  Size Constraint: {losses['size_constraint'].item():.6f}")
                print(f"  Frontal Area: {losses['frontal_area'].item():.6f}")
                print(f"  Top Area: {losses['top_area'].item():.6f}")
                print(f"  Smoothness: {losses['smoothness'].item():.6f}")

            # Save best model
            if losses['total'].item() < best_loss:
                best_loss = losses['total'].item()
                if save_checkpoints:
                    self.save_model('best_model.pth')

            # Periodic checkpoints
            if save_checkpoints and iteration % checkpoint_interval == 0 and iteration > 0:
                checkpoint_path = f'checkpoint_iter_{iteration}.pth'
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

            # Adaptive loss weight adjustment (optional)
            if iteration == self.num_iterations // 4:
                # Increase emphasis on area optimization after initial shape formation
                self.loss_calculator.update_weights({
                    'frontal_area': 8.0,
                    'top_area': -5.0
                })
                print("Updated loss weights for area optimization focus")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        print(f"Best loss achieved: {best_loss:.6f}")

        # Save final model
        if save_checkpoints:
            self.save_model('final_model.pth')

    def save_model(self, filename):
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.sdf_net.state_dict(),
            'loss_history': self.loss_history,
            'outer_bounds': self.outer_bounds,
            'inner_bounds': self.inner_bounds
        }, filename)

    def load_model(self, filename):
        """Load model state dict"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.sdf_net.load_state_dict(checkpoint['model_state_dict'])
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        print(f"Model loaded from {filename}")

    def visualize_results(self, save_plots=True):
        """
        Generate comprehensive visualizations of the optimization results.

        Args:
            save_plots: Whether to save plot files
        """
        print("\nGenerating visualizations...")

        # Set network to evaluation mode
        self.sdf_net.eval()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"sdf_results_{timestamp}")
        if save_plots:
            output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Plot loss curves
        loss_plot_path = (output_dir / "loss_curves.png") if save_plots else None
        self.visualizer.plot_loss_curves(self.loss_history, save_path=loss_plot_path)

        # 2. Cross-sectional views
        cross_section_path = (output_dir / "cross_sections.png") if save_plots else None
        self.visualizer.plot_cross_sections(self.sdf_net, save_path=cross_section_path)

        # 3. 3D surface visualization (matplotlib)
        surface_3d_path = (output_dir / "surface_3d.png") if save_plots else None
        self.visualizer.plot_3d_surface_matplotlib(self.sdf_net, save_path=surface_3d_path)

        # 4. Interactive 3D visualization (plotly)
        interactive_3d_path = (output_dir / "interactive_3d.html") if save_plots else None
        self.visualizer.plot_3d_surface_plotly(self.sdf_net, save_path=interactive_3d_path)

        # 5. Compute and display final metrics
        metrics = self.visualizer.compute_final_metrics(self.sdf_net)

        if save_plots:
            # Save metrics to text file
            metrics_path = output_dir / "final_metrics.txt"
            
            # Prepare metrics text
            metrics_text = ["=== Final Shape Optimization Results ===\n"]
            metrics_text.extend(f"{key}: {value}\n" for key, value in metrics.items())
            metrics_text.append("\n=== Training Configuration ===\n")
            metrics_text.extend([
                f"Iterations: {self.num_iterations}\n",
                f"Learning Rate: {self.learning_rate}\n",
                f"Sample Points per Iteration: {self.sample_points_per_iter}\n",
                f"Network Architecture: {self.sdf_net.hidden} hidden units, {self.sdf_net.layers} layers\n",
                f"Final Loss: {self.loss_history['total'][-1]:.6f}\n"
            ])
            
            # Write all lines at once
            metrics_path.write_text("".join(metrics_text))

            print(f"\nAll results saved to directory: {output_dir}")
        return metrics

def main():
    """Main function to run the complete SDF optimization pipeline"""
    print("=== 3D SDF Vehicle Shape Optimization ===")
    print("Objective: Minimize frontal area (drag) while maximizing top area (solar collection)")
    print()

    # Initialize trainer
    trainer = SDF3DTrainer()

    # Run training
    trainer.train(save_checkpoints=True)

    # Generate visualizations and results
    metrics = trainer.visualize_results(save_plots=True)

    print("\n=== Optimization Complete ===")
    print("Check the generated plots and results directory for detailed analysis.")

    return trainer, metrics

if __name__ == "__main__":
    # Run the complete pipeline
    trainer, final_metrics = main()
