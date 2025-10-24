# 3D SDF Vehicle Shape Optimization

This project implements a neural signed distance field (SDF) approach for optimizing 3D vehicle shapes, specifically targeting aerodynamic efficiency and solar collection area for solar race cars.

## Overview

The system uses a neural network to represent a 3D shape as a signed distance field, then optimizes the shape to:
- **Minimize frontal area** (reduce aerodynamic drag)
- **Maximize top surface area** (increase solar panel area)
- **Satisfy size constraints** (fit within specified bounding boxes)
- **Maintain valid SDF properties** (Eikonal constraint)
- **Ensure smooth surfaces** (manufacturability)

## Files Description

### Core Modules
- `sdf_network.py` - 3D neural SDF network with skip connections and mixed activations
- `loss_functions.py` - Comprehensive loss functions for multi-objective optimization
- `visualization.py` - Advanced visualization tools for 3D shapes and training progress
- `main_training.py` - Complete training pipeline and experiment management

### Utility Files
- `demo.py` - Quick demo script for testing (reduced iterations)
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation file

## Installation

1. Install Python 3.7+ and pip
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo
For a fast test run (200 iterations):
```bash
python demo.py
```

### Full Optimization
For complete optimization (2000 iterations):
```bash
python main_training.py
```

### Custom Training
```python
from main_training import SDF3DTrainer

# Initialize trainer
trainer = SDF3DTrainer()

# Customize parameters
trainer.num_iterations = 1000
trainer.learning_rate = 0.002

# Adjust loss weights
trainer.loss_calculator.update_weights({
    'frontal_area': 10.0,  # Higher priority for drag reduction
    'top_area': -5.0       # Higher priority for solar area
})

# Run training
trainer.train()

# Generate results
metrics = trainer.visualize_results()
```

## Key Features

### Neural SDF Architecture
- **3D input coordinates** (x, y, z) → **signed distance output**
- **Skip connections** for better gradient flow
- **Mixed activations** (Tanh + LeakyReLU) for expressivity
- **Xavier initialization** for stable training

### Multi-Objective Optimization
1. **Eikonal Loss**: Ensures valid SDF properties (∇||SDF|| = 1)
2. **Size Constraints**: Enforces outer/inner bounding box limits
3. **Frontal Area**: Minimizes drag-relevant projected area
4. **Top Area**: Maximizes solar collection surface
5. **Smoothness**: Encourages manufacturable surfaces

### Advanced Visualization
- **3D surface rendering** with matplotlib and plotly
- **Cross-sectional analysis** (top, side, front views)
- **Area projections** for drag and solar analysis
- **Loss curve tracking** for all objectives
- **Interactive plots** for detailed inspection

## Optimization Strategy

The training process uses a multi-phase approach:

1. **Phase 1** (0-25% iterations): Focus on shape formation and constraints
2. **Phase 2** (25-100% iterations): Emphasis on area optimization
3. **Adaptive weighting**: Loss weights adjust during training
4. **Gradient clipping**: Ensures training stability
5. **Cosine annealing**: Learning rate scheduling

## Output Results

After training, the system generates:
- **Loss curves**: Training progress for all objectives
- **3D visualizations**: Interactive and static surface plots
- **Cross-sections**: Detailed 2D views of the optimized shape
- **Performance metrics**: Final area calculations and efficiency ratios
- **Model checkpoints**: Saved networks for later use

## Technical Details

### SDF Representation
The neural network learns to map 3D coordinates to signed distances:
- **Negative values**: Inside the vehicle shape
- **Positive values**: Outside the vehicle shape
- **Zero level set**: The vehicle surface

### Bounding Box Constraints
- **Outer bounds**: Maximum vehicle size limits
- **Inner bounds**: Minimum required volume
- **Soft constraints**: Penalize violations rather than hard limits

### Area Computation
- **Frontal area**: YZ projection at x=0 (drag-relevant)
- **Top area**: XY projection at z=0.5 (solar-relevant)
- **Smooth approximations**: Differentiable Heaviside functions

## Customization

### Adjusting Objectives
Modify loss weights in `loss_functions.py`:
```python
self.weights = {
    'eikonal': 1.0,
    'size_constraint': 10.0,
    'frontal_area': 5.0,    # Increase for more drag focus
    'top_area': -3.0,       # More negative for more solar focus
    'smoothness': 0.5
}
```

### Network Architecture
Modify network in `sdf_network.py`:
```python
sdf_net = SDF3DNet(
    hidden=256,              # Larger for more capacity
    layers=8,                # Deeper for more complexity
    use_skip_connection=True # Skip connections recommended
)
```

### Bounds and Constraints
Adjust bounding boxes in `main_training.py`:
```python
# Vehicle size limits
self.outer_bounds = (
    torch.tensor([-2.0, -2.0, -1.0]),  # Larger outer bounds
    torch.tensor([2.0, 2.0, 1.0])
)
```

## Performance Tips

1. **Use GPU**: Significant speedup on CUDA-capable devices
2. **Batch size**: Increase `sample_points_per_iter` for smoother gradients
3. **Learning rate**: Start with 0.001, reduce if training is unstable
4. **Iterations**: 2000+ recommended for high-quality results
5. **Checkpointing**: Save intermediate results for long training runs

## Expected Results

A successful optimization should achieve:
- **Low frontal area**: Streamlined shape for reduced drag
- **High top area**: Flat or curved top surface for solar panels
- **Smooth transitions**: Manufacturable geometry
- **Constraint satisfaction**: Shape fits within specified bounds

The final vehicle shape typically resembles an elongated droplet or optimized aerodynamic form with a flattened top surface.

## Troubleshooting

### Common Issues
1. **Training instability**: Reduce learning rate or increase gradient clipping
2. **Poor shape quality**: Increase Eikonal loss weight or iterations
3. **Constraint violations**: Increase size constraint loss weight
4. **Slow convergence**: Adjust loss weights or use learning rate scheduling

### GPU Memory Issues
- Reduce `sample_points_per_iter`
- Use smaller network (`hidden=64, layers=4`)
- Enable gradient checkpointing

## Future Extensions

Potential improvements:
- **CFD integration**: Real aerodynamic simulation feedback
- **Manufacturing constraints**: Thickness and material considerations
- **Multi-point optimization**: Different driving conditions
- **Structural analysis**: Stress and load considerations
- **Multi-objective Pareto optimization**: Trade-off curve analysis


## Contributing to this module

If you'd like to contribute to this code, please follow good practices as outlined in the subprojects


| WP4 Guidelines: | [Insert link to WP4 Guidelines here] |
| --------------- | ------------------------------------ |
|                 |                                      |


## License

This implementation is provided for educational and research purposes. Feel free to modify and extend for your specific applications.
