import torch
from geometries import sdf_circle, sdf_box, sdf_union_GT,sdf_intersection_GT, visualization
from training import train
import numpy


if __name__ == "__main__":
    
    # Create a grid of points
    x = torch.linspace(-1.0, 1.0, 400)
    y = torch.linspace(-1.0, 1.0, 400)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # Compute the SDF values for the union shape
    union_values = sdf_union_GT(points).reshape(X.shape)
    intersect_values = sdf_intersection_GT(points).reshape(X.shape)


    visualization(union_values,text="Union GT")
    visualization(intersect_values,text="Intersection GT")

    union_model=train(sdf_union_GT, epochs=1250, batch_size=512, lr=7e-4, device='cpu')
    union_outputs=union_model(points)
    visualization(union_outputs.reshape(X.shape),text="Union Network")


    intersect_model=train(sdf_intersection_GT, epochs=1250, batch_size=512, lr=7e-4, device='cpu')
    intersect_outputs=intersect_model(points)
    visualization(intersect_outputs.reshape(X.shape),text="Intersection Network")