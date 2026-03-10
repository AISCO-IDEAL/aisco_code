Overview over the Versions of the Solar estimator:

- V0: Everithing that has been implemented before the 8.Dec 2025, mostly by the freelancers
- V1: A non-differentiable solar loss based on the dotproduct-Approach
- V2: A differentiable solar loss based on the dotproduct-Approach and using the MeshSDF method to backpropagate the loss trough the mesh-extraction #clhuber
- V3: New version made in second semester
- V3B: Better, bugfixed version of the V3 version. Uses soft selection on the dotproduct approach to achieve better results. Still supports only a single solar vector
- V4: Uses data from Australia Telemetry to get the different solar vectors along the route. Works the same way as V3B but aggregates the quality so the whole route is considered
