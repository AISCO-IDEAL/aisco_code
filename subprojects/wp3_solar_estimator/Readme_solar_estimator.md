Overview over the Versions of the Solar estimator:

- V0: Everithing that has been implemented before the 8.Dec 2025, mostly by the freelancers
- V1: A non-differentiable solar loss based on the dotproduct-Approach
- V2: A differentiable solar loss based on the dotproduct-Approach and using the MeshSDF method to backpropagate the loss trough the mesh-extraction #clhuber
- V3: New version made in second semester
- V3B: Better, bugfixed version of the V3 version. Uses soft selection on the dotproduct approach to achieve better results. Still supports only a single solar vector
- V4: Uses data from Australia Telemetry to get the different solar vectors along the route. Works the same way as V3B but aggregates the quality so the whole route is considered. It uses the RouteV1 class to load the data. Based on the australia.yaml config it chooses a certain number of points in intervals, then based on the position a number of solar samples will be collected (uniformly distributed among the racing window times). The solar quality is then aggregated, the rest is unchanged from V3B
- V5: Uses RouteV2. This mode now has a validation tool included. Main difference to V4 is that the Route now assumes a constant speed (configurable) and based on this figures out at what point the car will be where. For each timestep, now only a single solar vector (corresponding to the exact time) will be sampled. Same as in V4 the qualities are then aggregated
