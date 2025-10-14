  # noqa: N999
""" Structure overview
This folder is until now documented by comments in the code files.
It was started on 13.10.2025 by Clemens.

Structure Overview:
0. Imports
1. Definition of Functions
    1.1 Geometry Helper Functions
    1.2 Ray-Segment Intersection
2. Definition of Classes
    2.1 Shape
    2.2 Area
    2.3 Geni
3. Main Code Execution
    3.1 Definition of Areas for Training
    3.2 Setup and Training of the Geni
    3.3 Visualization of Results
"""

#################################
# 0. Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

##################################
# 1. Definition of Functions

# 1.1 Geometry Helper Functions
def _cart_to_polar(x, y, ox=0.0, oy=0.0):
    dx, dy = x - ox, y - oy
    r = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx)
    return r, theta

def _polar_to_cart(r, theta, ox=0.0, oy=0.0):
    x = ox + r * np.cos(theta)
    y = oy + r * np.sin(theta)
    return x, y

# 1.2 Ray-Segment Intersection
def _ray_segment_intersection(ox, oy, theta, p1, p2):
    """
    Compute intersection between ray from origin (ox, oy) at angle theta
    and segment (p1, p2). Returns distance r if intersection exists, else None.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Ray direction
    dx_r = np.cos(theta)
    dy_r = np.sin(theta)

    # Segment direction
    dx_s = x2 - x1
    dy_s = y2 - y1

    # Solve:
    # (ox, oy) + t_r*(dx_r, dy_r) = (x1, y1) + t_s*(dx_s, dy_s)
    # => two linear equations:
    a = np.array([[dx_r, -dx_s],
                  [dy_r, -dy_s]])
    b = np.array([x1 - ox, y1 - oy])

    det = np.linalg.det(a)
    if abs(det) < 1e-12:  # noqa: PLR2004
        return None  # parallel

    t_r, t_s = np.linalg.solve(a, b)

    if t_r < 0 or t_s < 0 or t_s > 1:
        return None  # intersection behind ray or outside segment

    return t_r  # this is r distance along ray
##################################
# 2. Definition of Classes
# 2.1 Shape
class Shape:
    # 2.1.1 Initialisation, with two possible ways: if pointcloud is given, or if cornor points are given (both with variable points)
    def __init__(self, points, pointcloud=False, *, origin=(0.0, 0.0), dtheta_deg=1.0):
        self.origin = tuple(origin)
        self.dtheta = np.deg2rad(dtheta_deg)
        self.orgcloud = pointcloud
        if pointcloud:
            self.pointcloud = np.array(points, dtype=float)
            self.size = len(points)
        else:
            self.points = [list(map(float, p)) for p in points]
            self.pointcloud = self.create_pointcloud()
            self.size = len(self.pointcloud)

    # 2.1.2 Create pointcloud by ray-segment intersection, if necessary
    def create_pointcloud(self):
        """Sample the shape by sweeping a ray from origin every dtheta."""
        ox, oy = self.origin
        pts = list(self.points)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        # Collect all vertex angles around origin
        all_thetas = []
        for x, y in pts:
            _, th = _cart_to_polar(x, y, ox, oy)
            all_thetas.append(th)
        th_min, th_max = np.min(all_thetas), np.max(all_thetas)
        # optionally make it a full circle
        th_min, th_max = -np.pi, np.pi

        # Sweep over constant angular steps
        thetas = np.arange(th_min, th_max, self.dtheta)
        cloud = []
        for th in thetas:
            r_hits = []
            for i in range(len(pts) - 1):
                r = _ray_segment_intersection(ox, oy, th, pts[i], pts[i + 1])
                if r is not None:
                    r_hits.append(r)
            if len(r_hits) == 0:
                # No intersection → skip or append NaN
                continue
            # choose the nearest intersection (outer boundary)
            r = min(r_hits)
            x, y = _polar_to_cart(r, th, ox, oy)
            cloud.append([x, y])

        if len(cloud) == 0:
            raise RuntimeError("No_intersections_found_-_check_origin_placement.")

        cloud = np.array(cloud, dtype=float)
        self.pointcloud_polar = np.stack(
            [_cart_to_polar(x, y, ox, oy) for x, y in cloud]
        )
        return cloud

    # 2.1.3 Plotting and point access
    def plot(self):
        plt.scatter(self.pointcloud[:, 0], self.pointcloud[:, 1], s=1)
        if not self.orgcloud:
            pts = np.array(self.points)
            plt.scatter(pts[:, 0], pts[:, 1], color='red', marker='x')
        plt.axis('equal')

    # 2.1.4 Point access
    def p(self, theta):
        if theta < 0 or theta >= len(self.pointcloud):
            raise ValueError("t_out_of_range")
        return self.pointcloud[theta]

# 2.2 Area
class Area:
    # 2.2.1 Initialisation with inner and outer shape, and parameters
    def __init__(self, inner_shape, outer_shape, params):
        self.inner_shape = inner_shape
        self.outer_shape = outer_shape
        self.size = min(inner_shape.size, outer_shape.size)
        self.params = params

    # 2.2.2 Create intermediate pointcloud by interpolation with mu
    def p(self, t, mu):
        if t < 0 or t >= self.size:
            raise ValueError("t_out_of_range")
        if not 0 <= mu <= 1:
            raise ValueError("mu_must_be_in_[0,1]")
        inner_point = self.inner_shape.p(t)
        outer_point = self.outer_shape.p(t)
        po = (1 - mu) * np.array(inner_point) + mu * np.array(outer_point)
        return po.tolist()

    # 2.2.3 Create shape from interpolation points
    def create_shape(self, mu):
        if len(mu) != self.size:
            raise ValueError("mu-wrong-size")
        points = [self.p(t, mu[t]) for t in range(self.size)]
        return Shape(points, pointcloud=True)

    # 2.2.4 Plotting
    def plot(self):
        self.inner_shape.plot()
        self.outer_shape.plot()

    # 2.2.5 Pointcloud as torch tensor for training
    def pointcloud_torch(self, mu: torch.Tensor) -> torch.Tensor:
        inner = torch.tensor(self.inner_shape.pointcloud, dtype=torch.float32, device=mu.device)
        outer = torch.tensor(self.outer_shape.pointcloud, dtype=torch.float32, device=mu.device)
        mu2 = mu.view(-1, 1)
        return (1 - mu2) * inner + mu2 * outer
    
# 2.3 Geni
class Geni:
    def __init__(self, areas):
        self.areas = areas
        self.area_size = areas[0].size
        self.latent_size = len(areas[0].params)
        self.model = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.area_size),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)    
  
    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            self.optimizer.zero_grad()

            batch_losses = []
            for area in self.areas:
                z = area.params
                z = torch.tensor(z, dtype=torch.float32)  # (latent_size,)
                mu = self.model(z)                        # (size,) differentiable
                pts = area.pointcloud_torch(mu)           # (size,2) differentiable wrt mu
                loss = self.loss_function(pts)            # scalar tensor
                batch_losses.append(loss)

            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}")

    def loss_function(self, points, alpha=10):
        def soft_max(x): return torch.logsumexp(alpha * x, dim=0) / alpha
        def soft_min(x): return -torch.logsumexp(-alpha * x, dim=0) / alpha

        x_max = soft_max(points[:, 0])
        x_min = soft_min(points[:, 0])
        y_max = soft_max(points[:, 1])
        y_min = soft_min(points[:, 1])

        width = x_max - x_min
        height = y_max - y_min

        diffs = points[1:] - points[:-1]
        # clacualte length of each segment
        seg_lengths = torch.sqrt(torch.sum(diffs**2, dim=1) + 1e-8)
        # calculate a penalty for each segment that is longer than 
        beauty = torch.sum((F.relu(seg_lengths - 0.03))**2)

        loss = beauty*10  +(height-0.5) - width 
        return loss

    def get_mu(self, area):
        latent = area.params
        latent = torch.tensor(latent, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(latent)
        mu = output.numpy()
        return mu


##################################
# 3. Main Code Execution

# 3.1 Definition of Areas for Training
areas = []
for x in range(5):
    for y in range(5):
        points1 = [[-1-x/5, -1], [1, -1], [1, 1+y/5], [-1, 1]]
        points2 = [[-0.5-x/10, -0.5], [0.5, -0.5], [0.5, 0.5-y/10], [-0.5, 0.5]]
        quader1 = Shape(points1)
        quader2 = Shape(points2)
        area1 = Area(quader2, quader1, params = [x, y])
        areas.append(area1)

# 3.2 Setup and Training of the Geni
geni = Geni(areas)

for _ in range(10):
    geni.train(50)
    area = areas[0]
    mu = geni.get_mu(area)
    shape = area.create_shape(mu)
    area.plot()
    shape.plot()
    plt.show()

# 3.3 Visualization of Results
for area in areas:
    mu = geni.get_mu(area)
    shape = area.create_shape(mu)
    area.plot()
    shape.plot()
    plt.show()