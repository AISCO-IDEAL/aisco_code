from subprojects.wp3_solar_estimator.raytracing.diffable.version2.sdf_ray_visibility_v2 import sdf_visibility_v2


#Wrapper since V3 of sdf visibility does not work yet
def sdf_visibility_v3(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    *,
    normal_source: Literal["mesh", "sdf"] = "mesh",
    method: Literal["sigmoid", "absorption"] = "sigmoid",
    # Sigmoid method params
    sdf_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    return sdf_visibility_v2(
        sdf_fn=sdf_fn,
        vertices=vertices,
        faces=faces,
        sun_dir=sun_dir,
        normal_source=normal_source,
        method=method,
        sdf_cfg=sdf_cfg
    )