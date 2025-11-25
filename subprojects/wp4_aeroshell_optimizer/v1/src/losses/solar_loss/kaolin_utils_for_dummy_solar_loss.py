


from kaolin.ops.conversions import FlexiCubes, voxelgrids_to_trianglemeshes, sdf_to_voxelgrids


def marching_tetrahedra(model,
                        grid_res: list[int] | int = 32,
                        origin: float = 0.0,
                        box_size: float = 12000.0,
                        upsampling: int = 4,
                        dev: str="cuda"):
    """
    Alternative mesh extraction using marching tetrahedra from Kaolin.
    
    Should provide more stable normals than flexicubes, though it can't remove discontinuities completely
    
    """
    
    model.cpu()
    adjusted_model = lambda x: (model(x)[0].cpu()).squeeze(-1)  # return CPU for Kaolin

    grid = sdf_to_voxelgrids(
        [adjusted_model],
        bbox_center=origin,
        bbox_dim=box_size,
        init_res=grid_res,
        upsampling_steps=upsampling,
        
    )
    model.cuda() 
    grid = grid.to(dev)

    voxels = grid.float()
    if voxels.sum() == 0:
        print("Warning: Empty voxel grid generated!")
    verts, faces = voxelgrids_to_trianglemeshes(voxels)
    verts = verts[0] / (voxels.shape[1] - 1)          # normalize to [0,1]
    verts = verts * box_size + (origin - box_size/2)  # map back to real world coords

    return verts, faces[0]