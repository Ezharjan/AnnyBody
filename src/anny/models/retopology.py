# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import torch
from anny.models.phenotype import RiggedModelWithPhenotypeParameters
from anny.utils import obj_utils
from anny.models.full_model import create_model
import os
from anny.utils.mesh_utils import triangulate_faces
from anny.paths import ANNY_CACHE_DIR, ANNY_ROOT_DIR
import collections
import roma

def create_alternative_topology_model(rig="default",
                                      topology="default",
                                bones_to_remove=set(),
                                all_phenotypes=False,
                                skinning_method=None,
                                default_pose_parameterization: str = "root_relative_world",
                                extrapolate_phenotypes=False,
                                local_changes=None,
                                root_dirname=ANNY_ROOT_DIR,
                                cache_dirname=ANNY_CACHE_DIR):
    # Disable eyes and tongue bones by default
    reference_model = create_model(rig=rig,
                                eyes=False,
                                tongue=False,
                                bones_to_remove=bones_to_remove,
                                all_phenotypes=all_phenotypes,
                                skinning_method=skinning_method,
                                default_pose_parameterization=default_pose_parameterization,
                                extrapolate_phenotypes=extrapolate_phenotypes,
                                local_changes=local_changes,
                                remove_unattached_vertices=False,
                                root_dirname=root_dirname,
                                cache_dirname=cache_dirname)
    
    # Use a local import to avoid a dependency to WARP if not needed
    from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs

    transformation = roma.Rotation(roma.euler_to_rotmat("x", [90.], degrees=True, dtype=torch.float64)[None])

    # Load a reference mesh to establish correspondences.
    reference_vertices, _, template_groups = obj_utils.load_obj_file(os.path.join(root_dirname, "data/topology/default.obj"), dtype=torch.float64)
    reference_vertices = transformation.apply(reference_vertices)
    reference_faces = template_groups['noname']['face_vertex_indices']
    template_triangulated_faces = torch.tensor(triangulate_faces(reference_vertices, reference_faces.numpy().tolist()), dtype=torch.int64)

    # Load the mesh with a different topology
    mesh_filename = os.path.join(root_dirname, "data/topology/", topology + ".obj")
    vertices, vertex_texture_coordinates, groups = obj_utils.load_obj_file(mesh_filename, dtype=torch.float64, pack_as_tensor=False)
    vertices = torch.as_tensor(vertices, dtype=torch.float64)
    vertices = transformation.apply(vertices)
    faces = groups['noname']['face_vertex_indices']
    faces = torch.tensor(triangulate_faces(vertices, faces), dtype=torch.int64)

    # Compute barycentric coefficients on the closest triangle in the reference mesh for each vertex in the low resolution model.
    distances, face_ids, uvs = point_to_mesh_distance_and_face_uvs(points=vertices.to(dtype=torch.float32),
                                        vertices=reference_vertices.to(dtype=torch.float32),
                                        faces=template_triangulated_faces,
                                        max_dist=1000.)
    
    assert distances.max() < 1.5e-2, "Some vertices are too far from the reference model."
    uvs = uvs.to(dtype=torch.float64)

    # Interpolate skinning weights and blendshapes based on barycentric coefficients.
    u, v = uvs[:,0], uvs[:,1]
    w = 1. - u - v
    barycentric_coeffs = [u, v, w]
    template_vertex_indices = template_triangulated_faces[face_ids]
    blendshapes = sum([reference_model.blendshapes[:,template_vertex_indices[:,i]] * barycentric_coeffs[i][None,:,None] for i in range(3)])

    vertex_bone_weights = []
    vertex_bone_indices = []
    for vertex_id in range(len(vertices)):
        new_weights = collections.defaultdict(lambda : 0.)
        for i in range(3):
            template_vertex_id = template_vertex_indices[vertex_id, i].item()
            coeff = barycentric_coeffs[i][vertex_id].item()
            
            for bone_idx, bone_weight in zip(reference_model.vertex_bone_indices[template_vertex_id], reference_model.vertex_bone_weights[template_vertex_id]):
                new_weights[bone_idx.item()] += coeff * bone_weight.item()
        # Remove zero weights
        new_weights = {k:v for k,v in new_weights.items() if v > 0.}
        vertex_bone_weights.append(list(new_weights.values()))
        vertex_bone_indices.append(list(new_weights.keys()))

    # Pad the lists to have the same length for each vertex
    max_bones_per_vertex = max([len(indices) for indices in vertex_bone_indices])
    print(f"{max_bones_per_vertex=}")
    for indices, weights in zip(vertex_bone_indices, vertex_bone_weights):
        while len(indices) < max_bones_per_vertex:
            indices.append(0)
            weights.append(0.)
    vertex_bone_indices = torch.as_tensor(vertex_bone_indices, dtype=torch.int64)
    vertex_bone_weights = torch.as_tensor(vertex_bone_weights, dtype=torch.float64)
    vertex_bone_weights /= torch.sum(vertex_bone_weights, dim=-1, keepdim=True)


    downsampled_model = RiggedModelWithPhenotypeParameters(template_vertices=vertices,
                                     faces=faces,
                                     blendshapes=blendshapes,
                                     template_bone_heads=reference_model.template_bone_heads,
                                     template_bone_tails=reference_model.template_bone_tails,
                                     bone_heads_blendshapes=reference_model.bone_heads_blendshapes,
                                     bone_tails_blendshapes=reference_model.bone_tails_blendshapes,
                                     bone_rolls_rotmat=reference_model.bone_rolls_rotmat,
                                     bone_parents = reference_model.bone_parents,
                                     bone_labels = reference_model.bone_labels,
                                     vertex_bone_weights = vertex_bone_weights,
                                     vertex_bone_indices = vertex_bone_indices,
                                     stacked_phenotype_blend_shapes_mask=reference_model.stacked_phenotype_blend_shapes_mask,
                                     local_change_labels=reference_model.local_change_labels,
                                     default_pose_parameterization=reference_model.default_pose_parameterization,
                                     base_mesh_vertex_indices=None,
                                     extrapolate_phenotypes=extrapolate_phenotypes,
                                     all_phenotypes=all_phenotypes,
                                     skinning_method=skinning_method,
                                     texture_coordinates=None,
                                     face_texture_coordinate_indices=None)
    return downsampled_model


if __name__ == "__main__":
    lowres_model = create_lowres_fullbody_model()
    print(lowres_model)