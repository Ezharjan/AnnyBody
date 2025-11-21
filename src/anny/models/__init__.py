# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import anny.models.full_model
import torch
from anny.paths import ANNY_CACHE_DIR
from anny.face_segmentation import get_face_segmentation_mask
import anny.models.retopology

_eye_bone_labels = {"eye.L", "eye.R"}
_tongue_bone_labels = {"tongue00", "tongue01", "tongue02", "tongue03", "tongue04", "tongue05.L", "tongue05.R",
                       "tongue06.L", "tongue06.R", "tongue07.L", "tongue07.R"}
_facial_expression_bone_labels = {
        "jaw", "special04", "oris02", "oris01", "oris06.L", "oris07.L", "oris06.R", "oris07.R",
        "levator02.L", "levator03.L",
        "levator04.L", "levator05.L", "levator02.R", "levator03.R", "levator04.R", "levator05.R",
        "special01", "oris04.L", "oris03.L", "oris04.R", "oris03.R", "oris06", "oris05", "special03",
        "levator06.L", "levator06.R", "special06.L", "special05.L", "orbicularis03.L",
        "orbicularis04.L", "special06.R", "special05.R", "orbicularis03.R", "orbicularis04.R",
        "temporalis01.L", "oculi02.L", "oculi01.L", "temporalis01.R", "oculi02.R", "oculi01.R",
        "temporalis02.L", "risorius02.L", "risorius03.L", "temporalis02.R", "risorius02.R", "risorius03.R"}
_toe_bone_labels = {
    "toe1-1.L", "toe1-2.L",
    "toe2-1.L", "toe2-2.L", "toe2-3.L",
    "toe3-1.L", "toe3-2.L", "toe3-3.L",
    "toe4-1.L", "toe4-2.L", "toe4-3.L",
    "toe5-1.L", "toe5-2.L", "toe5-3.L",
    "toe1-1.R", "toe1-2.R",
    "toe2-1.R", "toe2-2.R", "toe2-3.R",
    "toe3-1.R", "toe3-2.R", "toe3-3.R",
    "toe4-1.R", "toe4-2.R", "toe4-3.R",
    "toe5-1.R", "toe5-2.R", "toe5-3.R"}
_hand_bone_labels = {
    "metacarpal1.L", "finger1-1.L", "finger1-2.L", "finger1-3.L",
    "metacarpal2.L", "finger2-1.L", "finger2-2.L", "finger2-3.L",
    "metacarpal3.L", "finger3-1.L", "finger3-2.L", "finger3-3.L",
    "metacarpal4.L", "finger4-1.L", "finger4-2.L", "finger4-3.L",
                     "finger5-1.L", "finger5-2.L", "finger5-3.L",
    "metacarpal1.R", "finger1-1.R", "finger1-2.R", "finger1-3.R",
    "metacarpal2.R", "finger2-1.R", "finger2-2.R", "finger2-3.R",
    "metacarpal3.R", "finger3-1.R", "finger3-2.R", "finger3-3.R",
    "metacarpal4.R", "finger4-1.R", "finger4-2.R", "finger4-3.R",
                     "finger5-1.R", "finger5-2.R", "finger5-3.R"}
_breast_bone_labels = {"breast.L", "breast.R"}

def create_fullbody_model(rig="default",
                          topology="default",
                          local_changes=False,
                          remove_unattached_vertices=True,
                          triangulate_faces=False,
                          default_pose_parameterization : str = "root_relative_world",
                          extrapolate_phenotypes=False,
                          all_phenotypes=False,
                          skinning_method=None,
                          cache_dirname=ANNY_CACHE_DIR):
    bones_to_remove = set()
    if rig.startswith("default"):
        rig_specs = rig.split("-")
        assert rig_specs[0] == "default"
        for spec in rig_specs[1:]:
            if spec == "noeyes":
                bones_to_remove.update(_eye_bone_labels)
            elif spec == "notongue":
                bones_to_remove.update(_tongue_bone_labels)
            elif spec == "noexpression":
                bones_to_remove.update(_facial_expression_bone_labels)
                bones_to_remove.update(_eye_bone_labels)
                bones_to_remove.update(_tongue_bone_labels)
            elif spec == "notoes":
                bones_to_remove.update(_toe_bone_labels)
            elif spec == "nohands":
                bones_to_remove.update(_hand_bone_labels)
            elif spec == "nobreasts":
                bones_to_remove.update(_breast_bone_labels)
            else:
                raise ValueError(f"Unknown rig specifier: {spec}")
        rig = "default"

    if topology.startswith("default") or topology.startswith("makehuman"):
        topology_specs = topology.split("-")
        assert topology_specs[0] == "default"

        eyes = True
        tongue = True
        for spec in topology_specs[1:]:
            if spec == "noeyes":
                eyes = False
            elif spec == "notongue":
                tongue = False
            else:
                raise ValueError(f"Unknown topology specifier: {spec}")
        topology = topology_specs[0]

        return anny.models.full_model.create_model(rig=rig,
                                            topology=topology,
                                            eyes=eyes,
                                            tongue=tongue,
                                            local_changes=local_changes,
                                            bones_to_remove=bones_to_remove,
                                            default_pose_parameterization=default_pose_parameterization,
                                            extrapolate_phenotypes=extrapolate_phenotypes,
                                            all_phenotypes=all_phenotypes,
                                            remove_unattached_vertices=remove_unattached_vertices,
                                            triangulate_faces=triangulate_faces,
                                            skinning_method=skinning_method,
                                            cache_dirname=cache_dirname)
    else:
        return anny.models.retopology.create_alternative_topology_model(
                            rig=rig,
                            topology=topology,
                            all_phenotypes=all_phenotypes,
                            bones_to_remove=bones_to_remove,
                            default_pose_parameterization=default_pose_parameterization,
                            extrapolate_phenotypes=extrapolate_phenotypes,
                            local_changes=local_changes,
                            skinning_method=skinning_method,
                            cache_dirname=cache_dirname)

def create_hand_model(side='R',
                      local_changes=False,
                      remove_unattached_vertices=True,
                      triangulate_faces=False,
                      default_pose_parameterization="root_relative",
                      extrapolate_phenotypes=False,
                      all_phenotypes=False,
                      cache_dirname=ANNY_CACHE_DIR):
    hand_bones = {f"wrist.{side}",
                    f"finger1-1.{side}",
                    f"finger1-2.{side}",
                    f"finger1-3.{side}",
                    f"metacarpal1.{side}",
                    f"finger2-1.{side}",
                    f"finger2-2.{side}",
                    f"finger2-3.{side}",
                    f"metacarpal2.{side}",
                    f"finger3-1.{side}",
                    f"finger3-2.{side}",
                    f"finger3-3.{side}",
                    f"metacarpal3.{side}",
                    f"finger4-1.{side}",
                    f"finger4-2.{side}",
                    f"finger4-3.{side}",
                    f"metacarpal4.{side}",
                    f"finger5-1.{side}",
                    f"finger5-2.{side}",
                    f"finger5-3.{side}"}
    
    fullbody_model = anny.models.full_model.create_model(default_pose_parameterization=default_pose_parameterization,
                                                         all_phenotypes=all_phenotypes,
                                                        cache_dirname=cache_dirname)
    bones_to_remove = {label for label in fullbody_model.bone_labels if not label in hand_bones}
    faces_to_keep = get_face_segmentation_mask(fullbody_model, [f"hand.{side}"])

    return anny.models.full_model.create_model(bones_to_remove=bones_to_remove,
                                               faces_to_keep=faces_to_keep,
                                               local_changes=local_changes,
                                               remove_unattached_vertices=remove_unattached_vertices,
                                               triangulate_faces=triangulate_faces,
                                               default_pose_parameterization=default_pose_parameterization,
                                               extrapolate_phenotypes=extrapolate_phenotypes,
                                               all_phenotypes=all_phenotypes,
                                               cache_dirname=cache_dirname)

def create_head_model(eyes=True,
                    tongue=True,
                    local_changes=False,
                    default_pose_parameterization="root_relative",
                    extrapolate_phenotypes=False,
                    all_phenotypes=False,
                    remove_unattached_vertices=True,
                    triangulate_faces=False,
                    cache_dirname=ANNY_CACHE_DIR):
    fullbody_model = anny.models.full_model.create_model(eyes=eyes,
                                                         tongue=tongue,
                                                         default_pose_parameterization=default_pose_parameterization,
                                                         extrapolate_phenotypes=extrapolate_phenotypes,
                                                         all_phenotypes=all_phenotypes,
                                                        cache_dirname=cache_dirname)
    
    face_bones = {"neck01", "neck02", "neck03", "head"}
    face_bones.update(_facial_expression_bone_labels)
    if eyes:
        face_bones.update(_eye_bone_labels)
    if tongue:
        face_bones.update(_tongue_bone_labels)

    bones_to_remove = {label for label in fullbody_model.bone_labels if not label in face_bones}
    faces_to_keep = get_face_segmentation_mask(fullbody_model, ["head",
                                                                "eye_cavity.R",
                                                                "eye_cavity.L",
                                                                "mouth_cavity",
                                                                "eye_front.L",
                                                                "eye_back.L",
                                                                "eye_front.R",
                                                                "eye_back.L",
                                                                "tongue"])

    return anny.models.full_model.create_model(eyes=eyes,
                                               tongue=tongue,
                                               local_changes=local_changes,
                                               bones_to_remove=bones_to_remove,
                                               faces_to_keep=faces_to_keep,
                                               remove_unattached_vertices=remove_unattached_vertices,
                                               triangulate_faces=triangulate_faces,
                                               default_pose_parameterization=default_pose_parameterization,
                                               extrapolate_phenotypes=extrapolate_phenotypes,
                                               all_phenotypes=all_phenotypes,
                                               cache_dirname=cache_dirname)
