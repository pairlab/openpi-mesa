"""Name helpers for cameras in Depth/PointCloud observations.

Observation dicts use these key conventions:
  image:        "{camera}_0_rgb"
  depth:        "{camera}_0_depth"
  intrinsics:   "{camera}_intrinsics"
  extrinsics:   "{camera}_extrinsics"
  point cloud:  "{camera}_point_cloud"
"""


def get_camera_names(observation) -> list[str]:
    # Imported lazily to avoid a circular import with ``openpi.models.model``.
    from openpi.models import model as _model

    if isinstance(observation, _model.PointCloudObservation):
        source = observation.point_clouds
    elif isinstance(observation, _model.DepthObservation):
        source = observation.depth_images
    else:
        source = observation.images
    return [get_camera_name(k) for k in source]


def get_image_key(camera_name: str) -> str:
    return f"{camera_name}_0_rgb"


def get_depth_key(camera_name: str) -> str:
    return f"{camera_name}_0_depth"


def get_extrinsics_key(camera_name: str) -> str:
    return f"{camera_name}_extrinsics"


def get_intrinsics_key(camera_name: str) -> str:
    return f"{camera_name}_intrinsics"


def get_point_cloud_key(camera_name: str) -> str:
    return f"{camera_name}_point_cloud"


def get_camera_name(key: str) -> str:
    for suffix in ("_0_rgb", "_0_depth", "_extrinsics", "_intrinsics", "_point_cloud"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    raise ValueError(f"Invalid key: {key!r}")
