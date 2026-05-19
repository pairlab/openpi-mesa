import numpy as np

from openpi.models import model as _model
from openpi.policies import mesa_policy
from openpi.training import config as _config


def test_multicam_2d_inputs_drop_depth_and_calibration():
    transform = mesa_policy.MESABimanualAdapt3RMultiCamInputs(
        model_type=_model.ModelType.PI05,
        cameras=("egocentric", "leftshoulder", "rightshoulder", "midshoulder"),
        include_depth=False,
    )
    image = np.zeros((3, 224, 224), dtype=np.float32)
    data = {
        "observation/state": np.zeros((14,), dtype=np.float32),
        "observation/image_egocentric": image,
        "observation/image_leftshoulder": image,
        "observation/image_rightshoulder": image,
        "observation/image_midshoulder": image,
        "prompt": np.asarray("smoke"),
        "actions": np.zeros((20, 14), dtype=np.float32),
    }

    inputs = transform(data)

    assert "depth" not in inputs
    assert "calibration" not in inputs
    batched_inputs = {
        "image": {key: value[None] for key, value in inputs["image"].items()},
        "image_mask": {key: np.asarray([value]) for key, value in inputs["image_mask"].items()},
        "state": inputs["state"][None],
    }
    observation = _model.observation_from_dict(batched_inputs)
    assert type(observation) is _model.Observation


def test_multicam_2d_train_config_disables_depth():
    train_config = _config.get_config("pi05_mesa_bimanual_lora_2d_gen_camdrop")

    assert isinstance(train_config.data, _config.MESABimanualAdapt3RMultiCamDataConfig)
    assert train_config.data.include_depth is False
