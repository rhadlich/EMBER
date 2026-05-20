import gymnasium as gym
from gymnasium import spaces
from .reward_typing import RewardFn
import numpy as np
import torch
from typing import Union
from pathlib import Path

from core.environments.predictor import Predictor

import logging
import utils.logging_setup as logging_setup


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTOR_WEIGHTS = PROJECT_ROOT / "src/core/digital_twin/models/model_weights_mac_local_new.pth"
FALLBACK_PREDICTOR_CONFIG = {
    "input_dim": 3,
    "output_dim": 3600,
    "num_hidden": 4,
    "hidden_exp": 10,
    "dropout": 0.1,
}


def _build_predictor_from_checkpoint(weights_path: Path) -> Predictor:
    if not weights_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if model_config is None:
        model_config = FALLBACK_PREDICTOR_CONFIG
        logging.getLogger("MyRLApp.Environment").warning(
            "No model_config found in %s; using fallback predictor architecture values.",
            weights_path,
        )

    required_keys = ("input_dim", "output_dim", "num_hidden", "hidden_exp", "dropout")
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise KeyError(
            f"Missing required model_config keys in checkpoint {weights_path}: {missing_keys}"
        )

    predictor = Predictor()
    predictor.init_model(
        input_size=int(model_config["input_dim"]),
        num_layers=int(model_config["num_hidden"]),
        layer_exp=int(model_config["hidden_exp"]),
        out_size=int(model_config["output_dim"]),
        dropout=float(model_config["dropout"]),
        weights_path=str(weights_path),
    )
    return predictor


def _resolve_predictor_weights_path(weights_path: Union[str, Path, None]) -> Path:
    if weights_path is None:
        return DEFAULT_PREDICTOR_WEIGHTS
    path = Path(weights_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class EngineEnvDiscrete(gym.Env):
    """
    This is the class that defines the engine environment.

    The observation will be a dictionary with two keys: state and target. The state will be two-dimensional and will
    contain the value of the IMEP and MPRR achieved for a cycle. The target will be one-dimensional and will hold the
    value of the desired IMEP for that cycle.

    observation_space = spaces.Dict(
        "state":
    )

    The action space is of dimension 3 and contains the injection duration, injection pressure, and injection timing.
    """

    metadata = {'render.modes': []}

    def __init__(self,
                 observation_space: spaces.Space = None,
                 action_space: spaces.Space = None,
                 reward: RewardFn = None,
                 predictor_weights_path: Union[str, Path, None] = None,
                 ):

        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.imep_space = np.arange(1.6, 4.1, 0.1)
            self.mprr_space = np.arange(0, 15, 0.5)
            # self.observation_space = spaces.Dict(
            #     {
            #         "state": spaces.MultiDiscrete([len(self.imep_space), len(self.mprr_space)]),
            #         "target": spaces.Discrete(len(self.imep_space))
            #     }
            # )
            self.observation_space = spaces.Discrete(len(self.imep_space))

        if action_space is not None:
            self.action_space = action_space
        else:
            self.inj_p_space = np.arange(450, 950, 100)
            self.soi_space = np.arange(-5.6, 2.7, 0.1)
            self.inj_d_space = np.arange(0.31, 0.64, 0.01)
            self.action_space = spaces.Tuple((
                # spaces.Discrete(len(self.inj_p_space)),
                spaces.Discrete(len(self.soi_space)),
                spaces.Discrete(len(self.inj_d_space))
            ))
            # self.action_space = spaces.MultiDiscrete([len(self.inj_p_space),
            #                                           len(self.soi_space),
            #                                           len(self.inj_d_space)])

        self.reward = reward

        self._current_mprr = None
        self._current_imep = None
        self._desired_imep = None

        predictor_weights = _resolve_predictor_weights_path(predictor_weights_path)
        self.predictor = _build_predictor_from_checkpoint(predictor_weights)

        self.logger = logging.getLogger("MyRLApp.Environment")

    def reset(
            self,
            seed: int = None,
            options=None,
    ):
        super().reset(seed=seed)

        # sample random values for observations
        self._current_imep = self.np_random.choice(self.imep_space)
        self._current_mprr = self.np_random.choice(self.mprr_space[self.mprr_space < 7])

        # sample random desired IMEP (must be different from observation)
        while True:
            self._desired_imep = self.np_random.choice(self.imep_space)
            if self._desired_imep != self._current_imep:
                break

        observation = self._get_obs()

        info = {"current imep": self._current_imep, "mprr": self._current_mprr}

        return observation, info

    def step(self,
             filtered_action_vals: Union[list, np.ndarray] = None,
             nominal_action_vals: Union[list, np.ndarray] = None,
             ):
        """
        action_vals is expected to be in the order -> [inj_p, soi, inj_d] and to be
        values rather than indices. Helper functions are provided to convert between
        the two formats, but must be done prior to calling this function.
        """

        # send action values to torch model and get new state
        pressure, self._current_imep, self._current_mprr, cad = (
            self.predictor.model_predict(filtered_action_vals, noise_in_percent=1))

        # get observation in the right format
        observation = self._get_obs()

        # package inputs for reward
        reward_inputs = {
            "target": self._desired_imep,
            "current imep": self._current_imep,
            "mprr": self._current_mprr,
            "filtered action": filtered_action_vals,
            "nominal action": nominal_action_vals
            }

        # calculate reward
        reward_vec = self.reward(reward_inputs)
        reward = np.sum(reward_vec) + 18.0

        # clip observation values to make sure it is within the expected space
        self._current_imep = np.clip(self._current_imep, self.imep_space[0], self.imep_space[-1])
        self._current_mprr = np.clip(self._current_mprr, self.mprr_space[0], self.mprr_space[-1])
        observation = self._get_obs()

        terminated = 0      # will decide in the controller if it is terminated or not
        info = {"current imep": self._current_imep, "mprr": self._current_mprr, "pressure": pressure}

        return observation, reward, reward_vec, terminated, False, info

    # def _get_obs(self):
    #     return {"state": np.array([self._current_imep, self._current_mprr]), "target": self._desired_imep}
    def action_ind_to_vals(self, action_ind: Union[list, np.ndarray]) -> np.ndarray:
        # inj_p = self.inj_p_space[action_ind[0]]
        soi = self.soi_space[action_ind[0]]
        inj_d = self.inj_d_space[action_ind[1]]
        # return np.array([inj_p, soi, inj_d]).astype(np.float32)
        return np.array([soi, inj_d]).astype(np.float32)
    
    def _find_nearest_index(self, value: Union[float, np.float32], array: np.ndarray) -> int:
        value = np.float32(value)
        return np.argmin(np.abs(array - value))
    
    def action_vals_to_ind(self, action_vals: np.ndarray) -> Union[list, np.ndarray]:
        inj_p_ind = self._find_nearest_index(action_vals[0], self.inj_p_space)
        soi_ind = self._find_nearest_index(action_vals[1], self.soi_space)
        inj_d_ind = self._find_nearest_index(action_vals[2], self.inj_d_space)
        return np.array([inj_p_ind, soi_ind, inj_d_ind])

    def _get_obs(self):
        return np.array(self._desired_imep)


class EngineEnvContinuous(gym.Env):
    """
    This is the class that defines the engine environment with continuous observation and state spaces.

    The observation will be a dictionary with two keys: state and target. The state will be two-dimensional and will
    contain the value of the IMEP and MPRR achieved for a cycle. The target will be one-dimensional and will hold the
    value of the desired IMEP for that cycle.

    observation_space = spaces.Dict(
        "state":
    )

    The action space is of dimension 3 and contains the injection duration, injection pressure, and injection timing.
    """

    metadata = {'render.modes': []}

    def __init__(self,
                 observation_space: spaces.Space = None,
                 action_space: spaces.Space = None,
                 reward: RewardFn = None,
                 predictor_weights_path: Union[str, Path, None] = None,
                 ):

        if action_space is not None:
            self.action_space = action_space
        else:
            self.ID1_lims = [0.6, 0.9]
            self.ID2_lims = [0.0, 1.0]
            self.SOI2_lims = [-140, -10]
            self.action_space = spaces.Box(
                low=np.array([self.ID1_lims[0], self.ID2_lims[0], self.SOI2_lims[0]], dtype=np.float32),
                high=np.array([self.ID1_lims[1], self.ID2_lims[1], self.SOI2_lims[1]], dtype=np.float32),
            )
        
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.imep_sample_lims = [1.6, 4.1]
            self.mprr_sample_lims = [1, 8]
            self.imep_env_limits = [-1.5, 6.0]
            self.mprr_env_limits = [0, 15]
            self.observation_space = spaces.Box(
                low=np.array([
                    self.soi_lims[0],
                    self.inj_d_lims[0],
                    self.imep_lims[0],
                    self.imep_lims[0],
                    self.imep_lims[0]],
                    dtype=np.float32
                    ),
                high=np.array([
                    self.soi_lims[1],
                    self.inj_d_lims[1],
                    self.imep_lims[1],
                    self.imep_lims[1],
                    self.imep_lims[1]],
                    dtype=np.float32
                    ),
                dtype=np.float32
            )

        self.reward = reward

        self._current_mprr = None
        self._current_imep = None
        self._desired_imep = None

        predictor_weights = _resolve_predictor_weights_path(predictor_weights_path)
        self.predictor = _build_predictor_from_checkpoint(predictor_weights)

        self.logger = logging.getLogger("MyRLApp.Environment")

    def reset(
            self,
            seed: int = None,
            options=None,
    ):
        super().reset(seed=seed)

        # sample random values for observations
        self._current_imep = self.np_random.uniform(self.imep_lims[0], self.imep_lims[1])
        self._current_mprr = self.np_random.uniform(1, 7)

        # sample random desired IMEP (must be different from observation)
        while True:
            self._desired_imep = self.np_random.uniform(self.imep_lims[0], self.imep_lims[1])
            if self._desired_imep != self._current_imep:
                break

        observation = self._get_obs()

        info = {"current imep": self._current_imep, "mprr": self._current_mprr}

        return observation, info

    def step(self,
            filtered_action_vals: Union[list, np.ndarray] = None,
            nominal_action_vals: Union[list, np.ndarray] = None,
             ):

        # action_vals should be in the order -> [inj_p, soi, inj_d]

        # send action values to torch model and get new state
        pressure, self._current_imep, self._current_mprr, cad = (
            self.predictor.model_predict(filtered_action_vals, noise_in_percent=3))

        # package inputs for reward
        reward_inputs = {
            "target": self._desired_imep,
            "current imep": self._current_imep,
            "mprr": self._current_mprr,
            "filtered action": filtered_action_vals,
            "nominal action": nominal_action_vals
            }

        # calculate reward
        reward_vec = self.reward(reward_inputs)
        reward = (np.sum(reward_vec) + 15.0) * 0.1

        # clip observation values to make sure it is within the expected space
        self._current_imep = np.clip(self._current_imep, self.imep_lims[0], self.imep_lims[-1])
        self._current_mprr = np.clip(self._current_mprr, self.mprr_lims[0], self.mprr_lims[-1])
        observation = self._get_obs()

        terminated = 0      # will decide in the controller if it is terminated or not
        info = {"current imep": self._current_imep, "mprr": self._current_mprr, "pressure": pressure}

        return observation, reward, reward_vec, terminated, False, info

    # def _get_obs(self):
    #     return {"state": np.array([self._current_imep, self._current_mprr]), "target": self._desired_imep}
    def _get_obs(self):
        return self._desired_imep


def reward_fn(inputs):
    load_tracking = np.abs(inputs["current imep"] - inputs["target"]) * -5.0
    # load_tracking += np.int16((np.abs(inputs["current imep"] - inputs["target"]) - 0.05*inputs["target"]) < 0.0) * 4.0
    safety = (max(0, inputs["mprr"]-7)**2) * -0.0
    filter_interference = (np.linalg.norm(inputs["filtered action"] - inputs["nominal action"])**2) * -0.0
    return np.array([load_tracking, safety, filter_interference])
