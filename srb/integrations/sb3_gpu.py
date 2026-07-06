import gc
import shutil
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import gymnasium
import numpy
import torch
from isaacsim.simulation_app import SimulationApp
from rl_zoo3 import ALGOS
from stable_baselines3.common.callbacks import BaseCallback, tqdm
from stable_baselines3.common.logger import Video
from stable_baselines3.common.vec_env import unwrap_vec_normalize

from srb.integrations.sb3.exp_manager import ExperimentManager
from srb.integrations.sb3.wrapper import Sb3EnvWrapper
from srb.utils import logging
from srb.utils.cfg import last_file, stamp_dir
from srb.wrappers import maybe_wrap_action_smoothing

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecEnv

    from srb._typing import AnyEnv, AnyEnvCfg


FRAMEWORK_NAME = "sb3_gpu"
OFF_POLICY_ALGOS: Sequence[str] = ("qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc")


def _select_model_device(requested_device: str | None, env_device) -> str:
    if requested_device and requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    return str(env_device)


class EpisodeEvalVideoCallback(BaseCallback):
    def __init__(
        self,
        *,
        logdir: Path,
        eval_every_episodes: int,
        eval_n_episodes: int,
        eval_video: bool,
        eval_video_timesteps: int,
        eval_video_prefix: str,
        concat_eval_videos: bool,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.logdir = logdir
        self.eval_every_episodes = eval_every_episodes
        self.eval_n_episodes = eval_n_episodes
        self.eval_video = eval_video
        self.eval_video_timesteps = eval_video_timesteps
        self.eval_video_prefix = eval_video_prefix
        self.concat_eval_videos = concat_eval_videos

        self._episodes_since_eval = 0
        self._total_train_episodes = 0
        self._total_train_successes = 0
        self._eval_pending = False
        self._eval_root = self.logdir.joinpath("eval_videos")
        self._eval_ckpt_root = self.logdir.joinpath("eval_ckpt")

    def _init_callback(self) -> None:
        self._eval_root.mkdir(parents=True, exist_ok=True)
        self._eval_ckpt_root.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is None:
                continue
            self._episodes_since_eval += 1
            self._total_train_episodes += 1
            is_success = bool(
                info.get("is_success", episode.get("is_success", 0.0))
            )
            self._total_train_successes += int(is_success)
            success_rate = self._total_train_successes / max(
                self._total_train_episodes, 1
            )
            self.logger.record("train/success_count", self._total_train_successes)
            self.logger.record("train/episode_count", self._total_train_episodes)
            self.logger.record("train/success_rate_total", success_rate)
            logging.info(
                "Training successes: "
                f"{self._total_train_successes}/{self._total_train_episodes} "
                f"({100.0 * success_rate:.2f}%)"
            )

        if (
            self.eval_every_episodes > 0
            and self._episodes_since_eval >= self.eval_every_episodes
        ):
            self._eval_pending = True
        return True

    def _on_rollout_end(self) -> None:
        if not self._eval_pending:
            return
        self._eval_pending = False
        self._episodes_since_eval = 0
        try:
            self._run_eval()
        except Exception as exc:
            logging.warning(f"Skipping sb3_gpu eval callback due to error: {exc}")

    def _on_training_end(self) -> None:
        if self.concat_eval_videos:
            self._concat_eval_videos()

    def _run_eval(self) -> None:
        env = self.training_env
        vec_normalize = unwrap_vec_normalize(env)
        vec_normalize_training = None
        vec_normalize_norm_reward = None
        if vec_normalize is not None:
            vec_normalize_training = vec_normalize.training
            vec_normalize_norm_reward = vec_normalize.norm_reward
            vec_normalize.training = False
            vec_normalize.norm_reward = False

        try:
            ckpt_path = self._eval_ckpt_root.joinpath(
                f"{self.eval_video_prefix}_episode_{self._total_train_episodes}.zip"
            )
            self.model.save(ckpt_path.as_posix())

            returns, lengths, frames = self._rollout_eval(env)
            if not returns:
                logging.warning("sb3_gpu eval produced no completed episodes.")
                return

            mean_reward = float(numpy.mean(returns))
            std_reward = float(numpy.std(returns))
            mean_length = float(numpy.mean(lengths))

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_episode_length", mean_length)
            self.logger.record("eval/num_episodes", len(returns))
            self.logger.record("eval/train_episode", self._total_train_episodes)

            if self.eval_video and frames:
                video_path = self._save_video(frames)
                if video_path is not None:
                    logging.info(f"Saved sb3_gpu eval video: {video_path}")
                self._record_tensorboard_video(frames)

            self.logger.dump(self.num_timesteps)
        finally:
            obs = env.reset()
            self.model._last_obs = obs  # type: ignore[attr-defined]
            self.model._last_episode_starts = numpy.ones(  # type: ignore[attr-defined]
                (env.num_envs,), dtype=bool
            )
            if vec_normalize is not None:
                vec_normalize.training = vec_normalize_training
                vec_normalize.norm_reward = vec_normalize_norm_reward

    def _rollout_eval(self, env: "VecEnv") -> tuple[list[float], list[int], list[numpy.ndarray]]:
        obs = env.reset()
        episode_returns = numpy.zeros(env.num_envs, dtype=numpy.float32)
        episode_lengths = numpy.zeros(env.num_envs, dtype=numpy.int32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        frames: list[numpy.ndarray] = []

        self._append_render_frame(env, frames)
        max_steps = max(self.eval_video_timesteps, 1) * max(self.eval_n_episodes, 1)
        for _ in range(max_steps):
            action, _state = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = env.step(action)

            episode_returns += rewards
            episode_lengths += 1
            self._append_render_frame(env, frames)

            for env_id, done in enumerate(dones):
                if not done:
                    continue
                completed_returns.append(float(episode_returns[env_id]))
                completed_lengths.append(int(episode_lengths[env_id]))
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0
                if len(completed_returns) >= self.eval_n_episodes:
                    return completed_returns, completed_lengths, frames

        return completed_returns, completed_lengths, frames

    def _append_render_frame(self, env: "VecEnv", frames: list[numpy.ndarray]) -> None:
        if not self.eval_video:
            return
        frame = env.env_method("render")
        if isinstance(frame, (list, tuple)):
            frame = frame[0] if frame else None
        if frame is None:
            return
        frame = numpy.asarray(frame)
        if frame.ndim != 3:
            return
        if frame.dtype != numpy.uint8:
            frame = numpy.clip(frame, 0, 255).astype(numpy.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)

    def _save_video(self, frames: list[numpy.ndarray]) -> Path | None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            logging.warning("imageio is not available; eval mp4 video was not saved.")
            return None

        video_dir = self._eval_root.joinpath(f"episode_{self._total_train_episodes}")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir.joinpath(f"{self.eval_video_prefix}.mp4")
        fps = self._video_fps()
        try:
            writer = imageio.get_writer(video_path.as_posix(), fps=fps)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            return video_path
        except Exception as exc:
            logging.warning(f"Failed to save eval mp4 video: {exc}")
            return None

    def _record_tensorboard_video(self, frames: list[numpy.ndarray]) -> None:
        try:
            video = torch.as_tensor(numpy.stack(frames), dtype=torch.uint8)
            video = video.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
            self.logger.record(
                "eval/video",
                Video(video, fps=self._video_fps()),
                exclude=("stdout", "log", "json", "csv"),
            )
        except Exception as exc:
            logging.warning(f"Failed to write eval video to TensorBoard: {exc}")

    def _video_fps(self) -> int:
        unwrapped = getattr(self.training_env, "unwrapped", None)
        metadata = getattr(unwrapped, "metadata", {}) or {}
        return int(metadata.get("render_fps", 25))

    def _concat_eval_videos(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logging.warning("ffmpeg is not available; eval videos were not concatenated.")
            return

        videos = sorted(
            path
            for path in self._eval_root.glob("episode_*/*.mp4")
            if path.name != "eval_progress_compare.mp4"
        )
        if not videos:
            return

        concat_file = self._eval_root.joinpath("concat.txt")
        output_path = self._eval_root.joinpath("eval_progress_compare.mp4")
        with concat_file.open("w") as f:
            for video in videos:
                escaped_path = video.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file.as_posix(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path.as_posix(),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            logging.warning(f"Failed to concatenate eval videos: {exc.stderr}")
        finally:
            concat_file.unlink(missing_ok=True)


def run(
    workflow: Literal["train", "eval"],
    algo: str,
    env: "AnyEnv | gymnasium.Env",
    sim_app: SimulationApp,
    env_id: str,
    env_cfg: "AnyEnvCfg | None",
    agent_cfg: dict,
    logdir: Path,
    model: Path,
    continue_training: bool | None = None,
    **kwargs,
):
    log_interval = agent_cfg.pop("log_interval", -1)
    verbose = agent_cfg.pop("verbose", True)
    track = agent_cfg.pop("track", False)
    init_tensorboard = agent_cfg.pop("tensorboard", True)
    init_wandb = agent_cfg.pop("wandb", False)

    save_freq = agent_cfg.pop("save_freq", -1)
    save_replay_buffer = agent_cfg.pop("save_replay_buffer", False)

    n_trials = agent_cfg.pop("n_trials", 500)
    sampler = agent_cfg.pop("sampler", "tpe")
    pruner = agent_cfg.pop("pruner", "median")
    n_startup_trials = agent_cfg.pop("n_startup_trials", 10)
    n_evaluations = agent_cfg.pop("n_evaluations", 1)
    truncate_last_trajectory = agent_cfg.pop("truncate_last_trajectory", True)

    eval_every_episodes = int(agent_cfg.pop("eval_every_episodes", 50))
    eval_n_episodes = int(agent_cfg.pop("eval_n_episodes", 1))
    eval_video = bool(agent_cfg.pop("eval_video", True))
    eval_video_timesteps = int(agent_cfg.pop("eval_video_timesteps", 625))
    agent_cfg.pop("eval_video_num_envs", 1)
    eval_video_prefix = str(agent_cfg.pop("eval_video_prefix", "eval"))
    concat_eval_videos = bool(agent_cfg.pop("concat_eval_videos", True))
    model_device = _select_model_device(
        agent_cfg.pop("policy_device", agent_cfg.pop("device", None)),
        env.unwrapped.device,  # type: ignore
    )
    logging.info(
        f"Using SB3 policy/training device {model_device} "
        f"with environment device {env.unwrapped.device}"  # type: ignore
    )

    smoothing_cfg = agent_cfg.pop("smoothing", {})

    if model:
        from_checkpoint = model
    elif workflow == "eval" or continue_training:
        from_checkpoint = last_file(logdir.joinpath("ckpt"), modification_time=True)
    else:
        from_checkpoint = ""
    if from_checkpoint:
        logging.info(f"Loading model from {from_checkpoint}")

    if workflow == "eval":
        logdir = stamp_dir(logdir.joinpath("eval"))

    tensorboard_log = (
        logdir.joinpath("tensorboard") if track or init_tensorboard else None
    )
    if init_wandb:
        import wandb

        _run = wandb.init(
            name=f"{env_id}_{FRAMEWORK_NAME}_{algo}",
            sync_tensorboard=True,
            monitor_gym=True,
        )

    env = maybe_wrap_action_smoothing(
        env,  # type: ignore
        smoothing_cfg,
    )
    env = Sb3EnvWrapper(env)  # type: ignore

    exp_manager = ExperimentManager(
        args={},  # type: ignore
        algo=algo,
        env_id=env_id,
        log_folder=logdir.as_posix(),
        tensorboard_log=tensorboard_log.as_posix() if tensorboard_log else "",
        n_timesteps=0,
        eval_freq=0,
        n_eval_episodes=0,
        save_freq=save_freq,
        hyperparams=agent_cfg,
        trained_agent=from_checkpoint.as_posix()
        if isinstance(from_checkpoint, Path)
        else from_checkpoint,
        optimize_hyperparameters=workflow == "optimize",
        n_trials=n_trials,
        sampler=sampler,
        pruner=pruner,
        n_startup_trials=n_startup_trials,
        n_evaluations=n_evaluations,
        truncate_last_trajectory=truncate_last_trajectory,
        seed=env_cfg.seed if env_cfg else 0,
        log_interval=log_interval,
        save_replay_buffer=save_replay_buffer,
        verbose=verbose,
        n_eval_envs=0,
        no_optim_plots=False,
        device=model_device,
        config=None,
        show_progress=True,
        env=env,
    )

    match workflow:
        case "train":
            agent_model, _saved_hyperparams = exp_manager.setup_experiment()  # type: ignore
            if eval_every_episodes > 0:
                exp_manager.callbacks.append(
                    EpisodeEvalVideoCallback(
                        logdir=logdir,
                        eval_every_episodes=eval_every_episodes,
                        eval_n_episodes=eval_n_episodes,
                        eval_video=eval_video,
                        eval_video_timesteps=eval_video_timesteps,
                        eval_video_prefix=eval_video_prefix,
                        concat_eval_videos=concat_eval_videos,
                        verbose=1 if verbose else 0,
                    )
                )
            exp_manager.learn(agent_model)
            exp_manager.save_trained_model(agent_model)
        case "optimize":
            exp_manager.setup_experiment()
            exp_manager.hyperparameters_optimization()
        case "eval":
            env = exp_manager.create_envs(0, eval_env=True)  # type: ignore

            if algo in OFF_POLICY_ALGOS:
                agent_cfg.update(dict(buffer_size=1))
                if "optimize_memory_usage" in agent_cfg:
                    agent_cfg.update(optimize_memory_usage=False)
            if "HerReplayBuffer" in agent_cfg.get("replay_buffer_class", ""):
                agent_cfg["env"] = env

            agent = ALGOS[algo].load(
                from_checkpoint.as_posix(),  # type: ignore
                device=model_device,
            )

            episode_start = numpy.ones(
                (env.unwrapped.num_envs,),  # type: ignore
                dtype=bool,
            )
            lstm_states = None

            obs = env.reset()
            for _ in tqdm(range(agent_cfg["n_timesteps"])):
                if not sim_app.is_running():
                    break
                action, lstm_states = agent.predict(
                    obs,  # type: ignore
                    state=lstm_states,
                    episode_start=episode_start,  # type: ignore
                    deterministic=True,
                )
                obs, _reward, episode_start, _infos = env.step(action)  # type: ignore


def gc_tqdm(*args):
    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, gc_tqdm)
