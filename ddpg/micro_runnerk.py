import imageio
from pathlib import Path
import shutil
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import numpy as np
import matplotlib.pyplot as plt

from micro_gym import MicroEnv, pid_loop

def create_gif(run_name: str, png_folder: Path = (Path.cwd() / 'runs')):
    png_list = list(png_folder.glob('*.png'))
    num_list = sorted([int(png.stem) for png in png_list])
    png_list = [(png_folder / f'{i}.png') for i in num_list]
    with imageio.get_writer((png_folder / f'{run_name}.gif'), mode='I') as writer:
        for filepath in png_list[::2]:
            image = imageio.imread(filepath)
            writer.append_data(image)
        for filepath in png_list:
            filepath.unlink()

def control_loop(model, env, render=False):
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        if terminated or truncated:
            done = True

    if render:
        env.render()

    return episode_reward, episode_length

def noise_loop(model_path: Path, run_kwargs: dict):
    ddpg_controller = DDPG.load(model_path)
    run_folder = model_path.parent

    noise_levels = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2]

    avg_rewards = []
    std_rewards = []
    avg_lengths = []
    std_lengths = []

    for noise in noise_levels:
        run_kwargs['noise'] = noise
        env = MicroEnv(**run_kwargs)

        rewards = []
        lengths = []
        for _ in range(50):
            reward, length = control_loop(ddpg_controller, env)
            rewards.append(reward)
            lengths.append(length)

        avg_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
        avg_lengths.append(np.mean(lengths))
        std_lengths.append(np.std(lengths))

    avg_rewards = np.array(avg_rewards)
    std_rewards = np.array(std_rewards)
    avg_lengths = np.array(avg_lengths)
    std_lengths = np.array(std_lengths)
    noise_levels = np.array(noise_levels) * 100

    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=12)
    plt.rc('font', size=14)

    plt.clf()
    plt.errorbar(noise_levels, avg_rewards, yerr=std_rewards, fmt='-o', capsize=3)
    plt.fill_between(noise_levels, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2)
    plt.xlabel('Noise (standard power units)')
    plt.ylabel('Reward')
    plt.title('Reward vs Noise')
    plt.savefig(run_folder / 'reward_vs_noise.png')

    plt.clf()
    plt.errorbar(noise_levels, avg_lengths, yerr=std_lengths, fmt='-o', capsize=3)
    plt.fill_between(noise_levels, avg_lengths - std_lengths, avg_lengths + std_lengths, alpha=0.2)
    plt.xlabel('Noise (standard power units)')
    plt.ylabel('Episode Length')
    plt.title('Episode Length vs Noise')
    plt.savefig(run_folder / 'length_vs_noise.png')

def main(args):
    run_name = args.run_name
    run_folder = Path.cwd() / 'runs' / run_name
    run_folder.mkdir(exist_ok=True, parents=True)

    latest_model = list(run_folder.glob('*.zip'))
    latest_model = max(latest_model, key=lambda x: x.stat().st_mtime) if latest_model else None

    run_kwargs = {'run_name': run_name,
                  'run_mode': args.run_type,
                  'noise': args.noise,
                  'profile': args.profile,
                  'reward_mode': args.reward,
                  'scale_graphs': True}

    eval_env = MicroEnv(**run_kwargs)

    if args.run_type == 'train':
        tensorboard_dir = f'./runs/{run_name}/logs/'
        vec_env = make_vec_env(MicroEnv, n_envs=args.num_envs, env_kwargs=run_kwargs)
        vec_env = VecMonitor(vec_env, filename=f'./runs/{run_name}/logs/vec')

        if latest_model:
            model = DDPG.load(latest_model, env=vec_env, verbose=1, tensorboard_log=tensorboard_dir)
        else:
            model = DDPG('MultiInputPolicy', vec_env, verbose=1, tensorboard_log=tensorboard_dir)

        eval_callback = EvalCallback(eval_env, best_model_save_path=f'./runs/{run_name}',
                                      log_path=f'./runs/{run_name}/logs/',
                                      deterministic=True, eval_freq=4000)

        model.learn(total_timesteps=args.num_timesteps, callback=eval_callback, progress_bar=True)
        model.save(f'./runs/{run_name}/{model.num_timesteps}.zip')

    elif args.run_type == 'noisetest':
        noise_loop(latest_model, run_kwargs)
        return

    if latest_model:
        ddpg_controller = DDPG.load(latest_model)
        reward, length = control_loop(ddpg_controller, eval_env, render=True)
        print(f'Episode reward: {reward}')
        print(f'Episode length: {length}')
    else:
        print('Error: no trained model found')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate DDPG on MicroEnv')
    parser.add_argument('run_name', type=str, help='Name of the run')
    parser.add_argument('--run_type', type=str, default='train',
                        help='Must be train, load, or noisetest')
    parser.add_argument('--num_timesteps', type=int, default=800000)
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--noise', type=float, default=0.0, help='Observation noise level')
    parser.add_argument('--reward', type=str, default='optimal', help='Reward mode')
    parser.add_argument('--profile', type=str, default='train', help='Training profile')

    args = parser.parse_args()
    main(args)

