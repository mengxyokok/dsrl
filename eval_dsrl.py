"""
评估和可视化训练好的DSRL模型

用法:
    # 基本用法（自动查找最新checkpoint）
    python eval_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml
    
    # 指定模型路径（使用+前缀添加新配置项）
    python eval_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
        +model_path=logs/robomimic-dsrl/robomimic_can_dsrl_2025-11-28_11-19-35_1/2025-11-28_11-19-35_1/checkpoint/ft_policy_320000_steps.zip
    
    # 启用渲染和指定评估回合数
    python eval_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
        +model_path=xxx +render=true +n_episodes=20
    
    # 保存评估视频
    python eval_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
        +model_path=xxx +save_video=true +n_episodes=5
    
    # 使用随机策略评估
    python eval_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml \
        +deterministic=false
"""

import os
import warnings
warnings.filterwarnings("ignore")

# 抑制EGL清理时的警告（这些警告发生在析构函数中，不影响程序运行）
# 设置环境变量来抑制OpenGL错误输出
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# 在导入robosuite之前设置环境变量，避免Qt相关问题
# 禁用Qt平台插件（如果存在）
os.environ.pop("QT_QPA_PLATFORM", None)

import math
import torch
import random
import numpy as np
import hydra
from omegaconf import OmegaConf
import sys
import imageio
from datetime import datetime
sys.path.append('./dppo')
import gym, d4rl
import d4rl.gym_mujoco

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    # 从Hydra配置或命令行override中获取参数
    # 注意：使用+前缀添加新配置项，例如: +model_path=xxx
    # 或者使用环境变量: MODEL_PATH=xxx python eval_dsrl.py ...
    model_path = cfg.get('model_path', None) or os.environ.get('MODEL_PATH', None)
    render = cfg.get('render', False) or (os.environ.get('RENDER', 'false').lower() == 'true')
    n_episodes = cfg.get('n_episodes', None)
    if n_episodes is None:
        n_episodes = int(os.environ.get('N_EPISODES', '10'))
    deterministic_str = cfg.get('deterministic', None)
    if deterministic_str is None:
        deterministic = os.environ.get('DETERMINISTIC', 'true').lower() == 'true'
    else:
        deterministic = str(deterministic_str).lower() == 'true' if isinstance(deterministic_str, str) else bool(deterministic_str)
    
    # 视频保存相关参数
    save_video = cfg.get('save_video', False) or (os.environ.get('SAVE_VIDEO', 'false').lower() == 'true')
    video_dir = cfg.get('video_dir', None) or os.environ.get('VIDEO_DIR', None)
    
    # 如果没有指定模型路径，尝试自动查找最新的checkpoint
    if model_path is None:
        log_dir = cfg.logdir
        if os.path.exists(log_dir):
            checkpoint_dir = os.path.join(log_dir, 'checkpoint')
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoints:
                    # 按步数排序，选择最新的
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
                    model_path = os.path.join(checkpoint_dir, checkpoints[-1])
                    print(f"自动选择模型: {model_path}")
                else:
                    raise ValueError(f"在 {checkpoint_dir} 中未找到模型文件")
            else:
                raise ValueError(f"checkpoint目录不存在: {checkpoint_dir}")
        else:
            raise ValueError(f"日志目录不存在: {log_dir}。请使用 model_path=xxx 指定模型路径")
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    print(f"渲染模式: {render}")
    print(f"评估回合数: {n_episodes}")
    print(f"确定性策略: {deterministic}")
    print(f"保存视频: {save_video}")
    
    # 设置视频保存目录
    if save_video:
        if video_dir is None:
            # 默认保存在模型目录下的videos文件夹
            # model_path通常是: logs/.../checkpoint/ft_policy_xxx.zip
            # 我们想要: logs/.../videos/
            model_dir = os.path.dirname(model_path)  # checkpoint目录
            if 'checkpoint' in model_dir:
                model_dir = os.path.dirname(model_dir)  # 上一级目录
            video_dir = os.path.join(model_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        print(f"视频保存目录: {video_dir}")
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)
    
    # 创建评估环境（单个环境用于可视化）
    def make_eval_env():
        if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
            # 对于robomimic，需要直接创建环境以支持GUI渲染
            import robomimic.utils.env_utils as EnvUtils
            import robomimic.utils.obs_utils as ObsUtils
            import json
            
            obs_modality_dict = {
                "low_dim": cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
            }
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            
            robomimic_env_cfg_path = f'{cfg.dppo_path}/cfg/robomimic/env_meta/{cfg.env_name}.json'
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = False
            
            # 如果启用渲染或保存视频，修改env_meta中的has_renderer设置
            if render or save_video:
                env_meta["env_kwargs"]["has_renderer"] = render  # GUI渲染
                env_meta["env_kwargs"]["has_offscreen_renderer"] = save_video  # 离屏渲染用于视频录制
                # 设置MuJoCo使用glfw进行GUI渲染（需要X11显示）
                # 如果没有显示，可以使用osmesa进行软件渲染
                if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                    # 有X11显示，使用glfw
                    os.environ["MUJOCO_GL"] = "glfw"
                else:
                    # 无显示，使用osmesa软件渲染（但无法显示窗口）
                    print("警告: 未检测到DISPLAY环境变量，GUI渲染可能无法工作")
                    print("建议: 使用X11转发 (ssh -X) 或设置虚拟显示")
                    os.environ["MUJOCO_GL"] = "osmesa"
                # 禁用Qt平台插件（避免Qt相关错误）
                os.environ.pop("QT_QPA_PLATFORM", None)
            else:
                env_meta["env_kwargs"]["has_renderer"] = False
                env_meta["env_kwargs"]["has_offscreen_renderer"] = False
                # 无头渲染使用EGL
                if "MUJOCO_GL" not in os.environ:
                    os.environ["MUJOCO_GL"] = "egl"
            
            # 创建环境
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,  # GUI渲染
                render_offscreen=save_video,  # 离屏渲染用于视频录制
                use_image_obs=False,
            )
            env.env.hard_reset = False
            
            # 应用wrapper
            from dppo.env.gym_utils.wrapper import wrapper_dict
            wrappers = OmegaConf.create({
                'robomimic_lowdim': {
                    'normalization_path': cfg.normalization_path,
                    'low_dim_keys': cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                },
            })
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
            
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env
    
    # 加载基础策略
    base_policy = load_base_policy(cfg)
    
    # 创建单个环境（用于可视化）
    single_env = make_eval_env()
    
    # 保存底层robosuite环境的引用用于渲染
    robosuite_env = None
    if cfg.env_name in ['lift', 'can', 'square', 'transport'] and render:
        # 遍历wrapper找到robosuite环境
        env_temp = single_env
        while hasattr(env_temp, 'env'):
            env_temp = env_temp.env
            if hasattr(env_temp, 'sim') or hasattr(env_temp, 'viewer'):  # robosuite环境的特征
                robosuite_env = env_temp
                break
    
    if cfg.algorithm == 'dsrl_sac':
        eval_env = DiffusionPolicyEnvWrapper(DummyVecEnv([lambda: single_env]), cfg, base_policy)
    else:
        eval_env = DummyVecEnv([lambda: single_env])
    
    eval_env.seed(cfg.seed)
    
    # 加载模型
    print("正在加载模型...")
    if cfg.algorithm == 'dsrl_sac':
        model = SAC.load(model_path, env=eval_env, device=cfg.device)
    elif cfg.algorithm == 'dsrl_na':
        model = DSRL.load(model_path, env=eval_env, device=cfg.device)
    else:
        raise ValueError(f"未知的算法类型: {cfg.algorithm}")
    
    print("模型加载完成！")
    
    # 评估循环
    print("\n开始评估...")
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n回合 {episode + 1}/{n_episodes}")
        
        # 初始化视频录制
        video_frames = []
        video_writer = None
        if save_video:
            # 创建视频文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"episode_{episode+1:03d}_reward_{episode_reward:.2f}_{timestamp}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            try:
                video_writer = imageio.get_writer(video_path, fps=30)
                print(f"开始录制视频: {video_path}")
            except Exception as e:
                print(f"警告: 无法创建视频文件 {video_path}: {e}")
                video_writer = None
        
        step_count = 0
        while not done and step_count < MAX_STEPS:
            # 预测动作
            if cfg.algorithm == 'dsrl_sac':
                action, _ = model.predict(obs, deterministic=deterministic)
            elif cfg.algorithm == 'dsrl_na':
                action, _ = model.predict_diffused(obs, deterministic=deterministic)
            
            # 执行动作
            obs, reward, done, info = eval_env.step(action)
            
            # 处理向量化环境的返回值
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            episode_reward += reward
            episode_length += cfg.act_steps
            
            # 录制视频帧
            if save_video and video_writer is not None:
                try:
                    # 获取渲染帧（离屏渲染）
                    if cfg.env_name in ['lift', 'can', 'square', 'transport']:
                        # 对于robomimic环境，使用离屏渲染
                        if robosuite_env is not None:
                            # 使用robosuite的render方法获取RGB图像
                            frame = robosuite_env.sim.render(
                                camera_name="agentview",
                                width=512,
                                height=512,
                                depth=False
                            )
                            # robosuite返回的是翻转的图像，需要翻转回来
                            if frame is not None:
                                frame = np.flipud(frame)
                                video_writer.append_data(frame)
                        else:
                            # 尝试通过wrapper获取
                            wrapped_env = single_env
                            while hasattr(wrapped_env, 'env'):
                                wrapped_env = wrapped_env.env
                                if hasattr(wrapped_env, 'sim'):
                                    try:
                                        frame = wrapped_env.sim.render(
                                            camera_name="agentview",
                                            width=512,
                                            height=512,
                                            depth=False
                                        )
                                        if frame is not None:
                                            frame = np.flipud(frame)
                                            video_writer.append_data(frame)
                                        break
                                    except:
                                        pass
                    else:
                        # 对于其他环境，使用标准的render方法
                        if hasattr(eval_env, 'render'):
                            frame = eval_env.render(mode='rgb_array')
                            if frame is not None:
                                video_writer.append_data(frame)
                except Exception as e:
                    # 如果录制失败，打印警告但继续执行
                    if step_count % 50 == 0:  # 每50步打印一次，避免刷屏
                        print(f"视频录制警告: {e}")
            
            # 渲染 - 对于robomimic环境，需要调用底层robosuite环境的render方法
            if render:
                try:
                    if robosuite_env is not None:
                        # 直接调用robosuite环境的render方法
                        robosuite_env.render()
                    else:
                        # 尝试通过wrapper链调用render
                        if hasattr(eval_env, 'venv') and hasattr(eval_env.venv, 'envs'):
                            # 向量化环境，访问第一个环境
                            wrapped_env = eval_env.venv.envs[0]
                            # 遍历wrapper链找到底层环境
                            while hasattr(wrapped_env, 'env'):
                                wrapped_env = wrapped_env.env
                                if hasattr(wrapped_env, 'render'):
                                    wrapped_env.render()
                                    break
                        elif hasattr(eval_env, 'render'):
                            eval_env.render()
                except Exception as e:
                    # 如果渲染失败，打印错误但继续执行
                    if step_count % 10 == 0:  # 每10步打印一次，避免刷屏
                        print(f"渲染警告: {e}")
            
            step_count += 1
            
            # 检查episode是否结束
            if done:
                break
        
        # 关闭视频录制
        if save_video and video_writer is not None:
            try:
                video_writer.close()
                # 更新视频文件名（包含实际奖励）
                old_video_path = video_path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_video_filename = f"episode_{episode+1:03d}_reward_{episode_reward:.2f}_{timestamp}.mp4"
                new_video_path = os.path.join(video_dir, new_video_filename)
                if old_video_path != new_video_path and os.path.exists(old_video_path):
                    os.rename(old_video_path, new_video_path)
                    video_path = new_video_path
                print(f"视频已保存: {video_path}")
            except Exception as e:
                print(f"警告: 关闭视频文件时出错: {e}")
        
        # 判断成功标准：
        # 训练时使用的是：每一步reward > -reward_offset（即reward > -1）时标记为成功
        # 这里使用episode累积奖励：episode_reward > -reward_offset（即episode_reward > -1）
        # 
        # 奖励机制说明：
        # - 原始环境奖励：成功时=1.0，失败时=0.0
        # - 经过reward_offset处理：reward = 原始reward - 1
        #   * 原始1.0 → 处理后0.0
        #   * 原始0.0 → 处理后-1.0
        # - 如果episode_reward > -1，说明至少有一些步骤的原始奖励>0（有进展）
        success = episode_reward > -cfg.env.reward_offset
        
        if success:
            success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 打印详细信息，帮助诊断
        print(f"  奖励: {episode_reward:.2f}, 步数: {episode_length}, 成功: {success}")
        print(f"    成功阈值: episode_reward > {-cfg.env.reward_offset}")
        print(f"    平均每步奖励: {episode_reward/episode_length*cfg.act_steps:.4f} (原始环境奖励的近似)")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"总回合数: {n_episodes}")
    print(f"成功回合数: {success_count}")
    print(f"成功率: {success_count/n_episodes*100:.1f}%")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"最大奖励: {np.max(episode_rewards):.2f}")
    print(f"最小奖励: {np.min(episode_rewards):.2f}")
    print("="*50)
    
    # 优雅地关闭环境，避免EGL清理警告
    # 注意：如果看到 "Exception ignored in: <function MjRenderContext.__del__>" 错误
    # 这是无害的，发生在程序退出时EGL上下文清理过程中
    # 不影响评估结果的正确性
    try:
        # 先尝试正常关闭
        if hasattr(eval_env, 'close'):
            eval_env.close()
        
        # 如果是VecEnv，关闭所有子环境
        if hasattr(eval_env, 'venv') and hasattr(eval_env.venv, 'envs'):
            for sub_env in eval_env.venv.envs:
                if hasattr(sub_env, 'close'):
                    try:
                        sub_env.close()
                    except Exception:
                        pass
        
        # 关闭底层robosuite环境（如果存在）
        if robosuite_env is not None and hasattr(robosuite_env, 'close'):
            try:
                robosuite_env.close()
            except Exception:
                pass
        
        # 关闭单个环境（如果存在）
        if 'single_env' in locals() and hasattr(single_env, 'close'):
            try:
                single_env.close()
            except Exception:
                pass
                
    except Exception as e:
        # 忽略关闭时的错误（通常是EGL清理警告）
        # 这些错误发生在析构函数中，不会影响程序运行
        # 评估结果已经正确输出，可以安全忽略
        pass
    
    # 如果使用了渲染，尝试清理EGL上下文
    if render:
        try:
            # 清理OpenGL/EGL相关资源
            import gc
            gc.collect()  # 强制垃圾回收，帮助清理渲染资源
        except Exception:
            pass


if __name__ == "__main__":
    main()

