# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask



@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

class Robomaster(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robomaster_dof_noise = self.cfg["env"]["robomasterDofNoise"]
        self.robomaster_position_noise = self.cfg["env"]["robomasterPositionNoise"]
        self.robomaster_rotation_noise = self.cfg["env"]["robomasterRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # 机器人坐标朝向，钳子角度，钳子位置，球位置，目标位置，3 + 2 + 4 + 2 + 2
        self.cfg["env"]["numObservations"] = 13

        # 4个轮子速度，钳子关闭与否
        self.cfg["env"]["numActions"] = 5
        
        self.states = {}
        self.handles = {}
        self.actions = None
        self._init_ballA_state = None           # Initial state of ballA for the current env
        self._init_ballB_state = None
        self._ballA_state = None                # Current state of ballA for the current env
        self._ballB_state = None
        self._ballA_id = None                   # Actor ID corresponding to ballA for a given env
        self._ballB_id = None
        self.robomaster_dof_lower_limits = []
        self.robomaster_dof_upper_limits = []
        self._robomaster_effort_limits = []
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        self._root_state = None             # State of root body
        self._contact_forces = None     # Contact forces in sim
        self._wheel_control = None  # Tensor buffer for controlling wheels
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.robomaster_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0.035, 0.035], device=self.device
        )
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.1], device=self.device).unsqueeze(0)
        
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

        
    
    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/robomasterEP_description/robot/robomaster.urdf"
  
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = True
        # asset_options.use_mesh_materials = True
        robomaster_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        robomaster_dof_stiffness = to_torch([0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        robomaster_dof_damping = to_torch([0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        

        self.ballA_size = 0.050
        self.ballB_size = 0.070

        # Create ballA asset
        ballA_opts = gymapi.AssetOptions()
        ballA_asset = self.gym.create_sphere(self.sim, self.ballA_size, ballA_opts)
        ballA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create ballB asset
        ballB_opts = gymapi.AssetOptions()
        ballB_opts.disable_gravity = True
        ballB_asset = self.gym.create_sphere(self.sim, self.ballB_size, ballB_opts)
        ballB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_robomaster_bodies = self.gym.get_asset_rigid_body_count(robomaster_asset)
        self.num_robomaster_dofs = self.gym.get_asset_dof_count(robomaster_asset)

        robomaster_dof_props = self.gym.get_asset_dof_properties(robomaster_asset)
        for i in range(self.num_robomaster_dofs):
            robomaster_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 4 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                robomaster_dof_props['stiffness'][i] = robomaster_dof_stiffness[i]
                robomaster_dof_props['damping'][i] = robomaster_dof_damping[i]
            else:
                robomaster_dof_props['stiffness'][i] = 7000.0
                robomaster_dof_props['damping'][i] = 50.0

            self.robomaster_dof_lower_limits.append(robomaster_dof_props['lower'][i])
            self.robomaster_dof_upper_limits.append(robomaster_dof_props['upper'][i])
            self._robomaster_effort_limits.append(robomaster_dof_props['effort'][i])
        self.robomaster_dof_lower_limits = to_torch(self.robomaster_dof_lower_limits, device=self.device)
        self.robomaster_dof_upper_limits = to_torch(self.robomaster_dof_upper_limits, device=self.device)
        self._robomaster_effort_limits = to_torch(self._robomaster_effort_limits, device=self.device)
        self.robomaster_dof_speed_scales = torch.ones_like(self.robomaster_dof_lower_limits)
        self.robomaster_dof_speed_scales[[4, 5]] = 0.1
        robomaster_dof_props['effort'][4] = 200
        robomaster_dof_props['effort'][5] = 200

        # Define start pose for robomaster
        robomaster_start_pose = gymapi.Transform()
        robomaster_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2)
        robomaster_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        # Define start pose for balls (doesn't really matter since they're get overridden during reset() anyways)
        ballA_start_pose = gymapi.Transform()
        ballA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        ballA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        ballB_start_pose = gymapi.Transform()
        ballB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        ballB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        num_robomaster_bodies = self.gym.get_asset_rigid_body_count(robomaster_asset)
        num_robomaster_shapes = self.gym.get_asset_rigid_shape_count(robomaster_asset)
        max_agg_bodies = num_robomaster_bodies + 3     # 1 for table, ballA, ballB
        max_agg_shapes = num_robomaster_shapes + 3

        self.robomasters = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: robomaster should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robomaster
            # Potentially randomize start pose
            if self.robomaster_position_noise > 0:
                rand_xy = self.robomaster_position_noise * (-1. + np.random.rand(2) * 2.0)
                robomaster_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2)
            if self.robomaster_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.robomaster_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                robomaster_start_pose.r = gymapi.Quat(*new_quat)
            robomaster_actor = self.gym.create_actor(env_ptr, robomaster_asset, robomaster_start_pose, "robomaster", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, robomaster_actor, robomaster_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create balls
            self._ballA_id = self.gym.create_actor(env_ptr, ballA_asset, ballA_start_pose, "ballA", i, 2, 0)
            self._ballB_id = self.gym.create_actor(env_ptr, ballB_asset, ballB_start_pose, "ballB", i, 3, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._ballA_id, 0, gymapi.MESH_VISUAL, ballA_color)
            self.gym.set_rigid_body_color(env_ptr, self._ballB_id, 0, gymapi.MESH_VISUAL, ballB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robomasters.append(robomaster_actor)

        # Setup init state buffer
        
        self._init_ballA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_ballB_state = torch.zeros(self.num_envs, 13, device=self.device)

        self.init_data()
    
    def init_data(self):
        env_ptr = self.envs[0]
        robomaster_handle = 0
        self.handles = {
            # robomaster
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "base_link"),
            "left_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "left_gripper_link_1"),
            "right_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "right_gripper_link_1"),
            # balls
            "ballA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._ballA_id, "box"),
            "ballB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._ballB_id, "box"),
        }

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)

        # 关节位置，第一列为关节角度，第二列为关节速度
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2).view(self.num_envs, -1, 2)

        # link位置
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)

        self._dof_pos = self._dof_state[..., 0]
        self._dof_vel = self._dof_state[..., 1]

        self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["left_gripper_link"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["right_gripper_link"], :]

        self._ballA_state = self._root_state[:, self._ballA_id, :]
        self._ballB_state = self._root_state[:, self._ballB_id, :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._wheel_control = self._effort_control[:, :4]
        self._gripper_control = self._pos_control[:, 4:6]
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        
    
    def _update_states(self):
        self.states.update({
            # robomaster base position: pos=[:3], angle=[3:7]
            "robomaster_pos": self._base_link_state[:, :3],
            "gripper_pos": self._dof_pos[:, -2:],
            "eef_lf_pos": self._eef_lf_state[:, :2],
            "eef_rf_pos": self._eef_rf_state[:, :2],
            # balls
            "ballA_pos": self._ballA_state[:, :2],
            "ballB_pos": self._ballB_state[:, :2],
            "ballA_to_ballB_pos": self._ballB_state[:, :2] - self._ballA_state[:, :2],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Refresh states
        self._update_states()


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_robomaster_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        self._refresh()
        obs = ["ballA_pos", "ballA_to_ballB_pos", "robomaster_pos", "gripper_pos", 'eef_lf_pos', 'eef_rf_pos']
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset balls, sampling ball B first, then A
        # if not self._i:
        self._reset_init_ball_state(ball='B', env_ids=env_ids, check_valid=False)
        self._reset_init_ball_state(ball='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._ballA_state[env_ids] = self._init_ballA_state[env_ids]
        self._ballB_state[env_ids] = self._init_ballB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        pos = tensor_clamp(
            self.robomaster_default_dof_pos.unsqueeze(0) +
            self.robomaster_dof_noise * 2.0 * (reset_noise - 0.5),
            self.robomaster_dof_lower_limits.unsqueeze(0), self.robomaster_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.robomaster_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._dof_pos[env_ids, :] = pos
        self._dof_vel[env_ids, :] = torch.zeros_like(self._dof_vel[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._pos_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                       len(multi_env_ids_int32))

        # Update ball states
        multi_env_ids_balls_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_balls_int32), len(multi_env_ids_balls_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    
    def _reset_init_ball_state(self, ball, env_ids, check_valid=True):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_ball_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if ball.lower() == 'a':
            this_ball_state_all = self._init_ballA_state
            other_ball_state = self._init_ballB_state[env_ids, :]
            ball_heights = self.ballA_size*2
        elif ball.lower() == 'b':
            this_ball_state_all = self._init_ballB_state
            other_ball_state = self._init_ballA_state[env_ids, :]
            ball_heights = self.ballB_size*2
        else:
            raise ValueError(f"Invalid ball specified, options are 'A' and 'B'; got: {ball}")

        # Minimum ball distance for guarenteed collision-free sampling is the sum of each ball's effective radius
        min_dists = (self.ballA_size + self.ballB_size) * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the balls aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_ball_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_ball_state[:, 2] = self._table_surface_pos[2] + ball_heights
        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_ball_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on balls' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_ball_state[active_idx, :2] = centered_ball_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_ball_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                ball_dist = torch.linalg.norm(sampled_ball_state[:, :2] - other_ball_state[:, :2], dim=-1)
                active_idx = torch.nonzero(ball_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling ball locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_ball_state[:, :2] = centered_ball_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Lastly, set these sampled values as the new init state
        
        this_ball_state_all[env_ids, :] = sampled_ball_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # Split wheel and gripper command
        u_wheel, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # Control arm (scale value first)
        u_wheel = u_wheel * self.cmd_limit / self.action_scale
        # u_wheel = self._compute_osc_torques(dpose=u_wheel)
        self._wheel_control[:, :] = u_wheel*0

        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.robomaster_dof_upper_limits[-2].item(),
                                      self.robomaster_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.robomaster_dof_upper_limits[-1].item(),
                                      self.robomaster_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers
        # Deploy actions
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robomaster_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]


    # distance from hand to the ballA
    d_lf = torch.norm(states["ballA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["ballA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d_lf + d_rf) / 2)


    # how closely aligned ballA is to ballB (only provided if ballA is lifted)
    d_ab = torch.norm(states["ballA_to_ballB_pos"], dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_ab)) 

    # Dist reward is maximum of dist and align reward
    dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if ballA is close to target height and corresponding location, and gripper is not grasping)
    ballA_align_ballB = (torch.norm(states["ballA_to_ballB_pos"][:, :2], dim=-1) < 0.02)
    stack_reward = ballA_align_ballB

    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward,
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_align_scale"] * align_reward,
    )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf