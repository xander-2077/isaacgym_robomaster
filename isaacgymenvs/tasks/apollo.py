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




class Apollo(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.cfg["env"]["numObservations"] = 15

        # 2个前轮，三个舵
        self.cfg["env"]["numActions"] = 3
        
        self.states = {}
        self.joint_handles = {}
        self.actions = None
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        self._root_state = None             # State of root body
        self._vel_control = None 
        self._pos_control = None
        self.up_axis = "z"
        self.up_axis_idx = 2
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(-0.55, 0.55, 4)
            cam_target = gymapi.Vec3(-0.1, -0.1, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.apollo_default_dof_pos = torch.zeros(self.num_apollo_dofs, device=self.device)
        self.initial_root_state = torch.tensor([-0.45, 0, 0.001, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=self.device) 
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # self._refresh()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        
    
    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.dynamic_friction = 2
        plane_params.static_friction = 2
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)




        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/apollo/apollo.urdf"
        obstacle_asset_file = "cylinder_robot.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False

        # asset_options.flip_visual_attachments = True

        # asset_
        # options.use_mesh_materials = True
        apollo_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        obstacle_asset = self.gym.load_asset(self.sim, asset_root, obstacle_asset_file, asset_options)
        # 可通过self.joint_handles得到
        self.vel_control_idx = torch.tensor([3, 5], dtype=torch.long)
        self.pos_control_idx = torch.tensor([2, 4], dtype=torch.long)
        apollo_props = self.gym.get_asset_rigid_shape_properties(apollo_asset)
        for p in apollo_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.2
        obstacle_props = self.gym.get_asset_rigid_shape_properties(obstacle_asset)
        for p in obstacle_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.2
        self.gym.set_asset_rigid_shape_properties(apollo_asset, apollo_props)
        self.gym.set_asset_rigid_shape_properties(obstacle_asset, obstacle_props)

        self.num_apollo_bodies = self.gym.get_asset_rigid_body_count(apollo_asset)
        self.num_apollo_dofs = self.gym.get_asset_dof_count(apollo_asset)
        apollo_dof_props = self.gym.get_asset_dof_properties(apollo_asset)
        for i in range(self.num_apollo_dofs):
            if i in self.vel_control_idx:
                apollo_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL 
                apollo_dof_props['stiffness'][i] = 0
                apollo_dof_props['damping'][i] = 600
            elif i in self.pos_control_idx:
                apollo_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                apollo_dof_props['stiffness'][i] = 600
                apollo_dof_props['damping'][i] = 600
            else:
                apollo_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                apollo_dof_props['stiffness'][i] = 0
                apollo_dof_props['damping'][i] = 0

        self.num_obstacle_bodies = self.gym.get_asset_rigid_body_count(obstacle_asset)
        self.num_obstacle_dofs = self.gym.get_asset_dof_count(obstacle_asset)
        obstacle_dof_props = self.gym.get_asset_dof_properties(obstacle_asset)
        for i in range(self.num_obstacle_dofs):
            if i in self.vel_control_idx:
                obstacle_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL 
                obstacle_dof_props['stiffness'][i] = 0
                obstacle_dof_props['damping'][i] = 600
            elif i in self.pos_control_idx:
                obstacle_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                obstacle_dof_props['stiffness'][i] = 600
                obstacle_dof_props['damping'][i] = 600
            else:
                obstacle_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                obstacle_dof_props['stiffness'][i] = 0
                obstacle_dof_props['damping'][i] = 0  


        # Define start pose for apollo
        apollo_start_pose = gymapi.Transform()
        apollo_start_pose.p = gymapi.Vec3(0, 0.0, 0)
        apollo_start_pose.r = gymapi.Quat(0.0, 0.0, 0, 1.0)

        obstacle_start_pose = gymapi.Transform()
        obstacle_start_pose.p = gymapi.Vec3(0, 0.0, 0)
        obstacle_start_pose.r = gymapi.Quat(0.0, 0.0, 0, 1.0)


        self.apollos1 = []
        self.apollos2 = []
        self.obstacle = []
        self.envs = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.apollo_actor1 = self.gym.create_actor(env_ptr, apollo_asset, apollo_start_pose, "apollo1", i, 1, 0)
            self.apollo_actor2 = self.gym.create_actor(env_ptr, apollo_asset, apollo_start_pose, "apollo2", i, 2, 0)
            self.obstacle_actor = self.gym.create_actor(env_ptr, obstacle_asset, obstacle_start_pose, "obstacle", i, 3, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.apollo_actor1, apollo_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, self.apollo_actor2, apollo_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, self.obstacle_actor, obstacle_dof_props)
            self.envs.append(env_ptr)
            self.apollos1.append(self.apollo_actor1)
            self.apollos2.append(self.apollo_actor2)
            self.obstacle.append(self.obstacle_actor)


        self.init_data()
    
    def init_data(self):
        env_ptr = self.envs[0]
        apollo_handle = 0
        self.joint_handles = {
            'front_left_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "front_left_wheel_joint"),
            'front_right_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "front_right_wheel_joint"),
            'back_left_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "back_left_wheel_joint"),
            'back_right_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "back_right_wheel_joint"),
            'front_right_bar_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "front_right_bar_joint"),
            'front_left_bar_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "front_left_bar_joint"),
            'steer_joint':self.gym.find_actor_dof_handle(env_ptr, apollo_handle, "steer_joint"),
        }
        print(self.joint_handles)
        self._root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        self._root_state = gymtorch.wrap_tensor(self._root_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)
        
        # 关节位置，第一列为关节角度，第二列为关节速度
        self._dof_state = gymtorch.wrap_tensor(self._dof_state_tensor).view(self.num_envs, -1, 2).view(self.num_envs, -1, 2)
        self.inital_dof_state = self._dof_state.clone()

        # link位置
        self._rigid_body_state = gymtorch.wrap_tensor(self._rigid_body_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)

        self._dof_pos = self._dof_state[..., 0]
        self._dof_vel = self._dof_state[..., 1]

        self._pos_control = torch.zeros((self.num_envs, 3, self.num_apollo_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros((self.num_envs, 3, self.num_apollo_dofs), dtype=torch.float, device=self.device)
        # Initialize control
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_apollo_reward(
            self.reset_buf, self.progress_buf, self._root_state, self.reward_settings, self.max_episode_length
        )


    def compute_observations(self, env_ids=None):
        apollo1_pos = torch.cat((self._root_state[:, 0,:2],self._root_state[:, 0, 6].view(-1, 1)),1)
        apollo1_vel = torch.cat((self._root_state[:, 0,7:9],self._root_state[:, 0, -1].view(-1, 1)),1)

        apollo2_pos = torch.cat((self._root_state[:, 1,:2],self._root_state[:, 1, 6].view(-1, 1)),1)
        apollo2_vel = torch.cat((self._root_state[:, 1,7:9],self._root_state[:, 1, -1].view(-1, 1)),1)

        obstacle_pos = torch.cat((self._root_state[:, 2,:2],self._root_state[:, 2, 6].view(-1, 1)),1)
        obstacle_vel = torch.cat((self._root_state[:, 2,7:9],self._root_state[:, 2, -1].view(-1, 1)),1)
        self.obs_buf = torch.cat([apollo1_pos - apollo2_pos, apollo1_pos - obstacle_pos, apollo1_vel, apollo2_vel, obstacle_vel], dim=-1)
        return self.obs_buf


    def reset_idx(self, env_ids):

        self._root_state[env_ids] = self._reset_apollo_state(env_ids)

        # Reset the internal obs accordingly
        # self._dof_pos[env_ids, :] = pos.clone()
        # self._dof_vel[env_ids, :] = torch.zeros_like(self._dof_vel[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)
        
        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.inital_dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32))
        # self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._vel_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        
        # # reinit dof state
        # id_int32 = self._global_indices[env_ids, 0].flatten()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._pos_control),
        #                                       gymtorch.unwrap_tensor(id_int32), len(id_int32))

        multi_env_ids_int32 = self._global_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    
    def _reset_apollo_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        num_resets = len(env_ids)
        state = torch.zeros((num_resets, 3, 13), device=self.device, dtype=torch.float)
        state[:, 0, :2] = -0.5 - 0.5 * (torch.rand((num_resets, 2), device=self.device, dtype=torch.float32))
        state[:, 1, :2] = 0.5 + 0.5 * (torch.rand((num_resets, 2), device=self.device, dtype=torch.float32))
        state[:, 2, :2] = 0 + 0.2 *(torch.rand((num_resets, 2), device=self.device, dtype=torch.float32))
        state[:, :, 2] = 0
        # state[:, 5:7] = (torch.rand((num_resets, 2), device=self.device, dtype=torch.float32)-0.5).squeeze()
        state[:,:, 6] = 1
        return state
    

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        u_wheel, u_steer = self.actions[:, : 2], self.actions[:, 2].unsqueeze(1)
        # Control arm (scale value first)
        # u_wheel = 10*torch.ones((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # u_wheel = torch.tensor([[1, 1]], device=self.device, dtype=torch.float32)
        # u_steer= torch.tensor([[1, 1]], device=self.device, dtype=torch.float32)
        # self._effort_control[:, self.control_idx[2:]] = 10*u_wheel.clone()
        self._vel_control[:, 0, self.vel_control_idx] = 10*u_wheel
        self._pos_control[:, 0, self.pos_control_idx] = torch.cat((u_steer, u_steer), dim=1)
        u_wheel1 = torch.ones((self.num_envs, 2), device=self.device, dtype=torch.float32)
        u_steer1 = torch.ones((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._vel_control[:, 1, self.vel_control_idx] = 3*u_wheel1
        self._pos_control[:, 1, self.pos_control_idx] = u_steer1.clone()

        u_wheel2 = -torch.ones((self.num_envs, 2), device=self.device, dtype=torch.float32)
        u_steer2 = torch.ones((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._vel_control[:, 2, self.vel_control_idx] = 3*u_wheel2
        self._pos_control[:, 2, self.pos_control_idx] = u_steer2.clone()

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        
    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_apollo_reward(
    reset_buf, progress_buf, _root_state,  reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, float], float) -> Tuple[Tensor, Tensor]
    apollo1_pos = _root_state[:, 0, :2].clone()
    apollo1_vel = _root_state[:, 0, 7:9].clone()
    apollo2_pos = _root_state[:, 1, :2].clone()
    apollo2_vel = _root_state[:, 1,7:9].clone()
    obstacle_pos = _root_state[:, 2, :2].clone()
    norm_1 = torch.norm(apollo1_pos - apollo2_pos, dim=1)
    dist_reward1 = -torch.tanh(torch.sum((apollo1_vel - apollo2_vel) * (apollo1_pos - apollo2_pos), dim=1) / norm_1)

    norm_2 = torch.norm(apollo1_pos - obstacle_pos, dim=1)
    stack_reward = (norm_1 < 0.4)
    if torch.sum(stack_reward) > 0:
        print('sucess!')
    penalty1 = (norm_2 < 0.4)
    
    penalty2 = (torch.max(torch.abs(_root_state[:, 0, 3:5]), dim=1)[0] > 0.3)
    # Compose rewards

    rewards = torch.where(stack_reward + penalty1 + penalty2 > 0,
                          reward_settings["r_stack_scale"] * (stack_reward - 0.5 * penalty1),
                          reward_settings["r_dist_scale"] * (dist_reward1) )
    # rewards = reward_settings["r_stack_scale"] * (stack_reward- penalty1 - penalty2) + reward_settings["r_dist_scale"] * dist_reward1 + reward_settings["r_align_scale"] * dist_reward2 

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0)| (penalty1 > 0) | (penalty2 > 0), torch.ones_like(reset_buf), reset_buf)
    
    return rewards, reset_buf