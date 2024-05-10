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




class Robomaster(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.cfg["env"]["numObservations"] = 10

        # 4个轮子速度，钳子关闭与否
        self.cfg["env"]["numActions"] = 4
        self.gripper_length = 0.25
        self.states = {}
        self.link_handles = {}
        self.joint_handles = {}
        self.actions = None
        self._ballA_state = None                # Current state of ballA for the current env
        # self._ballB_state = None
        self._ballA_id = None                   # Actor ID corresponding to ballA for a given env
        # self._ballB_id = None
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        self._root_state = None             # State of root body
        self._contact_forces = None     # Contact forces in sim
        # self._effort_control = None         # Torque actions
        self._vel_control = None 
        self._dof_control = None 
        self.up_axis = "z"
        self.up_axis_idx = 2
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(-0.55, 0.55, 2)
            cam_target = gymapi.Vec3(0, 0, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        self.robomaster_default_dof_pos = torch.zeros(self.num_robomaster_dofs, device=self.device)
        self.initial_root_state = torch.tensor([-0.45, 0, 0.001, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=self.device)
        self._ballA_state = torch.zeros(self.num_envs, 13, device=self.device)                # Current state of ballA for the current env
        # self._ballB_state = torch.zeros(self.num_envs, 13, device=self.device)
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
        plane_params.dynamic_friction = 100
        plane_params.static_friction = 100
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/robomasterEP_description/robot/robomaster.urdf"



        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False

        # asset_options.flip_visual_attachments = True

        # asset_options.use_mesh_materials = True
        robomaster_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # 可通过self.joint_handles得到
        self.vel_control_idx = torch.tensor([2, 18, 34, 50], dtype=torch.long)
        self.dof_control_idx = torch.tensor([0, 1], dtype=torch.long)
        robomaster_props = self.gym.get_asset_rigid_shape_properties(robomaster_asset)
        for p in robomaster_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.2
        self.gym.set_asset_rigid_shape_properties(robomaster_asset, robomaster_props)



        self.ballA_size = 0.03

        # Create ballA asset
        ballA_opts = gymapi.AssetOptions()
        ballA_opts.disable_gravity = False
        ballA_opts.density = 1
        ballA_opts.linear_damping = 1
        ballA_opts.angular_damping = 1
        ballA_asset = self.gym.create_sphere(self.sim, self.ballA_size, ballA_opts)
        ballA_color = gymapi.Vec3(0.6, 0.1, 0.0)


        self.num_robomaster_bodies = self.gym.get_asset_rigid_body_count(robomaster_asset)
        self.num_robomaster_dofs = self.gym.get_asset_dof_count(robomaster_asset)
        robomaster_dof_props = self.gym.get_asset_dof_properties(robomaster_asset)
        for i in range(self.num_robomaster_dofs):
            if i in self.vel_control_idx:
                robomaster_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
                robomaster_dof_props['stiffness'][i] = 0
                robomaster_dof_props['damping'][i] = 600
            elif i in self.dof_control_idx:
                robomaster_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                robomaster_dof_props['stiffness'][i] = 600
                robomaster_dof_props['damping'][i] = 600
            else:
                robomaster_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                robomaster_dof_props['stiffness'][i] = 0
                robomaster_dof_props['damping'][i] = 0
        # Define start pose for robomaster
        robomaster_start_pose = gymapi.Transform()
        robomaster_start_pose.p = gymapi.Vec3(-0.2, 0.0, 0)
        robomaster_start_pose.r = gymapi.Quat(0.0, 0.0, 0, 1.0)


        # Define start pose for balls (doesn't really matter since they're get overridden during reset() anyways)
        ballA_start_pose = gymapi.Transform()
        ballA_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        ballA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        self.robomasters = []
        self.envs = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.robomaster_actor = self.gym.create_actor(env_ptr, robomaster_asset, robomaster_start_pose, "robomaster", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.robomaster_actor, robomaster_dof_props)

            # Create balls
            self._ballA_id = self.gym.create_actor(env_ptr, ballA_asset, ballA_start_pose, "ballA", i, 2, 0)
            # self._ballB_id = self.gym.create_actor(env_ptr, ballB_asset, ballB_start_pose, "ballB", i, 3, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._ballA_id, 0, gymapi.MESH_VISUAL, ballA_color)
            self.envs.append(env_ptr)
            self.robomasters.append(self.robomaster_actor)

        # Setup init state buffer
        
        self._ballA_state = torch.zeros(self.num_envs, 13, device=self.device)

        self.init_data()
    
    def init_data(self):
        env_ptr = self.envs[0]
        robomaster_handle = 0
        self.link_handles = {
            "left_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "left_gripper_link_5"),
            "right_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "right_gripper_link_5"),
        }
        self.joint_handles = {
            'left_gripper_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "left_gripper_joint_1"),
            'right_gripper_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "right_gripper_joint_1"),
            'front_left_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "front_left_wheel_joint"),
            'front_right_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "front_right_wheel_joint"),
            'rear_left_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "rear_left_wheel_joint"),
            'rear_right_wheel_joint':self.gym.find_actor_dof_handle(env_ptr, robomaster_handle, "rear_right_wheel_joint"),
        }
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

        # self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.link_handles["left_gripper_link"], :2]
        self._eef_rf_state = self._rigid_body_state[:, self.link_handles["right_gripper_link"], :2]


        # Initialize actions
        # self._effort_control = torch.zeros((self.num_envs, self.num_robomaster_dofs), dtype=torch.float, device=self.device)
        self._vel_control = torch.zeros((self.num_envs, self.num_robomaster_dofs), dtype=torch.float, device=self.device)
        self._dof_control = torch.zeros((self.num_envs, self.num_robomaster_dofs), dtype=torch.float, device=self.device)
        # Initialize control
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_robomaster_reward(
            self.reset_buf, self.progress_buf, self._root_state,(self._eef_lf_state+self._eef_rf_state)/2, self.reward_settings, self.max_episode_length
        )


    def compute_observations(self, env_ids=None):
        robomaster_pos = torch.cat((self._root_state[:, 0,:2],self._root_state[:, 0, 6].view(-1, 1)),1)
        robomaster_vel = torch.cat((self._root_state[:, 0,7:9],self._root_state[:, 0, -1].view(-1, 1)),1)
        gripper_dof_pos = self._dof_pos[:, :2]
        gripper_pos = (self._eef_lf_state+self._eef_rf_state)/2
        ballA_pos = self._ballA_state[:, :2]
        ballA_vel = self._ballA_state[:, 7:9]
        self.obs_buf = torch.cat([robomaster_pos, robomaster_vel, gripper_dof_pos, ballA_pos - gripper_pos], dim=-1)
        return self.obs_buf


    def reset_idx(self, env_ids):
        # Reset balls, sampling ball B first, then A
        self._reset_init_ball_state(env_ids=env_ids)
        self._root_state[env_ids, 0] = self._reset_robomaster_state(env_ids)
        self._root_state[env_ids, self._ballA_id, :] = self._ballA_state[env_ids].clone()
        # self._root_state[env_ids, self._ballB_id, :] = self._ballB_state[env_ids].clone()

        pos = self.robomaster_default_dof_pos.unsqueeze(0)


        # Reset the internal obs accordingly
        self._dof_pos[env_ids, :] = pos.clone()
        self._dof_vel[env_ids, :] = torch.zeros_like(self._dof_vel[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)
        self._vel_control[env_ids, :] = torch.zeros_like(pos)
        self._dof_control[env_ids, :] = torch.zeros_like(pos)
        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_velocity_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._vel_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        
        # reinit dof state
        id_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.inital_dof_state),
                                              gymtorch.unwrap_tensor(id_int32), len(id_int32))

        # Update ball states
        multi_env_ids_int32 = self._global_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    
    def _reset_robomaster_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        num_resets = len(env_ids)
        state = torch.zeros(num_resets, 13, device=self.device)
        # state[:,:2] = 0.4*(torch.rand((num_resets, 2), device=self.device, dtype=torch.float32))
        state[:,:2] = torch.tensor([0, 0], device=self.device, dtype=torch.float32)
        state[:,2] = 0
        # state[:, 5:7] = (torch.rand((num_resets, 2), device=self.device, dtype=torch.float32)-0.5).squeeze()
        # state[:, 6] = 2*(torch.rand((num_resets), device=self.device, dtype=torch.float32)-0.5)
        state[:, 6] = 1
        # state[:, 5] = torch.sqrt(1 - state[:, 6]**2)
        return state
    
    def _reset_init_ball_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)

        ball_heights = self.ballA_size
        ballA_state = torch.zeros(num_resets, 13, device=self.device)
        ballA_state[:, 2] = ball_heights + 0.001
        ballA_state[:, 6] = 1.0
        theta = torch.rand(num_resets, device=self.device, dtype=torch.float32).view(-1, 1)*3.1416*2
        ballA_state[:, :2] = 0.5 * torch.cat((torch.abs(torch.cos(theta)), torch.sin(theta)), dim=1)
        # ballA_state[:, :2] = -torch.tensor([0.1, 0.1], device=self.device, dtype=torch.float32)-0.1*(torch.rand(ballA_state[:, :2].shape, device=self.device, dtype=torch.float32))
        self._ballA_state[env_ids, :] = ballA_state

    
        
    def mecanum_tranform(self, vel):
        action = torch.zeros((len(vel), 4), device=self.device, dtype=torch.float32)
        action[:, 0] = vel[:, 0] - vel[:,1] - vel[:, 2]
        action[:, 1] = vel[:, 0] + vel[:,1] + vel[:, 2]
        action[:, 2] = vel[:, 0] + vel[:,1] - vel[:, 2]
        action[:, 3] = vel[:, 0] - vel[:,1] + vel[:, 2]
        return action



    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # Split wheel and gripper command
        u_wheel, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        # Control arm (scale value first)
        # u_wheel = 10*torch.ones((self.num_envs, 3), device=self.device, dtype=torch.float32)
        # u_wheel = torch.tensor([[0, 0, 0]], device=self.device, dtype=torch.float32)
        # self._effort_control[:, self.control_idx[2:]] = 10*u_wheel.clone()
        self._vel_control[:, self.vel_control_idx] = 3*self.mecanum_tranform(u_wheel)
        u_fingers = torch.ones_like(self._vel_control[:, :2])
        # finger_pos = torch.where(u_gripper > 0, torch.ones_like(u_gripper), -torch.ones_like(u_gripper))
        # # finger_pos = 1
        # u_fingers[:, 0] = -finger_pos
        # u_fingers[:, 1] = finger_pos
        u_fingers[:, 0] = u_gripper
        u_fingers[:, 1] = -u_gripper
        # # Write gripper command to appropriate tensor buffer
        # self._effort_control[:, self.control_idx[:2]] = u_fingers.clone()
        self._dof_control[:, self.dof_control_idx] = u_fingers.clone()
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        # Deploy actions
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # self._refresh()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robomaster_reward(
    reset_buf, progress_buf, _root_state, eef_pos, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor,Tensor, Dict[str, float], float) -> Tuple[Tensor, Tensor]
    robomaster_pos = _root_state[:, 0, :3].clone()
    robomaster_vel = _root_state[:, 0,7:9].clone()
    ballA_pos = _root_state[:, 1, :2]
    ballA_vel = _root_state[:, 1, 7:9]
    # ballB_pos =  _root_state[:, 2, :2]
    norm_1 = torch.norm(ballA_pos - eef_pos, dim=1)
    dist_reward1 = -torch.tanh(norm_1)
    # dist_reward1 = torch.tanh(torch.sum((robomaster_vel - ballA_vel) * (ballA_pos - eef_pos), dim=1) / norm_1) * 0
    norm_2 = torch.norm(ballA_pos - robomaster_pos[:, :2], dim=1)
    dist_reward2 = torch.tanh(norm_2 - norm_1 - 0.25) * 0
    # norm_1 = torch.norm(eef_pos, dim=1)
    # dist_reward1 = -torch.tanh(torch.sum(robomaster_vel * eef_pos, dim=1)/norm_1)
    
    # print('robomaster',robomaster_pos[0])
    # print('ball',ballA_pos[0])
    # print('eef',eef_pos[0])
    # print('vel',robomaster_vel[0])

    # norm_2 = torch.norm(ballA_pos, dim=1)
    # dist_reward2 = -torch.tanh(torch.sum(ballA_vel * ballA_pos, dim=1)/norm_2)
    
    stack_reward = (norm_1 < 0.03)
    if torch.sum(stack_reward) > 0:
        print('sucess!')
    penalty1 = (torch.max(torch.abs(robomaster_pos), dim=1)[0]>1)
    penalty2 = (torch.max(torch.abs(ballA_pos), dim=1)[0]>1)
    penalty3 = (torch.max(torch.abs(_root_state[:, 0, 3:5]), dim=1)[0] > 0.3)
    # Compose rewards
    # rewards = stack_reward
    rewards = torch.where(stack_reward + penalty1 + penalty2 + penalty3 > 0,
                          reward_settings["r_stack_scale"] * (stack_reward),
                          reward_settings["r_dist_scale"] * (dist_reward1 + dist_reward2) )
    # rewards = reward_settings["r_stack_scale"] * (stack_reward- penalty1 - penalty2) + reward_settings["r_dist_scale"] * dist_reward1 + reward_settings["r_align_scale"] * dist_reward2 

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0) | (penalty1 > 0) | (penalty2 > 0) | (penalty3 > 0), torch.ones_like(reset_buf), reset_buf)
    
    return rewards, reset_buf