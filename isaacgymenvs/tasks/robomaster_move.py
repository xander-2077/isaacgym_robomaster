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

        # 机器人坐标朝向，钳子角度，球位置，目标位置，3 + 2 + 2 + 2
        self.cfg["env"]["numObservations"] = 18

        # 4个轮子速度，钳子关闭与否
        self.cfg["env"]["numActions"] = 5
        
        self.states = {}
        self.handles = {}
        self.actions = None
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
        self._effort_control = None         # Torque actions

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.robomaster_default_dof_pos = to_torch(
            [0, 0, 0, 0, 0, 0], device=self.device
        )
        self._ballA_state = torch.zeros(self.num_envs, 13, device=self.device)                # Current state of ballA for the current env
        self._ballB_state = torch.zeros(self.num_envs, 13, device=self.device)
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
        # asset_file = "urdf/rm_ep/robomaster_ep.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        # asset_options.flip_visual_attachments = True
        # asset_options.use_mesh_materials = True
        robomaster_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        robomaster_props = self.gym.get_asset_rigid_shape_properties(robomaster_asset)
        for p in robomaster_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(robomaster_asset, robomaster_props)

        robomaster_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        robomaster_dof_damping = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)


        self.ballA_size = 0.030
        self.ballB_size = 0.030

        # Create ballA asset
        ballA_opts = gymapi.AssetOptions()
        ballA_opts.disable_gravity = False
        ballA_asset = self.gym.create_sphere(self.sim, self.ballA_size, ballA_opts)
        ballA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create ballB asset
        ballB_opts = gymapi.AssetOptions()
        ballB_opts.disable_gravity = True
        ballB_asset = self.gym.create_sphere(self.sim, self.ballB_size, ballB_opts)
        ballB_color = gymapi.Vec3(0.0, 0.0, 0.0)

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
        robomaster_start_pose.p = gymapi.Vec3(-0.6, 0.0, 0)
        robomaster_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 1.0)


        # Define start pose for balls (doesn't really matter since they're get overridden during reset() anyways)
        ballA_start_pose = gymapi.Transform()
        ballA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        ballA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        ballB_start_pose = gymapi.Transform()
        ballB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        ballB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # num_robomaster_bodies = self.gym.get_asset_rigid_body_count(robomaster_asset)
        # num_robomaster_shapes = self.gym.get_asset_rigid_shape_count(robomaster_asset)


        self.robomasters = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: robomaster should ALWAYS be loaded first in sim!
            # if self.aggregate_mode >= 3:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self.robomaster_actor = self.gym.create_actor(env_ptr, robomaster_asset, robomaster_start_pose, "robomaster", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.robomaster_actor, robomaster_dof_props)
            # if self.aggregate_mode == 2:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # if self.aggregate_mode == 1:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create balls
            self._ballA_id = self.gym.create_actor(env_ptr, ballA_asset, ballA_start_pose, "ballA", i, 1, 0)
            self._ballB_id = self.gym.create_actor(env_ptr, ballB_asset, ballB_start_pose, "ballB", i, 3, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._ballA_id, 0, gymapi.MESH_VISUAL, ballA_color)
            self.gym.set_rigid_body_color(env_ptr, self._ballB_id, 0, gymapi.MESH_VISUAL, ballB_color)

            # if self.aggregate_mode > 0:
            #     self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robomasters.append(self.robomaster_actor)

        # Setup init state buffer
        
        self._ballA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._ballB_state = torch.zeros(self.num_envs, 13, device=self.device)

        self.init_data()
    
    def init_data(self):
        env_ptr = self.envs[0]
        robomaster_handle = 0
        self.handles = {
            # robomaster
            # "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "base_link"),
            "left_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "left_gripper_link_1"),
            "right_gripper_link": self.gym.find_actor_rigid_body_handle(env_ptr, robomaster_handle, "right_gripper_link_1"),
            # balls
            # "ballA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._ballA_id, "box"),
            # "ballB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._ballB_id, "box"),
        }

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        
        self._root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        self._root_state = gymtorch.wrap_tensor(self._root_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)
        self.initial_root_state = self._root_state.clone()
        # 关节位置，第一列为关节角度，第二列为关节速度
        self._dof_state = gymtorch.wrap_tensor(self._dof_state_tensor).view(self.num_envs, -1, 2).view(self.num_envs, -1, 2)
        self.inital_dof_state = self._dof_state.clone()

        # link位置
        self._rigid_body_state = gymtorch.wrap_tensor(self._rigid_body_state_tensor).view(self.num_envs, -1, 13).view(self.num_envs, -1, 13)

        self._dof_pos = self._dof_state[..., 0]
        self._dof_vel = self._dof_state[..., 1]

        # self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["left_gripper_link"], :2]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["right_gripper_link"], :2]


        # Initialize actions
        self._effort_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # Initialize control
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        
    
    

    # def _refresh(self):
    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     self.gym.refresh_dof_state_tensor(self.sim)
    #     # self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     # Refresh states
    #     self._update_states()


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_robomaster_reward(
            self.reset_buf, self.progress_buf, self._root_state,(self._eef_lf_state+self._eef_rf_state)/2, self.reward_settings, self.max_episode_length
        )

    # def _update_states(self):
    #     self.states.update({
    #         # robomaster base position: pos=[:3], angle=[3:7]
    #         "robomaster_pos": torch.cat((self._root_state[:, 0,:2],self._root_state[:, 0, 5].view(-1, 1)),1),
    #         "gripper_dof_pos": self._dof_pos[:, :2],
    #         # balls
    #         "ballA_pos": self._ballA_state[:, :2],
    #         "ballB_pos": self._ballB_state[:, :2],
    #         "ballA_to_ballB_pos": self._ballB_state[:, :2] - self._ballA_state[:, :2],
    #     })

    def compute_observations(self, env_ids=None):
        robomaster_pos = torch.cat((self._root_state[:, 0,:2],self._root_state[:, 0, 5].view(-1, 1)),1)
        robomaster_vel = torch.cat((self._root_state[:, 0,7:9],self._root_state[:, 0, -1].view(-1, 1)),1)
        gripper_dof_pos = self._dof_pos[:, :2]
        gripper_pos = torch.cat((self._eef_lf_state, self._eef_rf_state), 1)
        ballA_pos = self._ballA_state[:, :2]
        
        ballA_vel = self._ballA_state[:, 7:9]
        ballA_to_ballB_pos = self._ballB_state[:, :2] - self._ballA_state[:, :2]
        self.obs_buf = torch.cat([robomaster_pos,robomaster_vel, gripper_dof_pos,gripper_pos, ballA_pos, ballA_vel, ballA_to_ballB_pos], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # Reset balls, sampling ball B first, then A
        self._reset_init_ball_state(ball='B', env_ids=env_ids)
        # self._reset_init_ball_state(ball='A', env_ids=env_ids)
        self._root_state[env_ids, 0] = torch.tensor([-0.45,0,0,0,0,0,1,0,0,0,0,0,0], device=self.device)
        self._root_state[env_ids, self._ballA_id, :] = self._ballA_state[env_ids].clone()
        self._root_state[env_ids, self._ballB_id, :] = self._ballB_state[env_ids].clone()

        pos = self.robomaster_default_dof_pos.unsqueeze(0)


        # Reset the internal obs accordingly
        self._dof_pos[env_ids, :] = pos.clone()
        self._dof_vel[env_ids, :] = torch.zeros_like(self._dof_vel[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
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
    
    def _reset_init_ball_state(self, ball, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        # sampled_ball_state = torch.zeros(num_resets, 13, device=self.device)

        # # Get correct references depending on which one was selected
        # if ball.lower() == 'a':
        #     this_ball_state_all = self._ballA_state
        #     ball_heights = self.ballA_size
        # elif ball.lower() == 'b':
        #     this_ball_state_all = self._ballB_state
        #     ball_heights = self.ballB_size
        # else:
        #     raise ValueError(f"Invalid ball specified, options are 'A' and 'B'; got: {ball}")
        

        # # Set z value, which is fixed height
        # sampled_ball_state[:, 2] = ball_heights + 0.001
        # # Initialize rotation, which is no rotation (quat w = 1)
        # sampled_ball_state[:, 6] = 1.0

        # # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # # We use a simple heuristic of checking based on balls' radius to determine if a collision would occur
        
        # sampled_ball_state[:, :2] = torch.tensor([0, 0], device=self.device, dtype=torch.float32)+(torch.rand(sampled_ball_state[:, :2].shape, device=self.device, dtype=torch.float32)-0.5)
        # # sampled_ball_state[:, :2] = torch.tensor([-0.45, 0], device=self.device, dtype=torch.float32)
        # this_ball_state_all[env_ids, :] = sampled_ball_state

        ball_heights = self.ballA_size
        ballA_state = torch.zeros(num_resets, 13, device=self.device)
        ballA_state[:, 2] = ball_heights + 0.001
        ballA_state[:, 6] = 1.0
        ballA_state[:, :2] = torch.tensor([0, -0.2], device=self.device, dtype=torch.float32)+0*(torch.rand(ballA_state[:, :2].shape, device=self.device, dtype=torch.float32)-0.5)
        self._ballA_state[env_ids, :] = ballA_state

        ballB_state = torch.zeros(num_resets, 13, device=self.device)
        ballB_state[:, 2] = ball_heights + 0.001
        ballB_state[:, 6] = 1.0
        ballB_state[:, :2] = torch.tensor([0, 0.2], device=self.device, dtype=torch.float32)+0*(torch.rand(ballB_state[:, :2].shape, device=self.device, dtype=torch.float32)-0.5)
        self._ballB_state[env_ids, :] = ballB_state
        



    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # Split wheel and gripper command
        u_wheel, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        # Control arm (scale value first)
        # u_wheel = self._compute_osc_torques(dpose=u_wheel)
        self._effort_control[:, 2:] = u_wheel.clone()

        u_fingers = torch.ones_like(self._effort_control[:, :2])
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, -1, 1)
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, 1, -1)
        # # Write gripper command to appropriate tensor buffer
        self._effort_control[:, :2] = u_fingers.clone()
        # Deploy actions
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))


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
    reset_buf, progress_buf, _root_state,eef_pos, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor,Tensor, Dict[str, float], float) -> Tuple[Tensor, Tensor]
    robomaster_pos = _root_state[:, 0, :3].clone()
    robomaster_vel = _root_state[:, 0,7:9].clone()
    ballA_pos = _root_state[:, 1, :2]
    ballA_vel = _root_state[:, 1, 7:9]
    ballB_pos =  _root_state[:, 2, :2]
    norm_1 = torch.norm(ballA_pos - eef_pos, dim=1)
    # print(eef_pos[0],'\n', robomaster_pos[0],'\n')
    dist_reward1 = torch.tanh(torch.sum(robomaster_vel * (ballA_pos - eef_pos), dim=1)/norm_1)
    ballA_to_ballB_pos = ballB_pos - ballA_pos
    norm_2 = torch.norm(ballA_to_ballB_pos, dim=1)
    dist_reward2 = torch.tanh(torch.sum(ballA_vel*ballA_to_ballB_pos, dim=1)/norm_2)
    

    stack_reward = (norm_2 < 0.06)

    penalty1 = (torch.max(torch.abs(robomaster_pos), dim=1)[0]>1)
    penalty2 = (torch.max(torch.abs(ballA_pos), dim=1)[0]>1)
    

    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    rewards = torch.where(stack_reward- penalty1 - penalty2,
                          reward_settings["r_stack_scale"] * (stack_reward- penalty1 - penalty2),
                          reward_settings["r_dist_scale"] * dist_reward1 + reward_settings["r_align_scale"] * dist_reward2 )
    # rewards = reward_settings["r_stack_scale"] * (stack_reward- penalty1 - penalty2) + reward_settings["r_dist_scale"] * dist_reward1 + reward_settings["r_align_scale"] * dist_reward2 

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0) | (penalty1 > 0)| (penalty2 > 0), torch.ones_like(reset_buf), reset_buf)
    
    return rewards, reset_buf