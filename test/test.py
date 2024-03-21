import isaacgym

def main():
    # 初始化Isaac Gym
    gym = isaacgym.gymapi.acquire_gym()

    # 创建仿真设置
    sim_params = isaacgym.gymapi.SimParams()
    sim_params.up_axis = isaacgym.gymapi.UP_AXIS_Z
    sim_params.gravity = isaacgym.gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0  # 模拟的时间步长

    # 创建仿真环境
    sim = gym.create_sim(0, 0, isaacgym.gymapi.SIM_PHYSX, sim_params)

    if sim is None:
        raise ValueError("无法创建仿真环境")

    # 创建查看器以进行渲染
    viewer = gym.create_viewer(sim, isaacgym.gymapi.CameraProperties())
    if viewer is None:
        raise ValueError("无法创建查看器")

    gym.viewer_camera_look_at(viewer, None, isaacgym.gymapi.Vec3(2, 2, 2), isaacgym.gymapi.Vec3(0, 0, 0))



    # 设置URDF文件的路径
    # asset_root = "/home/xander/Codes/isaacgym_robomaster/assets/urdf/robomasterEP_description/robot/"
    # asset_file = "robomaster.urdf"
    asset_root = "/home/xander/Codes/isaacgym_robomaster/assets/urdf/ycb/011_banana/"
    asset_file = "011_banana.urdf"

    # 加载URDF资产
    asset_options = isaacgym.gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # 创建环境实例
    plane_params = isaacgym.gymapi.PlaneParams()
    plane_params.normal = isaacgym.gymapi.Vec3(0, 0, 1) 
    plane_params.distance = 0
    gym.add_ground(sim, plane_params)

    env = gym.create_env(sim, isaacgym.gymapi.Vec3(-5, -5, 0), isaacgym.gymapi.Vec3(5, 5, 0), 1)

    # 创建机器人actor
    pose = isaacgym.gymapi.Transform()
    pose.p = isaacgym.gymapi.Vec3(0, 0, 0)
    pose.r = isaacgym.gymapi.Quat.from_euler_zyx(0, 0, 0)
    robot_actor = gym.create_actor(env, asset, pose, "robot", 0, 1, 0)

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # 同步渲染
        gym.sync_frame_time(sim)

    # 清理资源
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
