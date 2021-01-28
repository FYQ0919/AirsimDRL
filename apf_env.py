import time
import numpy as np
import airsim
import config
import math

clockspeed = 1
timeslice = 0.5 / clockspeed
goalX = 15
outX = -0.5
outY = 5
floorZ = 5
goals = [3, 6, 9, 12, goalX]
object_pos = [14, 0, 1]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']


class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0

    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation

    def step(self, quad_offset):
        # move with given velocity
        quad_offset = [float(i) for i in quad_offset]
        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.simPause(False)

        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], timeslice)
        # self.client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], timeslice)
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided
            # landed = quad_pos.y_val > 10 and self.client.getMultirotorState().landed_state == airsim.LandedState.Landed
            # landed = landed or (quad_pos.y_val > 10 and quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        self.client.simPause(True)
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        pos = np.array([quad_pos.x_val,quad_pos.y_val,quad_pos.z_val])
        # observe with depth camera
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        force = -self.add_rep_filed(pos,[5,0,-5])
        force += -self.add_rep_field(pos,[2,-3,-5])
        foce  += -self.add_rep_field(pos,[2,3,-5])
        foce  += -self.add_rep_field(pos,[7,5,-5])
        foce  += -self.add_rep_field(pos,[10,-5,-5])
        force += self.add_att_filed(pos,object_pos)
        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        # decide whether done
        dead = has_collided or quad_pos.x_val <= outX
        done = dead or quad_pos.x_val >= goalX

        # compute reward
        reward = self.compute_reward(quad_pos, quad_vel, dead)

        # log info
        info = {}
        info['X'] = quad_pos.x_val
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.x_val <= outX:
            info['status'] = 'out'
        elif quad_pos.x_val >= goalX:
            info['status'] = 'goal'
        else:
            info['status'] = 'going'
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation, reward, done, info,force

    def compute_reward(self, quad_pos, quad_vel, dead):
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        pos = np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val], dtype=np.float)
        bias = pos - object_pos
        success = np.linalg.norm(bias) < 2
        speed = np.linalg.norm(vel)
        if dead:
            reward = config.reward['dead']
        elif quad_pos.x_val >= goals[self.level]:
            self.level += 1
            # reward = config.reward['forward'] * (1 + self.level / len(goals))
            reward = config.reward['goal'] * (1 + self.level / len(goals))
        elif speed < speed_limit:
            reward = config.reward['slow']
        else:
            reward = float(vel[1]) * 0.1
        if success:
            reward += 5
        # elif vel[1] > 0:
        #     reward = config.reward['forward'] * (1 + self.level / len(goals))
        # else:
        #     reward = config.reward['normal']
        return reward

    def add_rep_field(self, pos1, pos2):
        dis = pos1 - pos2
        force = [0, 0, 0]
        q = 3
        D = np.linalg.norm(dis)
        theta = math.atan(dis[1] / dis[0])
        alpha = math.acos(math.sqrt(dis[0] ** 2 + dis[1] ** 2))
        if D < q:
            all_force = 0.5 * 1 * ((1 / dis) - (1 / q)) ** 2
            force[0] = all_force * math.cos(alpha) * math.cos(theta)
            force[1] = all_force * math.cos(alpha) * math.sin(alpha)
            force[3] = all_force * math.sin(alpha)
            return force
        if D > q:
            return force

    def add_att_filed(self, pos1, pos2):
        dis = pos1 - pos2
        force = [0, 0, 0]
        d = 5
        D = np.linalg.norm(dis)
        theta = math.atan(dis[1] / dis[0])
        alpha = math.acos(math.sqrt(dis[0] ** 2 + dis[1] ** 2))
        if D < d:
            all_force = 0.5 * 0.1 * D ** 2
            force[0] = all_force * math.cos(alpha) * math.cos(theta)
            force[1] = all_force * math.cos(alpha) * math.sin(alpha)
            force[3] = all_force * math.sin(alpha)
            return force
        if D > d:
            all_force = d * 0.1 * D - 0.5 * 0.1 * d ** 2
            force[0] = all_force * math.cos(alpha) * math.cos(theta)
            force[1] = all_force * math.cos(alpha) * math.sin(alpha)
            force[3] = all_force * math.sin(alpha)
            return  force

    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')