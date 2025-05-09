#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu, LaserScan
from std_msgs.msg import Float64
from geographic_msgs.msg import GeoPath
import math
import numpy as np
from common_utilities import gps_to_enu, quaternion_to_euler, euler_to_quaternion
import cv2
import heapq  # 用于优先队列实现 A* 算法
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import signal
import sys

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, 1)
    
    def forward(self, x, h_state=None):   #前向传播
        # 确保输入是3D张量 [batch_size, seq_len, input_size]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch维度
            
        # 如果没有提供隐藏状态或维度不匹配，初始化一个新的
        if h_state is None or h_state.size(1) != x.size(0):
            h_state = torch.zeros(1, x.size(0), 20, device=x.device)
            
        r_out, h_state = self.rnn(x, h_state)
        out = self.fc(r_out[:, -1, :])
        return out, h_state


class GetPosition(Node):
    def __init__(self, name):
        super().__init__(name)
        self.cur_pos = None  # 当前 2D 位置 (x, y)
        self.cur_rot = None  # 偏航角
        self.cur_position = None  # 当前 3D 位置 (x, y, z)
        self.cur_rotation = None  # 当前 3D 方向 (roll, pitch, yaw)
        self.obstacles = []  # 当前帧的障碍物
        self.global_map = set()  # 全局地图，使用集合存储障碍物以避免重复
        self.path = []  # 存储路径点
        self.angle = 0.0
        self.thrust = 80.0  # 初始推力设置为80
        
        # 模型保存参数
        self.model_save_dir = os.path.expanduser("../../rnn_model_save")
        self.model_save_path = os.path.join(self.model_save_dir, "boat_rnn_model.pt")
        self.optimizer_save_path = os.path.join(self.model_save_dir, "boat_rnn_optimizer.pt")
        self.last_save_time = time.time()
        self.save_interval = 60.0  # 每60秒保存一次模型
        
        # 创建保存目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 初始化RNN模型
        self.rnn_model = RNN()
        self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.05)
        self.criterion = nn.MSELoss()  # 均方误差损失函数
        self.hidden_state = None
        
        # 加载已保存的模型(如果存在)
        self.load_model()
        
        # 设置信号处理，确保在程序退出时保存模型
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 订阅 IMU、GPS 和 2D 激光雷达数据
        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self.imu_callback, 10)
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix', self.gps_callback, 10)
        self.create_subscription(LaserScan, '/wamv/sensors/lidars/lidar_wamv_sensor/scan', self.lidar_callback, 10)
        
        self.goal_pos = gps_to_enu(-33.7217013209, 150.67684609858)  # 转换为 ENU 坐标
        self.goal_pos = np.array(self.goal_pos[:2])  # 只取 x, y
        
        # 发布控制指令
        self.thrust_pub_l = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.thrust_pub_r = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.angle_pub_l = self.create_publisher(Float64, '/wamv/thrusters/left/pos', 10)
        self.angle_pub_r = self.create_publisher(Float64, '/wamv/thrusters/right/pos', 10)
        
        # 目标位置：-33.7220013209 150.67684609858
        
        self.timer = self.create_timer(0.1, self.controller)
        
        # 局部路径规划参数
        self.has_local_obstacle = False  # 是否有近距离障碍物
        self.local_path = []  # 局部路径
        self.obstacle_radius = 20.0  # 障碍物检测范围（米）
        self.repulsive_scale = 100.0  # 斥力场比例因子
        self.attractive_scale = 1.0  # 引力场比例因子
        self.goal_threshold = 2.0  # 目标点阈值（米）
        self.step_size = 2.0  # 路径点之间的步长（米）
        self.max_iterations = 100  # 最大迭代次数
        self.local_path_points = 10  # 局部路径生成的点数
        self.global_path_calculated = False  # 全局路径是否已计算
        self.global_path = []  # 保存全局路径副本
    
    def load_model(self):
        """加载已保存的模型"""
        try:
            if os.path.exists(self.model_save_path):
                self.get_logger().info(f"加载模型: {self.model_save_path}")
                self.rnn_model.load_state_dict(torch.load(self.model_save_path))
                
                if os.path.exists(self.optimizer_save_path):
                    self.optimizer.load_state_dict(torch.load(self.optimizer_save_path))
                    
                self.get_logger().info("模型加载成功")
            else:
                self.get_logger().info("没有找到已保存的模型，将使用新模型")
        except Exception as e:
            self.get_logger().error(f"加载模型失败: {str(e)}")
    
    def save_model(self):
        """保存模型到文件"""
        try:
            torch.save(self.rnn_model.state_dict(), self.model_save_path)
            torch.save(self.optimizer.state_dict(), self.optimizer_save_path)
            self.get_logger().info(f"模型已保存到 {self.model_save_path}")
            self.last_save_time = time.time()
        except Exception as e:
            self.get_logger().error(f"保存模型失败: {str(e)}")
    
    def signal_handler(self, sig, frame):
        """处理进程中断信号，确保在退出前保存模型"""
        self.get_logger().info("接收到中断信号，保存模型并退出...")
        self.save_model()
        sys.exit(0)


    def controller(self):
        if self.cur_pos is None or self.cur_rot is None:
            return

        # 首次运行时计算全局路径
        if not self.global_path_calculated and len(self.global_map) > 0:
            self.path = self.a_star(self.cur_pos, self.goal_pos)
            if len(self.path) > 0:
                self.global_path = self.path.copy()  # 保存全局路径副本
                self.global_path_calculated = True
                self.get_logger().info(f"全局路径规划完成，路径点数: {len(self.global_path)}")
            else:
                self.get_logger().warn("全局路径规划失败，无法找到有效路径")
                return

        # 检查是否需要进行局部路径规划
        self.check_for_local_obstacles()

        # 路径点1米内移除
        self.remove_close_points()

        # 根据情况选择使用全局路径或局部路径
        if self.has_local_obstacle:
            # 使用人工势场法进行局部路径规划
            if len(self.global_path) > 0:
                # 找到全局路径上合适的点作为局部规划目标
                local_goal = self.find_local_goal()
                self.local_path = self.artificial_potential_field(self.cur_pos, local_goal)
                if len(self.local_path) > 2:
                    self.get_logger().info(f"使用局部路径规划，路径点数: {len(self.local_path)}")
                    target_point = self.local_path[min(2, len(self.local_path)-1)]  # 获取局部路径的点作为目标
                else:
                    self.get_logger().warn("局部路径规划失败，使用全局路径")
                    target_point = self.global_path[min(2, len(self.global_path)-1)]
            else:
                self.get_logger().warn("无有效全局路径用于局部规划参考")
                self.thrust = 0.0
                return
        else:
            # 使用全局路径
            if len(self.path) > 2:
                target_point = self.path[2]  # 获取全局路径的第3个点作为目标
            else:
                # 路径点不足，可能已接近终点
                if len(self.path) > 0:
                    target_point = self.path[-1]  # 使用最后一个点
                else:
                    self.get_logger().warn("无有效路径点")
                    self.thrust = 0.0
                    return

        # 计算当前位置与目标点之间的方向
        dx = target_point[0] - self.cur_pos[0]
        dy = target_point[1] - self.cur_pos[1]
        angle_to_target = math.atan2(dy, dx)
        
        # 计算夹角a：当前位置与路径点所成的夹角
        angle_diff = self.cur_rot - angle_to_target
        
        # 记录船前进方向与目标方向的角度差
        self.get_logger().info(f"目标角度: {angle_to_target:.4f}, 当前偏航角: {self.cur_rot:.4f}, 角度差: {angle_diff:.4f}")
        
        # 训练RNN模型
        input_data = torch.tensor([[[self.cur_rot, angle_to_target]]], dtype=torch.float32)
        self.train_rnn(input_data, angle_to_target)
        
        # 发布控制指令
        thrust_msg_l = Float64()
        thrust_msg_r = Float64()
        angle_msg_l = Float64()
        angle_msg_r = Float64()
        
        # 确保推力为正，船才会前进
        thrust_value = abs(self.thrust)
        thrust_msg_l.data = float(thrust_value)
        thrust_msg_r.data = float(thrust_value)
        angle_msg_l.data = float(self.angle)
        angle_msg_r.data = float(self.angle)
        
        self.thrust_pub_l.publish(thrust_msg_l)
        self.thrust_pub_r.publish(thrust_msg_r)
        self.angle_pub_l.publish(angle_msg_l)
        self.angle_pub_r.publish(angle_msg_r)

        
        # 检查是否需要保存模型
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_model()
        
        
    def train_rnn(self, input_data, angle_to_target):
        """训练RNN模型"""
        # 执行前向计算
        propeller_angle, hidden_state = self.rnn_model(input_data, self.hidden_state)
        # 应用相同的限制
        propeller_angle_limited = torch.tanh(propeller_angle) * (math.pi*5/6)
        self.angle = propeller_angle_limited
        target = torch.tensor([[self.cur_rot - angle_to_target]], dtype=torch.float32,requires_grad=True)
        loss = self.criterion(self.angle, target)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 打印训练信息
        self.get_logger().info(f"训练损失: {loss.item():.4f}, 螺旋桨角度: {propeller_angle_limited.item():.4f}")

    def gps_callback(self, msg):
        enu = gps_to_enu(msg.latitude, msg.longitude)
        enu = np.array(enu[:3])
        # 0.85米修正
        self.gps_offset = 0.85
        enu[0] += self.gps_offset * math.cos(self.cur_rot)
        enu[1] += self.gps_offset * math.sin(self.cur_rot)
        self.cur_pos = np.array([enu[0], enu[1]])
        self.cur_position = enu[:3]
        
    def imu_callback(self, msg):
        self.cur_rotation = quaternion_to_euler(msg.orientation)
        self.cur_rot = self.cur_rotation[2]  # 偏航角

    def lidar_callback(self, msg):
        if self.cur_pos is None or self.cur_rot is None:
            return

        # 处理 LaserScan 数据
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges

        # 当前帧的障碍物点
        current_scan = set()

        for i, r in enumerate(ranges):
            if 1 < r < msg.range_max:  # 过滤掉无效的距离值
                angle = angle_min + i * angle_increment
                # 将激光雷达的局部坐标转换为全局坐标
                x_local = r * math.cos(angle)
                y_local = r * math.sin(angle)
                x_global = x_local * math.cos(self.cur_rot) - y_local * math.sin(self.cur_rot) + self.cur_pos[0]
                y_global = x_local * math.sin(self.cur_rot) + y_local * math.cos(self.cur_rot) + self.cur_pos[1]
                current_scan.add((int(x_global), int(y_global)))

        # 更新全局地图
        # 添加新点
        new_points = current_scan - self.global_map
        self.global_map.update(new_points)

        # 移除当前扫描中没有的点（但保留被遮挡的点）
        points_to_remove = self.global_map - current_scan
        for point in points_to_remove:
            # 检查是否可能是被遮挡的点
            if not self.is_point_occluded(point, current_scan):
                self.global_map.remove(point)

        # 计算路径
        self.path = self.a_star(self.cur_pos, self.goal_pos)

        # 绘制地图
        self.draw_map()

    def is_point_occluded(self, point, current_scan):
        """判断一个点是否可能被遮挡"""
        for neighbor in self.get_neighbors(point):
            if neighbor in current_scan:
                return True
        return False

    def a_star(self, start, goal):
        """A* 算法实现"""
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.global_map:  # 忽略障碍物
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 如果没有找到路径，返回空列表

    def heuristic(self, a, b):
        """启发式函数：使用欧几里得距离"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def distance(self, a, b):
        """计算两点之间的距离"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, node):
        """获取当前节点的邻居节点，8个节点"""
        x, y = node
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1)
        ]

    def reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def artificial_potential_field(self, start, goal):
        """使用人工势场法实现局部路径规划"""
        if not self.global_map:
            return [tuple(start)]
        
        path = [tuple(map(int, start))]
        current = np.array(start)
        
        for _ in range(self.max_iterations):
            # 如果接近目标点，结束规划
            if np.linalg.norm(current - goal) < self.goal_threshold:
                path.append(tuple(map(int, goal)))
                break
            
            # 计算引力（目标点对当前位置的吸引力）
            attractive_force = self.attractive_scale * (goal - current) / np.linalg.norm(goal - current)
            
            # 计算斥力（障碍物对当前位置的排斥力）
            repulsive_force = np.zeros(2)
            
            for obstacle in self.global_map:
                obstacle_pos = np.array(obstacle)
                dist = np.linalg.norm(current - obstacle_pos)
                
                # 只考虑一定范围内的障碍物
                if dist < self.obstacle_radius:
                    # 斥力随距离增大而迅速减小
                    if dist < 0.1:  # 防止除以零
                        dist = 0.1
                    force_magnitude = self.repulsive_scale * (1.0 / dist - 1.0 / self.obstacle_radius) * (1.0 / (dist * dist))
                    force_direction = (current - obstacle_pos) / dist
                    repulsive_force += force_magnitude * force_direction
            
            # 合力
            total_force = attractive_force + repulsive_force
            if np.linalg.norm(total_force) > 0:
                total_force = total_force / np.linalg.norm(total_force)
            
            # 沿合力方向移动一步
            next_pos = current + self.step_size * total_force
            
            # 检查下一个位置是否在障碍物内
            is_colliding = False
            for obstacle in self.global_map:
                if np.linalg.norm(next_pos - np.array(obstacle)) < 2.0:  # 安全距离
                    is_colliding = True
                    break
            
            if is_colliding:
                # 如果可能碰撞，尝试随机方向
                for _ in range(10):  # 尝试10次
                    random_angle = np.random.uniform(0, 2*np.pi)
                    random_dir = np.array([np.cos(random_angle), np.sin(random_angle)])
                    test_pos = current + self.step_size * random_dir
                    
                    is_test_colliding = False
                    for obstacle in self.global_map:
                        if np.linalg.norm(test_pos - np.array(obstacle)) < 2.0:
                            is_test_colliding = True
                            break
                    
                    if not is_test_colliding:
                        next_pos = test_pos
                        break
                
                # 如果所有尝试都失败，保持原位
                if is_colliding:
                    break
            
            current = next_pos
            path.append(tuple(map(int, current)))
            
            # 限制路径点数量
            if len(path) >= self.local_path_points:
                break
        
        return path

    def find_local_goal(self):
        """找到全局路径上离当前位置15-20米左右的点作为局部规划目标"""
        if not self.global_path:
            return self.goal_pos
        
        target_distance = 15.0
        best_point = self.global_path[-1]  # 默认使用终点
        best_distance_diff = float('inf')
        
        for point in self.global_path:
            dist = np.linalg.norm(np.array(point) - self.cur_pos)
            distance_diff = abs(dist - target_distance)
            
            if distance_diff < best_distance_diff:
                best_distance_diff = distance_diff
                best_point = point
        
        return np.array(best_point)
    
    def check_for_local_obstacles(self):
        """检查船舶附近是否有障碍物"""
        if self.cur_pos is None:
            return
        
        self.has_local_obstacle = False
        
        for obstacle in self.global_map:
            obstacle_pos = np.array(obstacle)
            dist = np.linalg.norm(self.cur_pos - obstacle_pos)
            
            if dist < self.obstacle_radius:
                self.has_local_obstacle = True
                self.get_logger().info(f"检测到{dist:.2f}米内的障碍物，将使用局部路径规划")
                break

    def remove_close_points(self):
        """移除已经接近的全局路径点及其前面的所有路径点"""
        if not self.global_path or len(self.global_path) == 0:
            return
        
        # 查找距离当前位置最近的路径点
        closest_point_idx = -1
        closest_distance = float('inf')
        
        for i, point in enumerate(self.global_path):
            dist = math.sqrt((self.cur_pos[0] - point[0])**2 + (self.cur_pos[1] - point[1])**2)
            if dist < closest_distance:
                closest_distance = dist
                closest_point_idx = i
        
        # 如果找到的最近点距离小于1米，移除该点及其前面的所有点
        if closest_distance < 1.0 and closest_point_idx >= 0:
            # 记录被移除的点数量
            removed_count = closest_point_idx + 1
            removed_points = self.global_path[:removed_count]
            
            # 移除全局路径中的点
            self.global_path = self.global_path[removed_count:]
            
            # 同步更新当前使用的路径点列表
            if len(self.path) > 0:
                # 找到path中对应的点并移除
                path_to_remove = []
                for i, path_point in enumerate(self.path):
                    for removed_point in removed_points:
                        if np.array_equal(np.array(path_point), np.array(removed_point)):
                            path_to_remove.append(i)
                            break
                
                # 从后向前删除，避免索引变化
                for idx in sorted(path_to_remove, reverse=True):
                    if idx < len(self.path):
                        self.path.pop(idx)
            
            self.get_logger().info(f"移除已通过的{removed_count}个路径点，最近点距离: {closest_distance:.2f}米, 剩余路径点: {len(self.global_path)}")
            
            # 如果移除了所有路径点，记录日志
            if len(self.global_path) == 0:
                self.get_logger().info("所有路径点已移除，即将到达目标点")

    def draw_map(self):
        map_size = 700
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255
        scale = 1  # 每个单位对应的像素数

        # 绘制船的位置（红色点）
        if self.cur_pos is not None:
            ship_x = int((self.cur_pos[0] + 700) * scale)
            ship_y = int(map_size - (self.cur_pos[1] - 100) * scale)
            cv2.circle(map_img, (ship_x, ship_y), 3, (0, 0, 255), -1)

            # 绘制小船的偏航角方向（紫色箭头）
            arrow_length = 20  # 箭头的长度
            arrow_x = int(ship_x + arrow_length * math.cos(self.cur_rot))
            arrow_y = int(ship_y - arrow_length * math.sin(self.cur_rot))  # 注意 y 轴方向是反的
            cv2.arrowedLine(map_img, (ship_x, ship_y), (arrow_x, arrow_y), (255, 0, 255), 2, tipLength=0.3)

        # 绘制全局地图中的障碍物（蓝色点）
        for obstacle in self.global_map:
            obs_x = int((obstacle[0] + 700) * scale)
            obs_y = int(map_size - (obstacle[1] - 100) * scale)
            if 0 <= obs_x < map_size and 0 <= obs_y < map_size:
                cv2.circle(map_img, (obs_x, obs_y), 1, (255, 0, 0), -1)

        # 绘制路径（绿色线）
        for i in range(1, len(self.path)):
            start = self.path[i - 1]
            end = self.path[i]
            start_x = int((start[0] + 700) * scale)
            start_y = int(map_size - (start[1] - 100) * scale)
            end_x = int((end[0] + 700) * scale)
            end_y = int(map_size - (end[1] - 100) * scale)
            cv2.line(map_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # 如果有局部路径，用黄色线绘制
        if self.has_local_obstacle and len(self.local_path) > 1:
            for i in range(1, len(self.local_path)):
                start = self.local_path[i - 1]
                end = self.local_path[i]
                start_x = int((start[0] + 700) * scale)
                start_y = int(map_size - (start[1] - 100) * scale)
                end_x = int((end[0] + 700) * scale)
                end_y = int(map_size - (end[1] - 100) * scale)
                cv2.line(map_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

        # 绘制终点（黄色点）
        goal_x = int((self.goal_pos[0] + 700) * scale)
        goal_y = int(map_size - (self.goal_pos[1] - 100) * scale)
        cv2.circle(map_img, (goal_x, goal_y), 5, (0, 255, 255), -1)
        
        # 显示下次模型保存时间
        seconds_to_save = self.save_interval - (time.time() - self.last_save_time)
        cv2.putText(map_img, f"Model save in: {seconds_to_save:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 显示图像
        cv2.imshow("Real-Time Map", map_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    rclpy.init()
    node = GetPosition('get_position_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 确保在程序退出前保存模型
        node.save_model()
    finally:
        node.destroy_node()
        rclpy.shutdown()