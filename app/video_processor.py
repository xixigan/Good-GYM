"""
视频处理模块 - 负责图像更新和姿态检测
"""

import cv2
import os
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
from core.translations import Translations as T

class VideoProcessor:
    """视频处理器类"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def update_image(self, frame, fps=0):
        """更新图像显示并处理姿态检测"""
        try:
            # 更新FPS值
            self.main_window.current_fps = fps
            
            # 使用推理帧进行姿态检测（如果可用）
            inference_frame = getattr(self.main_window, 'current_inference_frame', frame)
            
            # 姿态处理器处理推理帧，只获取关键点信息
            _, current_angle, keypoints = self.main_window.pose_processor.process_frame(
                inference_frame, self.main_window.exercise_type
            )
            
            # 准备高分辨率显示帧
            display_frame = frame.copy()
            
            # 如果有关键点信息，在高分辨率帧上绘制骨架
            if keypoints is not None and self.main_window.pose_processor.show_skeleton:
                # 计算缩放比例（推理帧到显示帧）
                scale_x = display_frame.shape[1] / inference_frame.shape[1]
                scale_y = display_frame.shape[0] / inference_frame.shape[0]
                
                # 将关键点坐标缩放到显示帧尺寸
                scaled_keypoints = keypoints.copy()
                scaled_keypoints[:, 0] *= scale_x
                scaled_keypoints[:, 1] *= scale_y
                
                # 在高分辨率帧上绘制骨架
                display_frame = self.draw_skeleton_on_frame(display_frame, scaled_keypoints)
            
            # 如果启用镜像模式，应用镜像处理
            if self.main_window.mirror_mode:
                display_frame = cv2.flip(display_frame, 1)
            
            # 转换BGR到RGB（Qt需要RGB格式）
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # 更新视频显示
            self.main_window.video_display.update_image(display_frame)
            
            # 更新UI组件
            self.update_ui_components(current_angle, keypoints)
            
        except Exception as e:
            print(f"Error updating image: {e}")
    
    def draw_skeleton_on_frame(self, frame, keypoints):
        """在高分辨率帧上绘制骨架"""
        try:
            # 定义连接关系 (COCO 17 keypoint format)
            connections = [
                # Head and face
                [0, 1], [0, 2], [1, 3], [2, 4],  # nose-eyes-ears
                # Torso
                [5, 6],   # left_shoulder-right_shoulder
                [5, 11],  # left_shoulder-left_hip
                [6, 12],  # right_shoulder-right_hip
                [11, 12], # left_hip-right_hip
                # Arms
                [5, 7], [7, 9],    # left_shoulder-left_elbow-left_wrist
                [6, 8], [8, 10],   # right_shoulder-right_elbow-right_wrist
                # Legs
                [11, 13], [13, 15], # left_hip-left_knee-left_ankle
                [12, 14], [14, 16]  # right_hip-right_knee-right_ankle
            ]
            
            # 定义颜色 (BGR format)
            colors = {
                'head': (51, 153, 255),    # Blue
                'torso': (255, 153, 51),   # Orange
                'arms': (153, 255, 51),    # Green
                'legs': (255, 51, 153)     # Pink
            }
            
            # 绘制连接线
            for connection in connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    
                    # 跳过无效点
                    if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                        continue
                    
                    # 选择颜色
                    if pt1_idx in [0, 1, 2, 3, 4]:  # Head
                        color = colors['head']
                    elif pt1_idx in [5, 6, 11, 12]:  # Torso
                        color = colors['torso']
                    elif pt1_idx in [7, 8, 9, 10]:  # Arms
                        color = colors['arms']
                    else:  # Legs
                        color = colors['legs']
                    
                    # 绘制连接线 - 根据分辨率调整线条粗细
                    line_thickness = max(2, int(frame.shape[1] / 640 * 3))
                    cv2.line(frame, 
                            (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), 
                            color, line_thickness)
            
            # 绘制关键点
            for i, point in enumerate(keypoints):
                if point[0] == 0 and point[1] == 0:  # 跳过无效点
                    continue
                
                # 选择颜色
                if i in [0, 1, 2, 3, 4]:  # Head
                    color = colors['head']
                elif i in [5, 6, 11, 12]:  # Torso
                    color = colors['torso']
                elif i in [7, 8, 9, 10]:  # Arms
                    color = colors['arms']
                else:  # Legs
                    color = colors['legs']
                
                # 根据分辨率调整关键点大小
                point_radius = max(3, int(frame.shape[1] / 640 * 5))
                cv2.circle(frame, (int(point[0]), int(point[1])), point_radius, color, -1)
                cv2.circle(frame, (int(point[0]), int(point[1])), point_radius + 2, color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing skeleton: {e}")
            return frame
    
    def update_ui_components(self, current_angle, keypoints):
        """更新UI组件显示"""
        try:
            # 更新角度显示 - 注释掉这部分代码
            # if current_angle is not None:
            #     self.main_window.control_panel.update_angle(str(int(current_angle)), self.main_window.exercise_type)
            
            # 更新阶段显示（上/下）
            if hasattr(self.main_window.exercise_counter, 'stage'):
                self.main_window.control_panel.update_phase(self.main_window.exercise_counter.stage)
            
            # 获取当前计数 - 直接使用计数器属性
            current_count = self.main_window.exercise_counter.counter
            
            # 如果计数增加且不是重置操作，播放声音（但不自动记录）
            if current_count > self.main_window.current_count and not self.main_window.is_resetting:
                # 为每次计数增加播放计数声音
                self.main_window.sound_manager.play_count_sound()
                
                # 每10次计数播放成功声音
                if current_count % 10 == 0:
                    self.main_window.sound_manager.play_milestone_sound(current_count)
                    self.main_window.statusBar.showMessage(f"Congratulations on completing {current_count} {self.main_window.control_panel.exercise_display_map[self.main_window.exercise_type]}!")
                
                # 更新缓存的当前计数
                self.main_window.current_count = current_count
            
            # 更新计数器显示
            self.main_window.control_panel.update_counter(str(current_count))
        except Exception as e:
            print(f"Error updating image: {str(e)}")
    
    def change_camera(self, index):
        """切换摄像头"""
        self.main_window.video_thread.set_camera(index)
        self.main_window.statusBar.showMessage(f"Switched to camera {index}")
    
    def toggle_rotation(self, rotate):
        """切换视频旋转模式"""
        # 更新视频线程旋转设置
        self.main_window.video_thread.set_rotation(rotate)
        
        # 更新视频显示方向设置
        # rotate=True表示竖屏，False表示横屏
        self.main_window.video_display.set_orientation(portrait_mode=rotate)
        
        if rotate:
            self.main_window.toggle_rotation_action.setText("Turn off rotation mode")
            self.main_window.statusBar.showMessage("Switched to portrait mode (9:16)")
        else:
            self.main_window.toggle_rotation_action.setText("Turn on rotation mode")
            self.main_window.statusBar.showMessage("Switched to landscape mode (16:9)")
    
    def toggle_skeleton(self, show):
        """切换骨架显示"""
        self.main_window.pose_processor.set_skeleton_visibility(show)
        if show:
            self.main_window.statusBar.showMessage("Show skeleton lines")
        else:
            self.main_window.statusBar.showMessage("Hide skeleton lines")
    
    def toggle_mirror(self, mirror):
        """切换镜像模式"""
        self.main_window.mirror_mode = mirror
        self.main_window.statusBar.showMessage(f"Mirror mode: {'ON' if mirror else 'OFF'}")
        
        # 更新菜单动作状态
        if hasattr(self.main_window, 'toggle_mirror_action'):
            self.main_window.toggle_mirror_action.setChecked(mirror)
    
    def open_video_file(self):
        """打开视频文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.main_window,
            T.get("open_video"),
            "",
            T.get("video_files"),
            options=options
        )
        
        if file_name:
            try:
                # 清除当前计数状态
                self.main_window.reset_exercise_state()
                
                # 切换到运动模式（如果当前不是）
                if hasattr(self.main_window, 'stacked_layout') and hasattr(self.main_window, 'exercise_container'):
                    if not self.main_window.stacked_layout.currentWidget() == self.main_window.exercise_container:
                        self.main_window.switch_to_workout_mode()
                
                # 设置状态栏信息
                video_name = os.path.basename(file_name)
                self.main_window.statusBar.showMessage(f"Current video: {video_name}")
                
                # 将文件路径传递给视频线程，设置为非循环播放模式
                self.main_window.video_thread.set_video_file(file_name, loop=False)
            except Exception as e:
                print(f"Error opening video file: {e}")
                self.main_window.statusBar.showMessage(f"Failed to open video file: {str(e)}")
    
    def switch_to_camera_mode(self):
        """切换回摄像头模式"""
        try:
            # 清除当前计数状态
            self.main_window.reset_exercise_state()
            
            # 切换到运动模式（如果当前不是）
            if hasattr(self.main_window, 'stacked_layout') and hasattr(self.main_window, 'exercise_container'):
                if not self.main_window.stacked_layout.currentWidget() == self.main_window.exercise_container:
                    self.main_window.switch_to_workout_mode()
                
            # 设置状态栏信息
            self.main_window.statusBar.showMessage("Current mode: Camera")
            
            # 返回摄像头模式
            self.main_window.video_thread.set_camera(0)  # 使用默认摄像头
        except Exception as e:
            print(f"Error switching to camera mode: {e}")
            self.main_window.statusBar.showMessage(f"Failed to switch to camera mode: {str(e)}")
    
    def change_model(self, model_mode):
        """切换RTMPose模型模式"""
        try:
            if model_mode == self.main_window.model_mode:
                # 如果是相同模式，无需重新加载
                return
                
            # 停止视频处理
            self.main_window.video_thread.stop()
            
            # 显示状态信息
            self.main_window.statusBar.showMessage(f"Switching RTMPose mode to: {model_mode}...")
            
            # 更新模型模式
            old_model_mode = self.main_window.model_mode
            self.main_window.model_mode = model_mode
            
            print(f"Switching RTMPose mode: {old_model_mode} -> {model_mode}")
            
            # 更新RTMPose处理器模式
            self.main_window.pose_processor.update_model(model_mode)
            
            # 重新初始化视频线程
            self.main_window.setup_video_thread()
            
            # 重新启动视频处理
            QTimer.singleShot(500, self.main_window.start_video)  # 延迟500ms后开始视频
            
            # 更新状态栏
            self.main_window.statusBar.showMessage(f"Switched to RTMPose {model_mode} mode")
            
        except Exception as e:
            # 如果切换失败，显示错误消息
            error_msg = f"RTMPose mode switching failed: {str(e)}"
            self.main_window.statusBar.showMessage(error_msg)
            print(error_msg)
            
            # 尝试回滚到原始模式
            try:
                self.main_window.model_mode = old_model_mode
                self.main_window.pose_processor.update_model(old_model_mode)
                self.main_window.setup_video_thread()
                QTimer.singleShot(500, self.main_window.start_video)
                self.main_window.statusBar.showMessage(f"Rolled back to RTMPose {old_model_mode} mode")
                
            except:
                # 如果回滚也失败，显示严重错误
                self.main_window.statusBar.showMessage("Critical error in RTMPose mode switching") 