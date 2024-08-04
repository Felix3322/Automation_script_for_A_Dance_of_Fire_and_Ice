import cv2
import numpy as np
import mss
from collections import deque


# 计算两个点之间的欧几里得距离
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 检测并返回对比度较大的色块
def find_high_contrast_areas(frame, min_area=3600, max_area=15000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 使用形态学操作来增强边缘检测
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 填充边缘中的空隙并进行分割
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            detected_contours.append(contour)
    return detected_contours


# 启动屏幕捕捉
with mss.mss() as sct:
    # 获取屏幕的尺寸
    monitor = sct.monitors[1]

    # 创建一个可调整大小且置顶的窗口
    window_name = "Detected Blocks and Balls"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    # 用于存储最近五帧的圆体位置
    recent_circles = deque(maxlen=5)
    recent_contours = deque(maxlen=5)

    while True:
        # 捕捉屏幕
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # 检测对比度较大的色块
        initial_color_blocks = find_high_contrast_areas(frame)

        # 检测球体
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=100, param2=30, minRadius=25, maxRadius=40)

        current_circles = []

        # 确保至少找到一个圆
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center_x, center_y, radius = i[0], i[1], i[2]

                # 创建一个掩膜来检测圆内的像素
                mask = np.zeros_like(gray)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                circle_area = cv2.bitwise_and(gray, gray, mask=mask)

                # 计算圆内的白色像素（假设背景较暗）
                circle_pixels = cv2.countNonZero(circle_area)
                expected_pixels = np.pi * radius ** 2

                # 如果圆内的白色像素接近于理论值，则认为是完整的圆
                if circle_pixels >= 0.9 * expected_pixels:
                    current_circles.append((center_x, center_y, radius))

        # 将当前帧的圆体位置添加到队列
        recent_circles.append(current_circles)
        recent_contours.append(initial_color_blocks)

        # 查找相对静止的色块
        static_blocks = []
        for contour in initial_color_blocks:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # 计算色块在历史帧中的平均位置变化
                total_distance = 0
                valid_count = 0

                for history in recent_contours:
                    for hist_contour in history:
                        hist_M = cv2.moments(hist_contour)
                        if hist_M['m00'] != 0:
                            hist_cx = int(hist_M['m10'] / hist_M['m00'])
                            hist_cy = int(hist_M['m01'] / hist_M['m00'])
                            total_distance += calculate_distance((cx, cy), (hist_cx, hist_cy))
                            valid_count += 1

                if valid_count > 0:
                    avg_movement = total_distance / valid_count

                    # 如果色块的平均位置变化小于一定阈值，则认为是静止的
                    if avg_movement < 5:
                        static_blocks.append(contour)

        # 绘制球体
        for (center_x, center_y, radius) in current_circles:
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)

        # 绘制相对静止的色块
        for contour in initial_color_blocks:
            color_block = np.zeros_like(frame)
            cv2.drawContours(color_block, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            mask = cv2.cvtColor(color_block, cv2.COLOR_BGR2GRAY)
            is_static = contour in static_blocks
            color = (0, 0, 255) if is_static else (255, 0, 0)  # 红色：静止，蓝色：不静止

            # 确定色块是否包含球体中心点
            for (center_x, center_y, _) in current_circles:
                if mask[center_y, center_x] == 255:
                    cv2.drawContours(frame, [contour], -1, color, 2)
                    break
            else:
                cv2.drawContours(frame, [contour], -1, color, 2)

        # 显示结果帧
        cv2.imshow(window_name, frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 关闭所有窗口
cv2.destroyAllWindows()
