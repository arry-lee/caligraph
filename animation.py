"""
笔记动线还原算法
描述：一张黑色图片上有一个白色的连通域，找到一条路径点列表，连成折线可以代表这个路径

1.找到笔记的起点和终点
2.边缘检测，只保留边缘线，

"""
import math
import random
import cv2
import imageio
import numpy as np

from scipy.spatial import distance_matrix

from inscribed_circle import find_inscribed_circle_center
from strokes import char_frames, char_strokes
from loguru import logger


def find_two_end(pts):
    """
    绘制外界矩形，按长边分成两半，可以相信每一半有个端点，用
    :param pts:
    :type pts:
    :return:
    :rtype:
    """


def reconstruct_path(binary):
    # 找到连通域的轮廓
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    logger.debug(len(contours))
    # 对轮廓进行简化或拟合，获得路径点列表
    # 根据轮廓长度进行排序
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 选择长度最长的轮廓
    longest_contour = sorted_contours[0]

    # 对轮廓进行简化或拟合，获得路径点列表
    epsilon = max(0.1 * cv2.arcLength(longest_contour, True), 2)
    # logger.debug(epsilon)

    approx = cv2.approxPolyDP(longest_contour, 4, True)
    path = approx.squeeze().tolist()
    logger.debug(path)

    if path:
        path.append(path[0])

    return path, longest_contour.squeeze().tolist()


def high_quality_mask(binary, epsilon=0.001, size=None):
    # 找到连通域的轮廓
    # if size:
    #     binary = cv2.resize(binary,size)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_SIMPLE
    )

    logger.debug(len(contours))
    # 对轮廓进行简化或拟合，获得路径点列表
    # 根据轮廓长度进行排序
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 选择长度最长的轮廓
    longest_contour = sorted_contours[0]

    # 对轮廓进行简化或拟合，获得路径点列表
    epsilon = min(epsilon * cv2.arcLength(longest_contour, True), 2)
    # logger.debug(epsilon)

    approx = cv2.approxPolyDP(longest_contour, 1, True)
    if size:
        resized_approx = np.array(approx * size[0] / 150, np.int32)
    else:
        resized_approx = approx

    # mask = np.zeros_like(binary, np.uint8)
    mask = np.zeros(size, np.uint8)

    cv2.fillPoly(mask, [resized_approx], 255)
    return mask


def compute_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    cross_product = np.cross(vector1, vector2)
    dot_product = np.dot(vector1, vector2)

    angle = np.arctan2(cross_product, dot_product)

    if angle < 0:
        angle += 2 * np.pi

    return np.degrees(angle)


def find_min_angle_vertices(pts):
    """找笔画多边形的起点和终点
    """
    num_vertices = len(pts)

    x, y, w, h = cv2.boundingRect(pts)

    if w > h:
        half = x + w // 2
    else:
        half = y + h // 2

    first = []
    second = []

    # angles = []
    for i in range(0, num_vertices):
        # 获取当前顶点及相邻的两个顶点
        vertex1 = pts[i - 1]
        vertex2 = pts[i]
        vertex3 = pts[(i + 1) % num_vertices]

        # 计算当前顶点与相邻两个顶点形成的夹角
        angle = compute_angle(vertex1, vertex2, vertex3)
        if angle <= 120:
            if w > h:
                if vertex2[0] < half:
                    first.append((angle, i))
                else:
                    second.append((angle, i))
            else:
                if vertex2[1] < half:
                    first.append((angle, i))
                else:
                    second.append((angle, i))
            # angles.append((angle, i))
    first.sort()
    second.sort()
    if len(first) > 0 and len(second) > 0:
        return (first[0], second[0])
    else:
        raise Exception("没有找到两个以上的锐角")

    # 下面是旧算法
    # angles.sort()
    # if len(angles) == 2:
    #     return angles
    # elif len(angles) > 2:
    #     # if angles[1]-angles[0]<
    #     a = angles.pop(0)
    #     si = a[1]
    #     d = (num_vertices) // 2
    #     # 多个锐角的话，找到最小的和与最小的距离最远的一个
    #     angles.sort(
    #         key=lambda x: (abs(x[1] - si), x[0])  # 坐标到最小点,坐标距离可能是一样的此时用最小的角
    #         if d >= abs(x[1] - si)
    #         else (num_vertices - abs(x[1] - si), x[0]),
    #         reverse=True,
    #     )
    #     return [a, angles[0]]
    # else:
    #     raise Exception(f"没有找到两个以上的锐角{angles}")


def find_center_radius(points1, points2):
    if len(points1) < len(points2):
        points1, points2 = points2, points1
    # 计算距离矩阵
    distances = distance_matrix(points1, points2)

    # 找到每个点的最小距离及其索引
    min_distances = np.min(distances, axis=1)
    min_distance_indices = np.argmin(distances, axis=1)

    # 找到最短距离对应的点
    closest_points = points2[min_distance_indices]

    # 计算中点
    midpoint = (points1 + closest_points) / 2

    return np.array(midpoint, np.int32), np.array(min_distances / 2, np.int32), closest_points


def find_start_end_point(points, stroke_code):
    """找到points中的起点和终点index"""
    angles = find_min_angle_vertices(np.array(points, np.int32))
    s, e = angles[0][1], angles[1][1]
    # assert points[s] in contours and points[e] in contours
    if stroke_code == "1":
        if points[s][0] >= points[e][0]:
            s, e = e, s
    else:
        if points[s][1] >= points[e][1]:  # 除了横以外，一般情况y坐标大的为终点
            s, e = e, s
    return s, e


# def animation(char, fp=None, approx=False):
#     """将文字转换成书写GIF"""
#     if len(char) != 1:
#         raise ValueError('only support single character')
#     # 读取原始图像
#     if not fp:
#         fp = f"{char}.mp4"
#
#     W, H = 150, 150
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video_writer = cv2.VideoWriter(fp, fourcc, 30.0, (W, H))
#     gif_writer = imageio.get_writer(f"{char}.gif", fps=30)
#
#     bg = np.zeros((W, H, 3), np.uint8)
#     key_frame = np.zeros((W, H), np.uint8)
#     for stroke_image, stroke_code in zip(char_frames(char, vibe=False)[1:], char_strokes(char)):
#         msk = np.zeros((W, H), np.uint8)
#         mask = stroke_image.point(lambda x: 0 if x > 127 else 255)
#         image = np.asarray(mask, np.uint8)
#         # 还原笔记动线
#
#         # 绘制路径闭合多边形
#         path, contours = reconstruct_path(image)
#
#         image = high_quality_mask(image)
#
#         logger.debug(len(path))
#
#         r = find_inscribed_circle_center(path)[2]  # 找到多边形的最大○
#         logger.debug(f'max ratio {r}')
#
#         points = path[:-1]
#
#         s, e = find_start_end_point(points, stroke_code)
#         start_point, end_point = points[s], points[e]
#
#         if approx:
#             if s < e:
#                 curves = points[s: e + 1]  # 两条轨迹
#                 curves2 = points[e:] + points[: s + 1]
#                 curves2.reverse()
#             else:
#                 curves = points[s:] + points[: e + 1]
#                 curves2 = points[e: s + 1]
#                 curves2.reverse()
#         else:
#             s = contours.index(points[s])
#             e = contours.index(points[e])
#             if s < e:
#                 curves = contours[s: e + 1]  # 两条轨迹
#                 curves2 = contours[e:] + contours[: s + 1]
#                 curves2.reverse()
#             else:
#                 curves = contours[s:] + contours[: e + 1]
#                 curves2 = contours[e: s + 1]
#                 curves2.reverse()
#
#         logger.debug("点数", len(curves), len(curves2))
#         if len(curves) < len(curves2):
#             curves, curves2 = curves2, curves
#
#         # # 计算所有公切圆
#         centers, rads, closest_points = find_center_radius(np.array(curves), np.array(curves2))
#         logger.debug(rads)
#
#         pts = np.array(path, np.int32)
#         pts = pts.reshape((-1, 1, 2))
#
#         cv2.fillPoly(bg, [pts], (255, 255, 255))
#         cv2.polylines(
#             bg,
#             [pts],
#             True,
#             (255, 0, 0),
#             1,
#         )
#         cv2.imshow("Original Image", bg)
#         cv2.circle(bg, start_point, 5, (0, 255, 255))  # 起点
#         cv2.circle(bg, end_point, 5, (255, 255, 0))  # 终点
#
#         for i in range(len(curves)):
#             # 找到 curves[i] 到 curves2 的最短距离
#             # cv2.line(msk, curves[i], closest_points[i], 255, 7)  # todo 半径要随着位置变化
#             # cv2.circle(msk, curves[i], closest_points[i], 255, 7)
#             # cv2.line(bg, curves[i], curves[i + 1], (0, 255, 0), 10 + rads[i] * 2)
#             # cv2.line(bg, curves[i], curves[i + 1], (255, 0, 0), 2)
#             cv2.line(bg, curves[i], closest_points[i], (255, 0, 0), 5)
#
#             cv2.circle(msk, centers[i], 1 + int(2.2 * rads[i]), 255, -1)
#             fr = cv2.bitwise_and(image, msk)
#             key_frame = cv2.bitwise_or(key_frame, fr)  # fixme
#
#             video_writer.write(cv2.cvtColor(key_frame, cv2.COLOR_GRAY2BGR))
#             # 写入gif文件中
#             gif_writer.append_data(cv2.cvtColor(key_frame, cv2.COLOR_BGR2RGB))
#
#         for i in range(len(centers) - 1):
#             cv2.line(bg, centers[i], centers[i + 1], (0, 255, 255), 2)
#
#         # # 显示结果
#         cv2.imshow("Original Image", bg)
#         cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite(f"{char}.png", bg)
#     video_writer.release()
#     gif_writer.close()
#

def sample_centers(centers, approx_centers):
    indexs = []
    for center in approx_centers:
        i = centers.index(center)
        indexs.append(i)
    samples = []
    for i in range(len(indexs) - 1):
        ls = list(range(indexs[i], indexs[i + 1]))
        samples.extend(nonlinear_sample_list(ls, min(24, len(ls))))
    return set(samples)


def nonlinear_sample_list(lst, num_samples):
    length = len(lst)
    if num_samples >= length:
        return lst

    # 根据采样数量生成非线性变换的参数
    x = np.linspace(0, 1, num_samples // 2)  # 在区间 [0, 1] 上均匀采样
    y = x ** 2  # 采用 sin 函数进行非线性变换

    # 根据参数对原始列表进行非线性采样
    samples = []
    for i in range(num_samples // 2):
        index = math.ceil(y[i] * ((length - 1) / 2))
        samples.append(index)
    samples.extend([length - 1 - i for i in samples[-2::-1]])
    # samples.append(samples[-1])
    return [lst[i] for i in samples]


def generate_frames(char, approx=False, speed=None, size=None):
    """生成帧"""
    if len(char) != 1:
        raise ValueError('only support single character')
    if not size:
        W, H = 150, 150
        ratio = 1
    else:
        W, H = size
        ratio = W / 150
    bg = np.zeros((W, H, 3), np.uint8)
    key_frame = np.zeros((W, H), np.uint8)
    for stroke_image, stroke_code in zip(char_frames(char, vibe=False)[1:], char_strokes(char)):
        msk = np.zeros((150, 150), np.uint8)
        mask = stroke_image.point(lambda x: 0 if x > 127 else 255)
        image = np.asarray(mask, np.uint8)

        path, contours = reconstruct_path(image)

        points = path[:-1]
        s, e = find_start_end_point(points, stroke_code)
        start_point, end_point = points[s], points[e]
        if approx:
            if s < e:
                curves = points[s: e + 1]
                curves2 = points[e:] + points[: s + 1]
                curves2.reverse()
            else:
                curves = points[s:] + points[: e + 1]
                curves2 = points[e: s + 1]
                curves2.reverse()
        else:
            s = contours.index(points[s])
            e = contours.index(points[e])
            if s < e:
                curves = contours[s: e + 1]
                curves2 = contours[e:] + contours[: s + 1]
                curves2.reverse()
            else:
                curves = contours[s:] + contours[: e + 1]
                curves2 = contours[e: s + 1]
                curves2.reverse()

        if len(curves) < len(curves2):
            curves, curves2 = curves2, curves

        centers, rads, closest_points = find_center_radius(np.array(curves), np.array(curves2))

        approx_centers = cv2.approxPolyDP(centers, 2, False).squeeze().tolist()

        if speed is None:
            speeds = sample_centers(centers.tolist(), approx_centers)

        pts = np.array(path, np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(bg, [pts], (255, 255, 255))

        cv2.polylines(bg, [pts], True, (255, 0, 0), 1)
        cv2.circle(bg, start_point, 5, (0, 255, 255))
        cv2.circle(bg, end_point, 5, (255, 255, 0))

        image = high_quality_mask(image, 0.001, size)
        # corners = cv2.goodFeaturesToTrack(image, maxCorners=2, qualityLevel=0.5, minDistance=7)

        for i in range(len(curves)):
            cv2.line(bg, curves[i], closest_points[i], (255, 0, 0), 5)  # todo 通过i来控制采样速度
            cv2.circle(msk, centers[i], 1 + int(2.2 * rads[i]), 255, -1)

            _mask = cv2.resize(msk, size)

            fr = cv2.bitwise_and(image, _mask)
            key_frame = cv2.bitwise_or(key_frame, fr)

            if speed:
                if i % speed == 0:
                    yield key_frame
            else:
                if i in speeds:
                    yield key_frame

        # 在原始图像上绘制角点
        # for corner in np.int0(corners):
        #     x, y = corner.ravel()
        #     cv2.circle(bg, (x, y), 3, (0, 255, 0), -1)

        for i in range(len(approx_centers) - 1):
            cv2.line(bg, approx_centers[i], approx_centers[i + 1], (0, 255, 255), 2)
            # # 显示结果
        #     cv2.imshow("Original Image", bg)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()
        yield key_frame


def write_to_file(char, fp=None, size=None):
    """写入文件"""
    if not fp:
        fp = f"{char}.mp4"
    if not size:
        size = (150, 150)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(fp, fourcc, 30.0, size)
    gif_writer = imageio.get_writer(f"{char}.gif", fps=30)

    for key_frame in generate_frames(char, size=size):
        video_writer.write(cv2.cvtColor(key_frame, cv2.COLOR_GRAY2BGR))
        gif_writer.append_data(cv2.cvtColor(key_frame, cv2.COLOR_BGR2RGB))

    video_writer.release()
    gif_writer.close()


def animation(text, height=150):
    n = len(text)
    size = (height, height)
    key_frame = np.zeros((height, height * n), np.uint8)
    for i, char in enumerate(text):
        for frame in generate_frames(char, approx=False, speed=None, size=size):
            x = i * height
            y = 0
            key_frame[y:y + height, x:x + height] = frame
            yield key_frame
