"""
笔记动线还原算法
描述：一张黑色图片上有一个白色的连通域，找到一条路径点列表，连成折线可以代表这个路径

1.找到笔记的起点和终点
2.边缘检测，只保留边缘线，

"""
import random
import cv2
import numpy as np

from scipy.spatial import distance_matrix
from strokes import char_frames, char_strokes


def reconstruct_path(binary):
    # 找到连通域的轮廓
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    print(len(contours))
    # 对轮廓进行简化或拟合，获得路径点列表
    # 根据轮廓长度进行排序
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 选择长度最长的轮廓
    longest_contour = sorted_contours[0]

    # 对轮廓进行简化或拟合，获得路径点列表
    epsilon = max(0.1 * cv2.arcLength(longest_contour, True), 2)
    # print(epsilon)

    approx = cv2.approxPolyDP(longest_contour, 4, True)
    path = approx.squeeze().tolist()
    print(path)

    if path:
        path.append(path[0])

    return path, longest_contour


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
    num_vertices = len(pts)
    min_angle = float("inf")
    min_angle_vertices = []

    angles = []
    for i in range(0, num_vertices):
        # 获取当前顶点及相邻的两个顶点
        vertex1 = pts[i - 1]
        vertex2 = pts[i]
        vertex3 = pts[(i + 1) % num_vertices]

        # 计算当前顶点与相邻两个顶点形成的夹角
        angle = compute_angle(vertex1, vertex2, vertex3)
        if angle <= 180:
            angles.append((angle, i))

    angles.sort()
    if len(angles) == 2:
        return angles
    elif len(angles) > 2:
        a = angles.pop(0)
        si = a[1]
        d = (num_vertices) // 2
        # 多个锐角的话，找到最小的和与最小的距离最远的一个
        angles.sort(
            key=lambda x: abs(x[1] - si)
            if d >= abs(x[1] - si)
            else num_vertices - abs(x[1] - si),
            reverse=True,
        )
        return [a, angles[0]]
    else:
        raise Exception(f"没有找到两个以上的锐角{angles}")


def find_cr(points1, points2):
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

    return np.array(midpoint, np.int32), np.array(min_distances / 2, np.int32)


def animation(char, fp="animation.mp4"):
    # 读取原始图像
    output_width, output_height = 150, 150
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(fp, fourcc, 30.0, (output_width, output_height))
    bg = np.zeros((150, 150, 3), np.uint8)
    frame = np.zeros((150, 150), np.uint8)
    msk = np.zeros((150, 150), np.uint8)
    for image, stroke in zip(char_frames(char, vibe=False)[1:], char_strokes(char)):
        # image = char_frames('国')[4]

        mask = image.point(lambda x: 0 if x > 127 else 255)
        # mask.show()
        image = np.asarray(mask, np.uint8)
        # 还原笔记动线
        frame = cv2.bitwise_or(frame, image)

        path, contours = reconstruct_path(image)

        print(len(path))
        # 绘制路径
        # 绘制路径闭合多边形
        r = 7  # find_inscribed_circle_center(path)[2] #找到多边形的最大○
        rad = int(r * 2)
        print(rad)
        print(path)
        if path:
            pts = np.array(path, np.int32)
            points = path[:-1]
            angles = find_min_angle_vertices(np.array(points, np.int32))
            s, e = angles[0][1], angles[1][1]

            assert points[s] in contours and points[e] in contours

            if stroke == "1":
                if points[s][0] >= points[e][0]:
                    s, e = e, s
            else:
                if points[s][1] >= points[e][1]:  # 除了横以外，一般情况y坐标大的为终点
                    s, e = e, s
            if s < e:
                curves = points[s: e + 1]  # 两条轨迹
                curves2 = points[e:] + points[: s + 1]
                curves2.reverse()
            else:
                curves = points[s:] + points[: e + 1]
                curves2 = points[e: s + 1]
                curves2.reverse()
                # min_,max_ = e,s
            contours = [p[0].tolist() for p in contours]
            print(contours)
            print(type(points))
            ss, ee = s, e
            s = contours.index(points[s])
            e = contours.index(points[e])

            if s < e:
                curves = contours[s: e + 1]  # 两条轨迹
                curves2 = contours[e:] + contours[: s + 1]
                curves2.reverse()
            else:
                curves = contours[s:] + contours[: e + 1]
                curves2 = contours[e: s + 1]
                curves2.reverse()

            print("点数", len(curves), len(curves2))
            if len(curves) < len(curves2):
                curves, curves2 = curves, curves2
            # # 计算所有公切圆
            centers, rads = find_cr(np.array(curves), np.array(curves2))
            # print(centers)
            print(rads)
            # # 可视化结果
            # plot_common_tangent_circles(circles, np.array(curves), np.array(curves2))

            for i in range(len(centers) - 1):
                cv2.line(bg, centers[i], centers[i + 1], (0, 255, 255), 1)
                # cv2.circle(bg, centers[i], rads[i], (0, 0, 255), 1)

            for i in range(len(curves) - 1):
                # 找到 curves[i] 到 curves2 的最短距离

                cv2.line(
                    msk, curves[i], curves[i + 1], 255, 10 + rads[i] * 2
                )  # todo 半径要随着位置变化
                cv2.line(
                    bg, curves[i], curves[i + 1], (255, 255, 255), 10 + rads[i] * 2
                )
                cv2.line(bg, curves[i], curves[i + 1], (255, 0, 0), 2)
                cv2.circle(bg, centers[i], 1 + int(7 * rads[i]), (0, 0, 255), 1)
                # cv2.circle(msk,curves[i], 14+rads[i]*2, 255, -1)

                fr = cv2.bitwise_and(frame, msk)
                video_writer.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))

            cv2.circle(bg, (points[ss]), 5, (0, 255, 255))  # 起点
            cv2.circle(bg, (points[ee]), 5, (255, 255, 0))  # 终点

            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                bg,
                [pts],
                True,
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                1,
            )
            cv2.fillPoly(bg, [pts], (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

        # 显示结果
        cv2.imshow("Original Image", bg)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("rui.png", bg)
    video_writer.release()
