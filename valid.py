import cv2
import matplotlib.pyplot as plt
import numpy as np
from detection import Detect
from humanpose import Pose


def draw_pose(image, pose, color=(0,255,0)):
    kpts = np.array(pose, dtype=int)
    skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                 [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    points_num = [num for num in range(17)]

    for sk in skelenton:
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(image, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points, 0]), int(kpts[points, 1]))
    cv2.circle(image, pos, 4, (0, 0, 255), -1)
    return image

if __name__ == '__main__':
    human_detect = Detect()
    human_pose = Pose()
    img = cv2.imread(r"C://Users/Xavier/Desktop/test.png")
    detect_res = human_detect(img)
    human_box = []
    for x, y, w, h in detect_res:
        x = x if x>=0 else 0
        y = y if y>=0 else 0
        human_box.append(img[y:y+h, x:x+w])
    poses = human_pose(human_box)
    for i in range(len(poses)):
        x,y,w,h = detect_res[i]
        pose = poses[i]
        pose[:,0], pose[:, 1] = pose[:, 0] * w + x, pose[:, 1] * h + y
        draw_pose(img, pose)
        # plt.imshow(img)
        # plt.scatter(pose[:,:,0]*w + x, pose[:,:,1]*h+y)
    # for x, y, w, h in detect_res:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.imshow(img)
    plt.show()
