import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer
from window import *
from detection import Detect
from humanpose import Pose
import numpy as np
from sort import Sort


class mwindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        self.Pose = Pose()
        self.Detect = Detect()
        self.cap = None
        self.ref = False
        self.mot_tracker = None
        self.frame_count = 0
        self.flag = {"video": False, "camera": True, "Detect": False, "Pose": False, "Track": False, "Count": False}
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_image)
        self.timer.start(0)

    def get_video(self):
        self.frame_count = 0
        if self.flag["video"]:
            self.open_camre.setText(u'打开摄像头')
            self.load_video.setText(u'关闭本地视频')
            video_path, _ = QFileDialog.getOpenFileName(self, '选择视频', 'D:\Jupter_book\Qt\onlinepose', 'Video files(*.mp4)')
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.open_camre.setText(u'关闭摄像头')
            self.load_video.setText(u'打开本地视频')
            self.cap = cv2.VideoCapture(0)
        self.ref, image = self.cap.read()
        show = cv2.resize(image, (960, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.img_box.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def get_human_boxes(self, image):
        return self.Detect(image)

    def get_human_pose(self, image):
        return self.Pose(image)

    def draw_box(self, boxes):
        for x, y, w, h in boxes:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 5)

    def draw_pose(self, pose, color=(0, 255, 0)):
        kpts = np.array(pose, dtype=int)
        skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                     [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        points_num = [num for num in range(17)]

        for sk in skelenton:
            pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
            pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
            if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
                cv2.line(self.image, pos1, pos2, color, 2, 8)
        for points in points_num:
            pos = (int(kpts[points, 0]), int(kpts[points, 1]))
        cv2.circle(self.image, pos, 4, (0, 0, 255), -1)

    def change_flag(self, type):
        if type == "video" or type == "camera":
            self.flag["video"] = not self.flag["video"]
            self.flag["camera"] = not self.flag["camera"]
            self.get_video()
        else:
            self.flag[type] = not self.flag[type]

    def sort(self, boxes):
        if self.mot_tracker == None:
            max_age = 300
            min_hits = 10
            iou_threshold = 0.3
            self.mot_tracker = Sort(max_age, min_hits, iou_threshold)
        for i in range(len(boxes)):
            boxes[i][2], boxes[i][3] = boxes[i][2] + boxes[i][0], boxes[i][3]+boxes[i][1]
        track_ed = self.mot_tracker.update(boxes)
        for i in track_ed:
            track_id = i[4]
            # print('track_id:', track_id)
            x1, y1, x2, y2 = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            # print('box:', x1, y1, x2, y2)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(self.image, "track_id: %d " % track_id, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    def pose_estimation(self, boxes):
        human_box = []
        for x, y, w, h in boxes:
            x = x if x >= 0 else 0
            y = y if y >= 0 else 0
            human_box.append(self.image[y:y + h, x:x + w])
        poses = self.Pose(human_box)
        for i in range(len(poses)):
            x, y, w, h = boxes[i]
            pose = poses[i]
            pose[:, 0], pose[:, 1] = pose[:, 0] * w + x, pose[:, 1] * h + y
            self.draw_pose(pose)

    def show_image(self):
        step = 1
        if self.cap:
            self.ref, self.image = self.cap.read()
            self.frame_count += 1
            if self.flag["Detect"] and self.frame_count % step == 0:
                boxes = self.Detect(self.image)
                if self.flag["Track"] and self.frame_count % step == 0:
                    try:
                        self.sort(np.array(boxes))
                    except:
                        pass
                if self.flag["Pose"] and self.frame_count % step == 0:
                    self.pose_estimation(boxes)
                self.draw_box(boxes)
            show = cv2.resize(self.image, (960, 480))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.img_box.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.get_video()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = mwindow()
    w.load_video.clicked.connect(lambda: w.get_video())
    w.load_video.clicked.connect(lambda: w.change_flag("video"))
    w.open_camre.clicked.connect(lambda: w.change_flag("camera"))
    w.detect.clicked.connect(lambda: w.change_flag("Detect"))
    w.pose.clicked.connect(lambda: w.change_flag("Pose"))
    w.track.clicked.connect(lambda: w.change_flag("Track"))
    w.show()
    sys.exit(app.exec_())