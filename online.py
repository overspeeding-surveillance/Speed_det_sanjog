import cv2
import dlib
import math
import torch

model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

video = cv2.VideoCapture('surya.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(
        location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 9.6
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        _, image = video.read()
        if type(image) == type(None):
            break
        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            result = model(image)
            df = result.pandas().xyxy[0]
            df = df.drop(['confidence', 'name'], axis=1)

            for (_x, _y, _xm, _ym, class_id) in df.values.astype(int):
                x = (_x)
                y = (_y)
                xm = _xm
                ym = _ym
                w = xm-x
                h = ym-y

                x_cen = x + 0.5 * w
                y_cen = y + 0.5 * h

                matchCarID = None
                if (x >= 10 and y >= 10):
                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_cen = t_x + 0.5 * t_w
                        t_y_cen = t_y + 0.5 * t_h

                        if ((t_x <= x_cen <= (t_x + t_w)) and (t_y <= y_cen <= (t_y + t_h)) and (x <= t_x_cen <= (x + w)) and (y <= t_y_cen <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(
                            image, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w, t_y + t_h), rectangleColor, 2)
            cv2.putText(resultImage, str(carID), (int(t_x - 50 + t_w/2),
                        int(t_y+t_h-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    cv2.line(resultImage, (0, 275),
                             (100, 275), (255, 255, 100), 2)
                    cv2.line(resultImage, (0, 285),
                             (100, 285), (255, 255, 100), 2)
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed(
                            [x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 70:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(
                            x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('result', resultImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
