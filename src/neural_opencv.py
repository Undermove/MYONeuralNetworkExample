import cv2
from simple_network_with_backquery import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

output_video_width = 240
output_video_height = 240
pixels_count = output_video_width * output_video_height
epochs = 5

network = NeuralNetwork(1, pixels_count, 2000, 2, .1)

cap = cv2.VideoCapture('video_dataset/golf/Evian_Masters_Junior_Cup_Highlights_2009_golf_f_nm_np1_ri_goo_3.avi')

if not cap.isOpened():
    print("Error opening video stream or file")

for i in range(epochs):
    while True:
        isTrue, frame = cap.read()
        if isTrue is False:
            break

        resizedFrame = cv2.resize(frame, (output_video_height, output_video_width), fx=0, fy=0,
                                  interpolation=cv2.INTER_CUBIC)
        grayResizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        flattenResizedFrame = np.array(grayResizedFrame).flatten()
        flattenResizedFrameScaled = np.asfarray(flattenResizedFrame) / 255.0 * 0.99
        scaled_input = flattenResizedFrameScaled + 0.01

        queryResult = network.query(scaled_input)

        network.train(scaled_input, [1, 0])
        backQueryResult = network.backquery([0.999, 0.0001])

        reshaped = np.reshape(backQueryResult, (output_video_height, output_video_width))

        plt.imshow(reshaped, interpolation="nearest", origin="upper")
        plt.colorbar()
        plt.show()

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Success")

    cap = cv2.VideoCapture('video_dataset/kick_ball/Amazing_Soccer_#2_kick_ball_f_cm_np1_ba_bad_5.avi')

    if not cap.isOpened():
        print("Error opening video stream or file")

    while True:
        isTrue, frame = cap.read()
        if isTrue is False:
            break

        resizedFrame = cv2.resize(frame, (output_video_height, output_video_width), fx=0, fy=0,
                                  interpolation=cv2.INTER_CUBIC)
        grayResizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        flattenResizedFrame = np.array(grayResizedFrame).flatten()
        flattenResizedFrameScaled = np.asfarray(flattenResizedFrame) / 255.0 * 0.99
        scaled_input = flattenResizedFrameScaled + 0.01

        queryResult = network.query(scaled_input)
        network.train(scaled_input, [0, 1])
        backQueryResult = network.backquery(queryResult.T)

        reshaped = np.reshape(backQueryResult, (output_video_height, output_video_width))

        plt.imshow(reshaped, interpolation="nearest", origin="upper")
        plt.colorbar()
        plt.show()

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

cap.release()
cv2.destroyAllWindows()

# 1. Берем кадр, меняем размер 
# 2. Делаем флаттен на него 
# 3. Скалируем изображение
# 3. Тренируем сеть 
# 4. Делаем обратный запрос 
# 5. Формируем кадр из него
# 6. Сохраняем этот кадр в новое видео
