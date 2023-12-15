import cv2

image_path = "learning_history.jpg"
image = cv2.imread(image_path)

cv2.imshow("learning_history", image)
cv2.waitKey(0)  # 顯示圖片並等待任意按鍵關閉視窗
cv2.destroyAllWindows()
