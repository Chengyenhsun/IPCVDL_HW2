import cv2

image_path = "Accuracy_Comparison.jpg"
image = cv2.imread(image_path)

cv2.imshow("Accuracy_Comparison", image)
cv2.waitKey(0)  # 顯示圖片並等待任意按鍵關閉視窗
cv2.destroyAllWindows()
