from ultralytics import YOLO
import cv2


image=cv2.imread("images/cars.png")
print()


width = 1000
height = 1000
dimensions = (width, height)

resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

cv2.imwrite("images/resized_cars.png", resized_image)

model=YOLO('/yolo_weights/yolov8n.pt')
# results=model("images/cars.png", show=True);
results = model("images/resized_cars.png", show=True)
cv2.waitKey(0)


