
<h1 align="center">Yolo Model Working Demonstration</h1>

![Project Logo](https://github.com/Nithinrkashyap/Computer-Vision-car-counter-using-Yolo-/blob/main/Vectorization_object_detection_yolo.jpg?raw=true)


<h4>
  YOLO Model Working Demonstration - Car Counter
The YOLOv8-based Car Counting System is designed for real-time vehicle detection and tracking in videos. Leveraging the powerful object detection capabilities of the YOLOv8 model, this system accurately identifies and tracks various types of vehicles such as cars, trucks, buses, and motorcycles within a designated region of interest (ROI).

Key functionalities include:

<ul>

  <li>
    Vehicle Detection: YOLOv8 detects vehicles in each frame, highlighting them with bounding boxes.
  </li>
  <li>
    Object Tracking: The SORT (Simple Online and Realtime Tracking) algorithm tracks detected vehicles, ensuring that each vehicle is given a unique ID across multiple frames, allowing for accurate vehicle counting.
  </li>
 <li>
   Non-Maximum Suppression (NMS): This is used to retain the most relevant bounding boxes when multiple overlapping detections occur for the same object.
Bitwise Masking: A bitwise mask is applied to focus detection within a specific region, improving accuracy.
 </li>

 <li>
   
Real-Time Feedback: The system provides real-time visual feedback by overlaying the bounding boxes and the vehicle count directly on the video feed.
 </li>
</ul>





</h4>


