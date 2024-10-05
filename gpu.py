import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt').cuda()  # Use .cuda() to move the model to the GPU

# Open the webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(frame)

    # Process the results
    for result in results:
        # Draw bounding boxes and labels on the image
        boxes = result.boxes  # Get the boxes from results
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get the coordinates
            conf = box.conf[0].cpu().numpy()  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Class ID
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[cls]}: {conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Webcam Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
