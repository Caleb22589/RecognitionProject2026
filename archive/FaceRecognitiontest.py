import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press q to quit.")

while True:
    # read every frame from the webcam and then analyze it for emotions using DeepFace
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        if isinstance(analysis, list):
            analysis = analysis[0]
            
        dominant_emotion = analysis['dominant_emotion']
        
        # Get face coordinates to draw a box (if a face was detected)
        region = analysis['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        pass 

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()