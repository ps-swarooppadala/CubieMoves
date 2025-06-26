import cv2
import os

# Create a folder to save the images
output_folder = "RubiksCubeImages"
os.makedirs(output_folder, exist_ok=True)

# Instructions and filenames for each face
instructions = [
    ("top face", "U.png"),
    ("front face", "F.png"),
    ("right face", "R.png"),
    ("back face", "B.png"),
    ("left face", "L.png"),
    ("down face", "D.png")
]

def capture_square_image(frame):
    """Crops the center of the frame to a square."""
    height, width, _ = frame.shape
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    return frame[start_y:start_y+size, start_x:start_x+size]

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for i, (instruction, filename) in enumerate(instructions):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the instructions and the live video feed
        square_frame = capture_square_image(frame)
        cv2.putText(square_frame, f"Position: {instruction} (Press 'c' to capture)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Rubik's Cube Face", square_frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the square image
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, square_frame)
            print(f"Saved {instruction} as {filename}")
            break
        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"All images saved in folder: {output_folder}")
