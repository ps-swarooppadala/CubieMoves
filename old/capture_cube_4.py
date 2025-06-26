import cv2
import os
import numpy as np

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

# Mapping of average color to Rubik's Cube colors
# color_map = {
#     "W": [255, 255, 255],  # White
#     "G": [0, 255, 0],      # Green
#     "R": [255, 0, 0],      # Red
#     "B": [0, 0, 255],      # Blue
#     "O": [255, 165, 0],    # Orange
#     "Y": [255, 255, 0]     # Yellow
# }

# Mapping of average color to Rubik's Cube colors BGR
color_map = {
    "W": [112, 103, 97],  # White
    "G": [39, 114, 40],      # Green
    "R": [47, 29, 120],      # Red
    "B": [88, 17, 0],      # Blue
    "O": [55, 78, 164],    # Orange
    "Y": [36, 104, 118]     # Yellow
}

def enhance_color(image):
    """Enhances the brightness and contrast of the image."""
    alpha = 1.5  # Contrast control
    beta = 50    # Brightness control
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

def map_color(avg_color):
    """Maps an average color to the nearest predefined color."""
    def color_distance(c1, c2):
        return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

    closest_color = min(color_map.keys(), key=lambda k: color_distance(avg_color, color_map[k]))
    return closest_color

def capture_square_image(frame):
    """Crops the center of the frame to a square."""
    height, width, _ = frame.shape
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    return frame[start_y:start_y+size, start_x:start_x+size]

def extract_colors(image):
    """Uses a circular mask to extract colors from the center of each grid section."""
    h, w, _ = image.shape
    cell_size = h // 3
    colors = []
    for i in range(3):
        row = []
        for j in range(3):
            cell_center_x = j * cell_size + cell_size // 2
            cell_center_y = i * cell_size + cell_size // 2
            radius = cell_size // 4

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cell_center_x, cell_center_y), radius, 255, -1)

            masked_cell = cv2.bitwise_and(image, image, mask=mask)
            avg_color_bgr = cv2.mean(masked_cell, mask=mask)[:3]  # Extract average color within the mask
            # avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

            print(f"Average color at row {i + 1}, col {j + 1}: {avg_color_bgr}")

            mapped_color = map_color(avg_color_bgr)
            row.append(mapped_color)
        colors.append(row)
    return colors

def edit_colors(colors):
    """Allows the user to manually edit the detected colors."""
    print("Detected colors:")
    for i, row in enumerate(colors):
        print(f"Row {i + 1}: {row}")

    while True:
        edit = input("Would you like to edit any color? (y/n): ").strip().lower()
        if edit == 'n':
            break
        if edit == 'y':
            try:
                row = int(input("Enter the row (1-3): ")) - 1
                col = int(input("Enter the column (1-3): ")) - 1
                new_color = input("Enter the new color (W, G, R, B, O, Y): ").strip().upper()
                if new_color in color_map:
                    colors[row][col] = new_color
                    print(f"Updated Row {row + 1}, Column {col + 1} to {new_color}")
                else:
                    print("Invalid color. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Try again.")
    return colors

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_colors = {}

for i, (instruction, filename) in enumerate(instructions):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the instructions and the live video feed
        square_frame = capture_square_image(frame)
        # enhanced_frame = enhance_color(square_frame)
        enhanced_frame = square_frame
        cv2.putText(enhanced_frame, f"Position: {instruction} (Press 'c' to capture)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Rubik's Cube Face", enhanced_frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the square image
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, enhanced_frame)
            print(f"Saved {instruction} as {filename}")

            # Convert to color array
            colors = extract_colors(enhanced_frame)
            colors = edit_colors(colors)  # Allow editing of colors
            face_colors[instruction] = colors
            print(f"Final colors for {instruction}:{colors}")
            break
        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the extracted color representation for the whole cube
print("\nRubik's Cube status:")
for face, colors in face_colors.items():
    print(f"{face}: {colors}")

print(f"All images saved in folder: {output_folder}")
