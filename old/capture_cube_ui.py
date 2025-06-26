import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

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

# Mapping of average color to Rubik's Cube colors BGR
color_map = {
    "W": [112, 103, 97],  # White
    "G": [39, 114, 40],   # Green
    "R": [47, 29, 120],   # Red
    "B": [88, 17, 0],     # Blue
    "O": [55, 78, 164],   # Orange
    "Y": [36, 104, 118]   # Yellow
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

class RubiksCubeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rubik's Cube Capture and Edit")

        self.face_colors = {}
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.quit()

        self.face_index = 0
        self.instruction_label = tk.Label(root, text=f"Position: {instructions[self.face_index][0]}", font=("Arial", 16))
        self.instruction_label.pack()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.capture_button = tk.Button(root, text="Capture", command=self.capture_face)
        self.capture_button.pack()

        self.next_button = tk.Button(root, text="Next Face", command=self.next_face)
        self.next_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        self.update_video_feed()

    def update_video_feed(self):
        """Update the live video feed."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((640, 480))
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update_video_feed)

    def capture_face(self):
        """Capture the current face image and process the colors."""
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            return

        square_frame = capture_square_image(frame)
        filepath = os.path.join(output_folder, instructions[self.face_index][1])
        cv2.imwrite(filepath, square_frame)

        colors = extract_colors(square_frame)
        colors = edit_colors(colors)  # Allow editing of colors
        self.face_colors[instructions[self.face_index][0]] = colors
        messagebox.showinfo("Captured", f"Captured {instructions[self.face_index][0]} and saved.")

    def next_face(self):
        """Move to the next face for capturing."""
        self.face_index += 1
        if self.face_index >= len(instructions):
            messagebox.showinfo("Completed", "All faces captured.")
            self.quit_app()
            return
        self.instruction_label.config(text=f"Position: {instructions[self.face_index][0]}")

    def quit_app(self):
        """Release resources and quit the application."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Set up the Tkinter root window
root = tk.Tk()
app = RubiksCubeApp(root)

# Start the Tkinter event loop
root.mainloop()
