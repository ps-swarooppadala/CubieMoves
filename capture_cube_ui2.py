import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import json

# Create a folder to save the images
output_folder = "RubiksCubeImages"
os.makedirs(output_folder, exist_ok=True)

# Instructions and filenames for each face
instructions = [
    ("top", "U.png"),
    ("left", "L.png"),
    ("front", "F.png"),
    ("right", "R.png"),
    ("back", "B.png"),
    ("down", "D.png")
]

class RubiksCubeState:
    """Class to store and manage Rubik's Cube faces."""
    def __init__(self):
        # Initialize a dictionary to store face data
        self.faces = {
            "top": None,
            "left": None,
            "front": None,
            "right": None,
            "back": None,
            "down": None,
        }

    def save_face(self, face_name, colors):
        """Save the colors for a given face."""
        if face_name in self.faces:
            self.faces[face_name] = colors

    def get_face(self, face_name):
        """Retrieve the colors for a given face."""
        return self.faces.get(face_name)

    def is_complete(self):
        """Check if all faces have been captured."""
        print(self.faces)
        return all(face is not None for face in self.faces.values())
    
    def save_to_file(self, filename="RubiksCubeState.json"):
        """Save the cube state to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.faces, f, indent=4)
        print(f"Rubik's Cube state saved to {filename}.")


# Mapping of average color to Rubik's Cube colors BGR
color_map = {
    "W": [112, 103, 97],  # White
    "G": [39, 114, 40],   # Green
    "R": [47, 29, 120],   # Red
    "B": [88, 17, 0],     # Blue
    "O": [55, 78, 164],   # Orange
    "Y": [36, 104, 118]   # Yellow
}

def capture_square_image(frame):
    """Crops the center of the frame to a square."""
    height, width, _ = frame.shape
    size = min(height, width)  # Use the smaller dimension for square size
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    return frame[start_y:start_y+size, start_x:start_x+size]

def map_color(avg_color):
    """Maps an average color to the nearest predefined color."""
    def color_distance(c1, c2):
        return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

    closest_color = min(color_map.keys(), key=lambda k: color_distance(avg_color, color_map[k]))
    return closest_color

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

class RubiksCubeApp:
    def __init__(self, root):
        self.root = root
        self.cube_state = RubiksCubeState()
        self.root.title("Rubik's Cube Capture and Edit")

        self.face_colors = {}
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.quit()

        self.face_index = 0
        
        self.instruction_label = tk.Label(root, text=f"Position: {instructions[self.face_index][0]}", font=("Arial", 16))
        self.instruction_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.video_frame = tk.Frame(root)
        self.video_frame.grid(row=1, column=0)

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()

        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=1, column=1, padx=20, sticky="n")

        self.grid_frame = tk.Frame(self.control_frame)
        self.grid_frame.pack(pady=10)

        self.capture_button = tk.Button(self.control_frame, text="Capture", command=self.capture_face, width=15)
        self.capture_button.pack(pady=5)

        self.next_button = tk.Button(self.control_frame, text=f"Next: {instructions[self.face_index + 1][0]}", command=self.next_face, width=15)
        self.next_button.pack(pady=5)

        self.quit_button = tk.Button(self.control_frame, text="Quit", command=self.quit_app, width=15)
        self.quit_button.pack(pady=5)

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
        self.face_colors[instructions[self.face_index][0]] = colors
        self.cube_state.save_face(instructions[self.face_index][0], colors)  # Save to RubiksCubeState
        print(self.cube_state.get_face(instructions[self.face_index][0]))
        self.display_color_grid(colors)
        # messagebox.showinfo("Captured", f"Captured {instructions[self.face_index][0]} and saved.")

    def display_color_grid(self, colors):
        """Display the captured colors in a grid layout."""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        for i, row in enumerate(colors):
            for j, color in enumerate(row):
                color_label = tk.Label(
                    self.grid_frame,
                    text=color,
                    bg=self.get_color_code(color),
                    width=8,
                    height=4,
                    relief="solid"
                )
                color_label.grid(row=i, column=j, padx=2, pady=2)
                color_label.bind("<Button-1>", lambda e, r=i, c=j: self.edit_color(r, c))

    def edit_color(self, row, col):
        """Edit the color of a specific cell."""
        face_name = instructions[self.face_index][0]
        current_color = self.face_colors[face_name][row][col]
        new_color = simpledialog.askstring(
            "Edit Color", f"Enter new color for cell ({row + 1}, {col + 1}):", initialvalue=current_color
        )
        if new_color and new_color.upper() in color_map:
            self.face_colors[face_name][row][col] = new_color.upper()
            self.display_color_grid(self.face_colors[face_name])
        else:
            messagebox.showwarning("Invalid Color", "Please enter a valid color (W, G, R, B, O, Y).")

    def get_color_code(self, color):
        """Get the hex code for a color."""
        bgr = color_map[color]
        return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

    def next_face(self):
        """Move to the next face for capturing."""
        self.face_index += 1
        if self.face_index >= len(instructions):
            self.cube_state.save_to_file("RubiksCubeState.json")
            messagebox.showinfo("Completed", "All faces captured. State saved to 'RubiksCubeState.json'.")
            self.quit_app()
            return        
        # if self.face_index >= len(instructions):
        #     messagebox.showinfo("Completed", "All faces captured.")
        #     self.quit_app()
        #     return
        self.instruction_label.config(text=f"Position: {instructions[self.face_index][0]}")

        # Update the button text for the next face
        next_face_text = f"Next: {instructions[self.face_index + 1][0]}" if self.face_index + 1 < len(instructions) else "Finish"
        self.next_button.config(text=next_face_text)

        # Reset the grid frame
        self.grid_frame.destroy()
        self.grid_frame = tk.Frame(self.control_frame)
        self.grid_frame.pack(pady=10)


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
