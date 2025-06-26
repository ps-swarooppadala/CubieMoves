import subprocess
import json

def main():
    # Path to the Rubik's Cube capturing application
    capture_app_path = "capture_cube_ui2.py"

    # Run the Rubik's Cube application as a subprocess
    print("Launching Rubik's Cube capturing application...")
    result = subprocess.run(["python", capture_app_path], check=True)

    # Check if the subprocess completed successfully
    if result.returncode == 0:
        print("Rubik's Cube application completed. Reading JSON file...")

        # Read the output JSON file
        json_file = "RubiksCubeState.json"
        try:
            with open(json_file, "r") as f:
                cube_state = json.load(f)
            print("Cube state:", cube_state)

            # Example logic: Print the top face
            top_face = cube_state.get("top")
            print("Top face colors:", top_face)

            # Further processing...
            if top_face == [["W", "W", "W"], ["W", "W", "W"], ["W", "W", "W"]]:
                print("Top face is all white!")
            else:
                print("Top face needs adjustment.")

        except FileNotFoundError:
            print(f"Error: JSON file '{json_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{json_file}'.")
    else:
        print("Rubik's Cube application encountered an error.")

if __name__ == "__main__":
    main()
