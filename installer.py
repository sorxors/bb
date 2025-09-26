import subprocess
import os
import sys

# This script installs the packages from requirements.txt

# Get the directory of the current script to find the requirements.txt file
script_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(script_dir, 'requirements.txt')

# Get the path to the python executable that is running this script
python_executable = sys.executable

print(f"--- Starting Requirement Installer ---")
print(f"Using Python from: {python_executable}")
print(f"Looking for requirements file at: {requirements_path}")

if not os.path.exists(requirements_path):
    print("\nERROR: requirements.txt was not found in the same directory as this script.")
else:
    print("\nFound requirements.txt. Starting installation...")
    try:
        # Run the pip install command
        subprocess.check_call([python_executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("\nSUCCESS: Installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: The installation process failed. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

print(f"--- Installer Finished ---")
