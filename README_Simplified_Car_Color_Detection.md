
# Car Exterior and Interior Color Detection

This project uses Detectron2 and KMeans clustering to identify the dominant exterior and interior colors of a car in an image.

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA (Optional)**: If you want to use GPU acceleration, ensure CUDA 11.6 or a compatible version is installed.

## Setup Instructions

1. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv car_color_env
   source car_color_env/bin/activate  # For Linux/macOS
   car_color_env\Scripts\activate   # For Windows
   ```

2. **Install Required Libraries**

   Update pip and install dependencies:

   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio
   pip install opencv-python-headless matplotlib scikit-learn
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.10/index.html
   ```

3. **Run the Script**

   - Update the `image_path` in `solution1.py` with the path to your image.
   - Run the script:

   ```bash
   python solution1.py
   ```

## Notes

- Ensure you have all dependencies installed in the virtual environment.
- GPU acceleration is recommended for faster processing.

