Michelson Interferometer Fringe Analysis 🌈
This repository contains Python scripts and an interactive Jupyter notebook for offline analysis and real-time processing of interference fringes from a Michelson interferometer setup.

The project demonstrates precise measurement and visualization of concentric interference fringes using computer vision and image processing techniques.

🧪 Project Overview
This project is intended for physicists, engineers, educators, and students interested in optics, interference phenomena, and experimental automation. It allows you to:

Capture and analyze fringe images using a Raspberry Pi Camera 2.

Perform real-time fringe detection with OpenCV.

Measure fringe radii and spacing accurately.

Conduct offline analysis using an interactive Jupyter notebook.

Quickly build and calibrate your experimental setup.

📂 Repository Structure
arduino
Copy
Edit
michelson-interferometer/
├── live_analysis/
│   └── fringe_detection_rpi.py
├── offline_analysis/
│   └── fringe_analysis_notebook.ipynb
├── images/
│   └── sample_fringe.jpg (replace with your image)
└── README.md
⚙️ Installation & Setup
🔹 Requirements
Python 3.x

OpenCV

NumPy

Matplotlib

Jupyter Notebook

ipywidgets

Picamera2 (for live analysis)

🔹 Installation Steps
Clone the repository and install dependencies:

bash
Copy
Edit
git clone https://ttps://github.com/evolveer/rpiMichelsonInterferometer.git

pip install opencv-python numpy matplotlib ipywidgets picamera2
jupyter nbextension enable --py widgetsnbextension
🖥️ Live Real-time Analysis (Raspberry Pi Camera)
Run the live analysis script directly on your Raspberry Pi to visualize real-time fringe detection:

bash
Copy
Edit
cd live_analysis
python fringe_detection_rpi.py
Press q to exit the real-time analysis viewer.

📊 Offline Fringe Analysis (Jupyter Notebook)
Use the provided Jupyter Notebook (fringe_analysis_notebook.ipynb) for offline fringe analysis, measurement, and calibration:


Load your own fringe images.

Interactively detect and measure fringe radii.

Accurately compute fringe spacing and widths.

Customize detection parameters easily.

📐 Calibration & Measurement
Ensure accurate results by calibrating the pixel-to-mm conversion factor (scale_mm_per_px) based on your experimental setup:

Measure a known distance or object in your image.

Calculate scale_mm_per_px as:


 
Replace this value in the notebook or script for precise real-world measurements.

🎯 Use Cases & Applications
Educational demonstrations in physics classes.

Research-grade interferometry experiments.

Low-cost optical instrumentation setups.

Student-led interdisciplinary projects.

🤝 Contributing
Feel free to contribute by opening issues, suggesting improvements, or submitting pull requests.

