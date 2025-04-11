# 🧠 Hair Analysis Tool

The Hair Analysis Tool is a computer vision-based application designed to analyze hair coverage within a specific region of the scalp, using hairline detection techniques. It provides visual and quantitative insights about hair density, which can be useful for research, cosmetic evaluations, or medical diagnostics.

---

## ✨ Features

- 🔍 **Hairline Detection** — Uses a robust API to detect the frontal hairline in input images.
- 📏 **Hair Coverage Calculation** — Measures hair coverage within the detected region, pixel-by-pixel.
- 📊 **Visualization** — Annotates and visualizes hairline boundaries and coverage areas on the original image.
- 📁 **Batch Support** — Process multiple images in one go (optional extension).

---

## 🧰 Tech Stack

- **Python** 🐍
- **OpenCV** for image processing
- **NumPy** for numerical operations
- Optional: Matplotlib / Streamlit for visualization or UI (if applicable)

---

## 🚀 How It Works

1. **Input**: The user uploads or provides an image of a scalp or head region.
2. **Hairline Detection**: The image is sent to a hairline detection API, which returns boundary coordinates.
3. **Coverage Analysis**: The tool evaluates pixel intensity or pattern within the region under the detected hairline.
4. **Output**: A report with metrics such as:
   - Hair coverage %
   - Visualization with bounding regions
   - Optional: CSV of data points or summary chart

---

## 📂 Folder Structure

```bash
hair-analysis-tool/
├── images/                # Input images
├── outputs/               # Output images + reports
├── src/                   # Source code
│   ├── hairline_api.py
│   ├── coverage_analysis.py
│   └── visualize.py
├── main.py                # Entry point script
├── requirements.txt       # Dependencies
└── README.md              # Project overview
```

---

## 🔧 Installation

```bash
git clone https://github.com/nazil-the-professor/hair-analysis-tool.git
cd hair-analysis-tool
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py --image path/to/image.jpg
```

Optional flags:
- `--batch path/to/folder/` — Process all images in a folder
- `--save` — Save annotated output images and report

---

## 📈 Example Output

![Example](outputs/sample_output.jpg)

> Detected hairline region with calculated hair coverage overlay

---

## ✍️ Author

**Nazil Sheikh**  
Junior Developer (Intern) @ Skyline Meridian  
GitHub: [@nazil-the-professor](https://github.com/nazil-the-professor)

