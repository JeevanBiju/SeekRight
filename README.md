# 🛣️ SeekRight – Road Marking Corner Detector

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://seekright.onrender.com/)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

🔗 **Live Web Application**: [https://seekright.onrender.com/](https://seekright.onrender.com/)

An automated computer vision tool to detect the **4 corner points** and compute the **polygon area** of painted road markings (speed-breakers and rumble strips).

Supports detection of **Yellow** and **White** road markings, with an **Auto** detection mode that scans and identifies the best marking set.

---

## 🚀 Features

- **Interactive Web App**: Built with Gradio.
- **Instant Sample Selection**: Click on preset reference images to auto-populate the input and set recommended colors.
- **Custom Image Upload**: Upload any road marking image (.png, .jpg, .jpeg).
- **Color Filter**: Choose `Auto`, `White`, or `Yellow`.
- **Detailed Output**: Visualizes the 4 corner points, bounding polygon, and calculates exact pixel area.

---

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JeevanBiju/SeekRight.git
   cd SeekRight
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```
   Open your browser at `http://localhost:7860`.

4. **Temporary Public Sharing**:
   ```bash
   python app.py --share
   ```

---

## 📁 Repository Structure

```
├── app.py              # Gradio web frontend
├── final.py            # Core computer vision detection pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration for cloud deployment
├── samples/            # Preset reference images
│   ├── img1.png
│   ├── img2.jpeg
│   └── img3.jpeg
└── README.md           # Documentation
```
