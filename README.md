
# 🚗 Vehicle Detection and Counting with YOLOv8 + SORT

This project detects and counts **cars** in a video using **YOLOv8** for detection and **SORT** for tracking. The system draws a custom line — and whenever a car crosses it, it increases the counter. A graphical overlay makes the UI look clean and modern.

---

## 🎥 Demo




---

## 📁 Project Structure

```

vehicle-counter/
├── main.py               # Core script for detection & counting
├── sort.py               # SORT tracking algorithm
├── mask.png              # Region of Interest (ROI) mask
├── graphics.png          # UI overlay image
├── Video/
│   └── 4.mp4             # Input video file
└── models/
|   └── yolo11n.pt        # Custom YOLOv8 trained model

````

---

## ⚙️ Setup Instructions

1. **Install required libraries**:
```bash
pip install ultralytics opencv-python numpy cvzone
````

2. **Run the script**:

```bash
python main.py
```

3. **What you need**:

   * A custom trained YOLOv8 model (`yolo11n.pt`)
   * A masked image (`mask.png`) to define the detection area
   * Input video in the `Video/` folder

---

## 🔍 How It Works

* Reads video frame-by-frame.
* Applies a mask to focus on a specific region.
* Detects objects using YOLOv8.
* Filters only **cars** with confidence > 0.3.
* Tracks objects using SORT and assigns unique IDs.
* If a car crosses a predefined line and hasn’t been counted before → count is incremented.
* Adds an animated counter UI overlay on the video.

---

## 🧪 Sample Output

* Tracked car boxes with IDs
* Counter updates when cars cross the line
* Optional debug printouts and region display

---

## 🚀 Future Ideas

* Add CSV logging
* Support for multiple object types
* Add live camera support
* Streamlit or web dashboard version

---

## 🙋‍♂️ About

Made for learning and demo purposes using YOLOv8 and OpenCV.
You can modify the model, line position, and ROI to fit other use cases too.

---

**Built with Python + OpenCV ❤️**

```
## 📄 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this code with proper attribution.
