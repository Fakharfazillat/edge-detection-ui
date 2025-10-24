# Interactive Edge Detection Application 🧠
**Course:** Computer Vision (CS-4218)  
**Name:** Fakhar Fazillat  
**Roll Number:** 0112-BSCS-22  
**Instructor:** Ma'am Arbish Akram  

---

## 📘 Project Overview
This project provides an **interactive web-based application** to visualize and experiment with different edge detection algorithms — **Sobel**, **Laplacian**, and **Canny** — using **Python**, **OpenCV**, and **Streamlit**.

The application allows users to upload an image, adjust parameters dynamically, and observe how edge detection results change in real-time.

---

## 🧰 Tools and Libraries
- **Python 3.11**  
- **Streamlit** – for the web interface  
- **OpenCV** – for image processing  
- **NumPy** – for numerical and array operations  
- **Pillow (PIL)** – for image handling  

---

## 🌟 Features
✅ Upload an image (JPG / PNG / BMP)  
✅ Choose between **Sobel**, **Laplacian**, and **Canny** algorithms  
✅ Adjust kernel size, thresholds, and gradient directions  
✅ Real-time parameter updates  
✅ Two panels — **Input (Original)** and **Output (Edge Detected)**  
✅ User-friendly interface with dynamic sliders and download option  

---

## 🖼️ Application Screenshots
Twelve screenshots are included in the **`screenshots`** folder.  
They show **four different images**, each tested with different parameters using:  
- **Sobel filter**  
- **Laplacian filter**  
- **Canny edge detector**

Each screenshot clearly demonstrates how varying parameters affect edge detection results.

---

## 🧩 How to Run the App
1. **Clone or download** this repository.  
2. Open the project folder in **VS Code** or **Command Prompt**.  
3. Run the following commands one by one:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # (for Windows)
   pip install -r requirements.txt
   streamlit run app.py
