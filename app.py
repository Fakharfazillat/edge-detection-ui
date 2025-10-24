"""
Interactive Edge Detection UI (Streamlit + OpenCV)
Meets CS-4218 A1 requirements:
 - Upload image (JPG/PNG/BMP)
 - Side-by-side Input / Output
 - Algorithm select: Sobel / Laplacian / Canny
 - Live parameter controls (sliders / selects)
 - Real-time update (or Apply button)
Author: Fakhar Fazillat
"""

from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Edge Detection UI", layout="wide")

# --- Helpers -----------------------------------------------------------------
def to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def ensure_odd(x):
    x = int(x)
    return x if x % 2 == 1 else x + 1

def apply_sobel(img_gray, ksize, direction):
    k = ensure_odd(ksize)
    # Use cv2.Sobel with cv2.CV_64F, then convert magnitude appropriately
    dx = cv2.Sobel(img_gray, cv2.CV_64F, 1 if direction in ("X", "Both") else 0, 0 if direction == "Y" else 1, ksize=k) if direction != "Y" else cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k)
    if direction == "X":
        res = dx
    elif direction == "Y":
        res = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k)
    else:  # Both
        gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k)
        res = np.sqrt(gx**2 + gy**2)
    # Normalize to 0-255 and convert to uint8
    res = np.absolute(res)
    res = np.uint8(255 * (res / (res.max() + 1e-8)))
    return res

def apply_laplacian(img_gray, ksize):
    k = ensure_odd(ksize)
    res = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=k)
    res = np.absolute(res)
    res = np.uint8(255 * (res / (res.max() + 1e-8)))
    return res

def apply_canny(img_gray, low, high, ksize_blur, sigma):
    # Gaussian blur before Canny
    k = ensure_odd(ksize_blur)
    if k > 1:
        img_blur = cv2.GaussianBlur(img_gray, (k, k), sigma)
    else:
        img_blur = img_gray
    edges = cv2.Canny(img_blur, threshold1=low, threshold2=high)
    return edges

def pil_from_cv2(img_cv2):
    # Convert BGR (cv2) or grayscale to RGB PIL Image
    if len(img_cv2.shape) == 2:
        return Image.fromarray(img_cv2).convert("RGB")
    else:
        # assume BGR
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

# --- UI ----------------------------------------------------------------------
st.title("Interactive Edge Detection UI")
st.markdown("Upload an image and experiment with Sobel, Laplacian, and Canny edge detectors.")

# Sidebar controls
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload image (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp"])

algo = st.sidebar.radio("Algorithm", ["Sobel", "Laplacian", "Canny"], index=0)

# Shared controls
real_time = st.sidebar.checkbox("Update in real time (toggle)", value=True)
apply_button = st.sidebar.button("Apply") if not real_time else None

# Algorithm specific
if algo == "Sobel":
    st.sidebar.subheader("Sobel parameters")
    sobel_ksize = st.sidebar.slider("Kernel size (odd)", min_value=1, max_value=31, value=3, step=2)
    sobel_dir = st.sidebar.selectbox("Gradient direction", ["Both", "X", "Y"])
elif algo == "Laplacian":
    st.sidebar.subheader("Laplacian parameters")
    lap_ksize = st.sidebar.slider("Kernel size (odd)", min_value=1, max_value=31, value=3, step=2)
elif algo == "Canny":
    st.sidebar.subheader("Canny parameters")
    canny_low = st.sidebar.slider("Lower threshold", 0, 500, 50)
    canny_high = st.sidebar.slider("Upper threshold", 0, 500, 150)
    blur_ksize = st.sidebar.slider("Gaussian kernel (odd)", min_value=1, max_value=31, value=5, step=2)
    blur_sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 1.0, step=0.1)

# Title toggle & contrast
st.sidebar.subheader("Display")
show_titles = st.sidebar.checkbox("Show titles above displays", value=True)
contrast = st.sidebar.slider("Output intensity multiplier", 1.0, 4.0, 1.0, step=0.1)

# Layout: two columns (side-by-side)
col1, col2 = st.columns([1,1])

with col1:
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Placeholder_view_vector.svg", caption="No image", use_column_width=True)
    else:
        # read image as OpenCV BGR
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_display = pil_from_cv2(img_bgr)
        if show_titles: st.subheader("Input (Original)")
        st.image(img_display, use_column_width=True)

with col2:
    if uploaded_file is None:
        st.empty()
    else:
        if (real_time) or (apply_button):
            # Process
            img_gray = to_gray(img_bgr)
            if algo == "Sobel":
                out = apply_sobel(img_gray, sobel_ksize, sobel_dir)
            elif algo == "Laplacian":
                out = apply_laplacian(img_gray, lap_ksize)
            else:
                out = apply_canny(img_gray, canny_low, canny_high, blur_ksize, blur_sigma)

            # intensity adjust
            out = np.clip(out * contrast, 0, 255).astype(np.uint8)

            if len(out.shape) == 2:
                out_pil = Image.fromarray(out).convert("RGB")
            else:
                out_pil = pil_from_cv2(out)

            if show_titles: st.subheader(f"Output ({algo})")
            st.image(out_pil, use_column_width=True)

            # Optional: allow download of output
            buf = cv2.imencode(".png", cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button("Download output image", data=buf, file_name="edges.png", mime="image/png")
        else:
            # waiting state
            st.info("Change parameters or toggle 'Update in real time' to apply the algorithm.")

# Small help text
st.markdown("""
**Notes:**  
- Canny typically needs the lower and upper thresholds tuned; use Gaussian blur to reduce noise.  
- Sobel kernel sizes must be odd; larger kernels detect broader gradients.  
- Laplacian kernel size influences the sensitivity (noise ↑ with large kernels).
""")
