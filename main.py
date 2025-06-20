import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load image
img = cv2.imread('lenna.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st.title("Gaussian Blur Demo with OpenCV")

# Sidebar controls
st.sidebar.header("Blur Parameters")
kernel_size = st.sidebar.slider("Kernel Size (odd)", min_value=3, max_value=31, step=2, value=7)
sigma = st.sidebar.slider("Sigma", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), sigma)

# Show images
st.subheader("Original Image")
st.image(img_rgb, channels="RGB", use_column_width=True)

st.subheader("Blurred Image")
st.image(blurred, channels="RGB", use_column_width=True)

# Show 1D Gaussian kernel
def gaussian_kernel_1d(ksize, sigma):
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    return ax, kernel

ax, kernel = gaussian_kernel_1d(kernel_size, sigma if sigma > 0 else 1)

# Calculate area under the discrete kernel (should be 1.0)
discrete_area = np.sum(kernel)

# Calculate area under the continuous Gaussian over the same range
half = (kernel_size - 1) // 2
# CDF at the edges
cdf_left = norm.cdf(-half - 0.5, loc=0, scale=sigma)
cdf_right = norm.cdf(half + 0.5, loc=0, scale=sigma)
continuous_area = cdf_right - cdf_left

# Percentage of area covered by the kernel
percentage = (continuous_area / discrete_area) * 100 if discrete_area > 0 else 0

st.subheader("1D Gaussian Kernel")
st.write(f"Kernel covers **{percentage:.2f}%** of the full Gaussian area.")

fig, ax1 = plt.subplots()

# Plot the actual kernel
ax1.plot(ax, kernel, marker='o', label='Kernel (normalized)')

# Draw vertical lines at kernel edges
edge_left = -((kernel_size - 1) // 2)
edge_right = (kernel_size - 1) // 2
ax1.axvline(edge_left, color='red', linestyle=':', label='Kernel edges')
ax1.axvline(edge_right, color='red', linestyle=':')

ax1.set_title(f"1D Gaussian Kernel (size={kernel_size}, sigma={sigma})")
ax1.legend()
st.pyplot(fig)