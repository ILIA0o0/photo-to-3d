import streamlit as st
import torch
import numpy as np
from PIL import Image
import trimesh
import base64

st.set_page_config(page_title="تبدیل عکس به مدل سه‌بعدی", layout="centered")

@st.cache_resource
def load_model():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return midas, transforms.small_transform

def estimate_depth(image, midas, transform):
    img = np.array(image.convert("RGB"))
    input_batch = transform(img)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return depth_map, img

def build_mesh(depth_map, color_img, downsample=4, depth_scale=50):
    depth_small = depth_map[::downsample, ::downsample]
    color_small = color_img[::downsample, ::downsample]
    hs, ws = depth_small.shape

    xs, ys = np.meshgrid(np.arange(ws), np.arange(hs))
    zs = depth_small * depth_scale

    vertices = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float64)
    colors = color_small.reshape(-1, 3)

    faces = []
    for i in range(hs - 1):
        for j in range(ws - 1):
            idx = i * ws + j
            faces.append([idx, idx + 1, idx + ws])
            faces.append([idx + 1, idx + ws + 1, idx + ws])
    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    return mesh

st.title("🖼️ تبدیل عکس به مدل سه‌بعدی رنگی")
st.write("یک عکس آپلود کن تا مدل سه‌بعدی رنگی از روی عمق تصویر ساخته بشه.")

uploaded_file = st.file_uploader("انتخاب عکس", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="عکس ورودی", use_column_width=True)

    if st.button("ساخت مدل سه‌بعدی"):
        with st.spinner("در حال پردازش... (چند لحظه صبر کن)"):
            midas, transform = load_model()
            depth_map, color_img = estimate_depth(image, midas, transform)
            mesh = build_mesh(depth_map, color_img)
            glb_bytes = trimesh.exchange.gltf.export_glb(mesh)

        st.success("مدل ساخته شد ✅")

        # نمایش زنده مدل سه‌بعدی در صفحه
        b64 = base64.b64encode(glb_bytes).decode("utf-8")
        viewer_html = f"""
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer src="data:model/gltf-binary;base64,{b64}"
            alt="3D model" auto-rotate camera-controls
            style="width:100%; height:500px; background:#222;">
        </model-viewer>
        """
        st.components.v1.html(viewer_html, height=520)

        st.download_button(
            label="⬇️ دانلود مدل سه‌بعدی (.glb)",
            data=glb_bytes,
            file_name="model.glb",
            mime="model/gltf-binary"
        )

        with st.expander("نمایش نقشه عمق"):
            st.image(depth_map, caption="Depth Map", use_column_width=True, clamp=True)
