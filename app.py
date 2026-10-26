import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim


# =====================================================
# PART 1: IMAGE PROCESSING FUNCTIONS
# =====================================================

def enhance(img, brightness=0, contrast=1.0, saturation=1.0, sharpen=0.0):
    out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2)
        out = cv2.addWeighted(out, 1 + sharpen, blur, -sharpen, 0)
    return out


def build_palette(img_bgr, n_colors):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) > 20000:
        idx = np.random.RandomState(0).choice(len(lab), 20000, replace=False)
        sample = lab[idx]
    else:
        sample = lab
    n_colors = max(1, min(int(n_colors), len(sample)))
    km = KMeans(n_clusters=n_colors, n_init=3, random_state=0).fit(sample)
    return km.cluster_centers_.astype(np.float32)


def palette_to_bgr(palette_lab):
    px = np.clip(palette_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    return cv2.cvtColor(px, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def nearest_idx(lab_colors, palette_lab):
    d = ((lab_colors[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def palette_image(palette_bgr):
    k = len(palette_bgr)
    sw = np.zeros((60, k * 60, 3), np.uint8)
    for i, c in enumerate(palette_bgr):
        sw[:, i * 60:(i + 1) * 60] = c
    return cv2.cvtColor(sw, cv2.COLOR_BGR2RGB)


def edge_map(img_bgr, low=80, high=160):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, low, high)


def quadtree_cells(img_bgr, edges, max_tile, min_tile, var_thresh, edge_aware):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    cells = []

    def split(x, y, w, h):
        if w <= 0 or h <= 0:
            return
        block = gray[y:y + h, x:x + w]
        too_big = max(w, h) > max_tile
        can_split = w >= 2 * min_tile and h >= 2 * min_tile
        busy = block.std() > var_thresh
        if edge_aware:
            busy = busy or (edges[y:y + h, x:x + w] > 0).mean() > 0.12
        if can_split and (too_big or busy):
            w2, h2 = w // 2, h // 2
            split(x, y, w2, h2)
            split(x + w2, y, w - w2, h2)
            split(x, y + h2, w2, h - h2)
            split(x + w2, y + h2, w - w2, h - h2)
        else:
            cells.append((x, y, w, h))

    for y in range(0, H, max_tile):
        for x in range(0, W, max_tile):
            split(x, y, min(max_tile, W - x), min(max_tile, H - y))
    return cells


def uniform_cells(shape, tile):
    H, W = shape[:2]
    return [(x, y, min(tile, W - x), min(tile, H - y))
            for y in range(0, H, tile) for x in range(0, W, tile)]


def draw_tile(canvas, x, y, w, h, color, shape, gap, rng, jitter):
    g = min(int(gap), max(0, min(w, h) // 6))
    x0, y0 = x + g // 2, y + g // 2
    x1, y1 = x + w - (g - g // 2), y + h - (g - g // 2)
    if x1 <= x0 or y1 <= y0:
        return
    if jitter > 0:
        j = int(jitter * min(w, h))
        if j > 0:
            x0 += rng.randint(0, j + 1)
            y0 += rng.randint(0, j + 1)
            x1 -= rng.randint(0, j + 1)
            y1 -= rng.randint(0, j + 1)
            if x1 <= x0 or y1 <= y0:
                return
        color = tuple(int(np.clip(c + rng.randint(-8, 9), 0, 255)) for c in color)

    if shape == "square":
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, -1)
    elif shape == "circle":
        c = ((x0 + x1) // 2, (y0 + y1) // 2)
        cv2.ellipse(canvas, c, (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
                    0, 0, 360, color, -1)
    elif shape == "diamond":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        pts = np.array([[cx, y0], [x1 - 1, cy], [cx, y1 - 1], [x0, cy]], np.int32)
        cv2.fillPoly(canvas, [pts], color)


def voronoi_mosaic(img_bgr, tile_size, palette_lab, edges, edge_aware,
                   gap, grout, seed=0):
    rng = np.random.RandomState(seed)
    H, W = img_bgr.shape[:2]
    pts = [(x + rng.uniform(0, tile_size), y + rng.uniform(0, tile_size))
           for y in range(0, H, tile_size) for x in range(0, W, tile_size)]
    if edge_aware:
        ys, xs = np.nonzero(edges)
        if len(xs) > 0:
            n_extra = min(len(xs), int(0.7 * len(pts)))
            if n_extra:
                pick = rng.choice(len(xs), n_extra, replace=False)
                pts += list(zip(xs[pick].astype(float), ys[pick].astype(float)))
    pts = np.array(pts)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    gy, gx = np.mgrid[0:H, 0:W]
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1)
    labels = cKDTree(pts).query(coords)[1]
    n = len(pts)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    counts[counts == 0] = 1
    means = np.stack([
        np.bincount(labels, weights=lab[:, c], minlength=n) / counts
        for c in range(3)
    ], axis=1).astype(np.float32)

    pal_bgr = palette_to_bgr(palette_lab)
    cell_color = pal_bgr[nearest_idx(means, palette_lab)]
    canvas = cell_color[labels].reshape(H, W, 3).astype(np.uint8)

    lab_map = labels.reshape(H, W)
    border = np.zeros((H, W), np.uint8)
    border[:, :-1] |= (lab_map[:, :-1] != lab_map[:, 1:]).astype(np.uint8)
    border[:-1, :] |= (lab_map[:-1, :] != lab_map[1:, :]).astype(np.uint8)
    if gap > 0:
        k = np.ones((int(gap), int(gap)), np.uint8)
        border = cv2.dilate(border, k)
    canvas[border > 0] = grout

    layout = img_bgr.copy()
    layout[border > 0] = (0, 0, 0)
    return canvas, int(len(np.unique(labels))), layout


def photo_mosaic(img_bgr, tile_imgs, tile_size, tint=0.2):
    """Build a photo mosaic from one or more user-uploaded tile images."""
    if not tile_imgs:
        raise ValueError("Upload at least one image in the Photo tiles section.")

    means = np.array([
        cv2.cvtColor(t, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
        for t in tile_imgs
    ], np.float32)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    out = np.zeros_like(img_bgr)
    cells = uniform_cells(img_bgr.shape, tile_size)

    for (x, y, w, h) in cells:
        avg_lab = lab[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0)
        best = int(np.argmin(((means - avg_lab) ** 2).sum(axis=1)))
        t = cv2.resize(tile_imgs[best], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        avg_bgr = img_bgr[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0)
        t = (1 - tint) * t + tint * avg_bgr
        out[y:y + h, x:x + w] = np.clip(t, 0, 255).astype(np.uint8)

    layout = img_bgr.copy()
    for (x, y, w, h) in cells:
        cv2.rectangle(layout, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 1)
    return out, len(cells), layout


def make_mosaic(img_bgr, tile_size=16, n_colors=16, shape="square",
                style="flat", adaptive=False, edge_aware=False,
                min_tile=6, var_thresh=18.0, gap=2, palette=None, seed=0):
    rng = np.random.RandomState(seed)
    edges = edge_map(img_bgr)
    if palette is None:
        palette = build_palette(img_bgr, n_colors)

    grout = (110, 110, 110)
    jitter = 0.0
    if style == "ancient":
        grout = (150, 170, 190)
        jitter = 0.18

    if shape == "stained glass":
        return voronoi_mosaic(img_bgr, tile_size, palette, edges, edge_aware,
                              max(gap, 1), (30, 30, 30), seed)

    if adaptive or edge_aware:
        cells = quadtree_cells(img_bgr, edges, tile_size, min_tile,
                               var_thresh, edge_aware)
    else:
        cells = uniform_cells(img_bgr.shape, tile_size)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pal_bgr = palette_to_bgr(palette)
    canvas = np.full_like(img_bgr, grout)
    layout = img_bgr.copy()

    for (x, y, w, h) in cells:
        avg = lab[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0)
        idx = nearest_idx(avg.reshape(1, 3).astype(np.float32), palette)[0]
        color = tuple(int(v) for v in pal_bgr[idx])
        draw_tile(canvas, x, y, w, h, color, shape, gap, rng, jitter)
        cv2.rectangle(layout, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 1)
    return canvas, len(cells), layout


def quality(original, mosaic):
    a = original.astype(np.float64)
    b = mosaic.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    s = float(ssim(original, mosaic, channel_axis=2, data_range=255))
    return {"MSE": round(mse, 2), "PSNR (dB)": round(psnr, 2), "SSIM": round(s, 4)}


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode(file_obj):
    data = np.frombuffer(file_obj.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# =====================================================
# LOOK AND FEEL
# =====================================================
def apply_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    .stApp {background: linear-gradient(135deg, #fff7fb 0%, #f5f0ff 50%, #eafaff 100%); font-family:'Poppins',sans-serif;}
    h1,h2,h3,h4,p,label,button {font-family:'Poppins',sans-serif;}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#fff0f7 0%,#f0eaff 55%,#e7faff 100%);border-right:2px solid white;}
    [data-testid="stSidebar"] h2 {color:#7147bd;font-size:1.1rem;font-weight:700;border-bottom:2px solid #e8b4e8;padding-bottom:7px;margin-top:14px;}
    .hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#d94db6 0%,#8b65e8 52%,#38bfd0 100%);border-radius:28px;padding:38px 42px;margin:18px 0 30px 0;box-shadow:0 15px 40px rgba(115,78,180,.22);}
    .hero::after {content:"";position:absolute;width:230px;height:230px;right:-75px;top:-100px;border-radius:50%;background:rgba(255,255,255,.10);pointer-events:none;}
    .hero h1 {color:white;margin:0 0 12px 0;font-size:2.6rem;font-weight:800;letter-spacing:-1px;}
    .hero p {color:white;font-size:1.08rem;line-height:1.8;margin:0;max-width:720px;}
    .hero .eyebrow {display:inline-block;color:white;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:6px 13px;margin-bottom:16px;font-size:.78rem;font-weight:600;letter-spacing:1px;}
    .hero .subtitle {margin-top:12px;opacity:.94;}
    [data-testid="stImage"] img {border-radius:18px;border:3px solid white;box-shadow:0 8px 24px rgba(90,60,160,.13);}
    [data-testid="stImageCaption"] {color:#7147bd;font-weight:600;text-align:center;}
    [data-testid="stMetric"] {background:rgba(255,255,255,.92);border-radius:18px;padding:16px 18px;box-shadow:0 6px 20px rgba(90,60,160,.09);border-left:5px solid #bd8be8;}
    [data-testid="stMetricValue"] {color:#7147bd;font-weight:700;}
    .stButton>button,.stDownloadButton>button {background:linear-gradient(100deg,#d94db6,#8263e6);color:white;border:none;border-radius:999px;padding:.65rem 1.6rem;font-weight:600;box-shadow:0 6px 18px rgba(130,99,230,.22);transition:all .2s ease;}
    .stButton>button:hover,.stDownloadButton>button:hover {transform:translateY(-2px);color:white;box-shadow:0 10px 25px rgba(130,99,230,.3);}
    .stTabs [data-baseweb="tab-list"] {gap:10px;border-bottom:none;}
    .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {display:none;}
    .stTabs button[role="tab"] {background:white;border-radius:999px;padding:8px 20px;box-shadow:0 3px 12px rgba(90,60,160,.08);}
    .stTabs button[role="tab"][aria-selected="true"] {background:linear-gradient(100deg,#d94db6,#8263e6);}
    .stTabs button[role="tab"][aria-selected="true"] p {color:white;font-weight:700;}
    [data-testid="stFileUploaderDropzone"] {background:rgba(255,255,255,.85);border:2px dashed #d69adf;border-radius:16px;}
    </style>
    <div class="hero">
      <div class="eyebrow">CREATIVE ART</div>
      <h1>🎨 Mosaic Studio</h1>
      <p>Transform your favourite photographs into beautiful, colourful mosaic artwork.</p>
      <p class="subtitle">Explore artistic tile styles, experiment with colours, and create a masterpiece of your own.</p>
    </div>
    """, unsafe_allow_html=True)


PRESETS = {
    "Custom (use sliders)": {},
    "Roman floor": dict(tile=12, colors=10, shape="square", style="ancient", adaptive=False, edge=False, gap=2),
    "Pixel art": dict(tile=14, colors=12, shape="square", style="flat", adaptive=False, edge=False, gap=0),
    "Pop-art dots": dict(tile=12, colors=8, shape="circle", style="flat", adaptive=False, edge=False, gap=1),
    "Stained glass": dict(tile=18, colors=14, shape="stained glass", style="flat", adaptive=False, edge=True, gap=2),
    "Smart detail (best quality)": dict(tile=16, colors=24, shape="square", style="flat", adaptive=True, edge=True, gap=1),
}


# =====================================================
# PART 2: STREAMLIT APPLICATION
# =====================================================
def main():
    st.set_page_config(page_title="Mosaic Studio", page_icon="🎨", layout="wide")
    apply_style()
    sb = st.sidebar

    sb.header("1. Photo")
    uploaded = sb.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="main_photo")

    sb.header("2. Mosaic type")
    mode = sb.radio("Mode", ["Color tiles", "Photo tiles (use my own images)"])
    preset_name = sb.selectbox("Quick preset", list(PRESETS.keys()))

    # Show tile-image uploader directly below the mode selector when Photo tiles is selected.
    tile_files = []
    tint = 0.2
    if mode.startswith("Photo"):
        sb.header("Photo tiles")
        sb.caption("Upload one or more small images used as the mosaic tiles.")
        tile_files = sb.file_uploader(
            "Upload tile image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="photo_tile_images",
            help="Select one or more images. To add more, open the picker again and select the complete set you want to use."
        )
        if tile_files:
            sb.success(f"{len(tile_files)} tile image(s) selected.")
            with sb.expander("View selected tile images"):
                for i, f in enumerate(tile_files, start=1):
                    sb.write(f"{i}. {f.name}")
        tint = sb.slider("Tint strength", 0.0, 0.6, 0.2)

    sb.header("3. Settings")
    tile = sb.slider("Tile size (pixels)", 6, 48, 16)
    colors = sb.slider("Number of colors", 4, 64, 16)
    shape = sb.selectbox("Tile shape", ["square", "circle", "diamond", "stained glass"])
    style = sb.selectbox("Mosaic style", ["flat", "ancient"])
    adaptive = sb.checkbox("Adaptive tile size", True)
    edge = sb.checkbox("Edge-aware tiles", True)
    gap = sb.slider("Gap between tiles", 0, 6, 1)

    sb.header("5. Improve photo first")
    bright = sb.slider("Brightness", -60, 60, 0)
    contrast = sb.slider("Contrast", 0.5, 2.0, 1.0)
    sat = sb.slider("Color strength", 0.0, 2.0, 1.0)
    sharp = sb.slider("Sharpen", 0.0, 2.0, 0.0)

    p = PRESETS[preset_name]
    if p:
        tile, colors, shape = p["tile"], p["colors"], p["shape"]
        style, adaptive, edge, gap = p["style"], p["adaptive"], p["edge"], p["gap"]
        sb.success("Preset is active. Choose 'Custom' to use the sliders.")

    if uploaded is None:
        st.info("Upload a main photo from the left panel to start.")
        return

    img = decode(uploaded)
    if img is None:
        st.error("The main photo could not be read. Please upload a valid JPG or PNG image.")
        return
    h, w = img.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img = enhance(img, bright, contrast, sat, sharp)

    palette = build_palette(img, colors)

    with st.spinner("Making your mosaic..."):
        if mode.startswith("Photo"):
            if not tile_files:
                st.warning("Upload at least one image in the separate Photo tiles section in the left panel.")
                return
            lib = [decode(f) for f in tile_files]
            lib = [t for t in lib if t is not None]
            if not lib:
                st.warning("The tile image(s) could not be read. Please upload valid JPG or PNG images.")
                return
            mosaic, n, layout = photo_mosaic(img, lib, tile, tint)
        else:
            mosaic, n, layout = make_mosaic(
                img, tile, colors, shape, style, adaptive, edge, gap=gap, palette=palette
            )
        q = quality(img, mosaic)

    tab1, tab2, tab3 = st.tabs(["Result", "How it works", "Compare tile sizes"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.image(to_rgb(img), caption="Original", use_container_width=True)
        c2.image(to_rgb(mosaic), caption=f"Mosaic ({n} tiles)", use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tiles used", n)
        m2.metric("MSE (lower is better)", q["MSE"])
        m3.metric("PSNR dB (higher is better)", q["PSNR (dB)"])
        m4.metric("SSIM (closer to 1 is better)", q["SSIM"])
        ok, buf = cv2.imencode(".png", mosaic)
        st.download_button("Download mosaic image", buf.tobytes(), "mosaic.png", "image/png")

    with tab2:
        st.subheader("The steps inside the program")
        a, b, c = st.columns(3)
        a.image(to_rgb(img), caption="Step 1: Improved input photo", use_container_width=True)
        b.image(edge_map(img), caption="Step 2: Edges found by Canny", use_container_width=True)
        c.image(to_rgb(layout), caption="Step 3: Tile layout", use_container_width=True)
        st.write("Step 4: The K-Means color palette. Each color tile uses the nearest palette color.")
        st.image(palette_image(palette_to_bgr(palette)))
        st.write("Step 5: MSE, PSNR and SSIM compare the mosaic with the original image.")

    with tab3:
        st.write("Compare the same settings using different tile sizes.")
        if mode.startswith("Photo"):
            st.info("The tile-size comparison is available in Color tiles mode.")
        elif st.button("Run comparison"):
            rows = []
            bar = st.progress(0)
            sizes = [8, 12, 16, 24, 32]
            for i, s in enumerate(sizes):
                m, cnt, _ = make_mosaic(img, s, colors, shape, style, adaptive, edge,
                                        gap=gap, palette=palette)
                qq = quality(img, m)
                rows.append({"Tile size": s, "Tiles": cnt, "MSE": qq["MSE"],
                             "PSNR (dB)": qq["PSNR (dB)"], "SSIM": qq["SSIM"]})
                bar.progress((i + 1) / len(sizes))
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.write("PSNR vs tile size")
            d1.line_chart(df.set_index("Tile size")["PSNR (dB)"])
            d2.write("SSIM vs tile size")
            d2.line_chart(df.set_index("Tile size")["SSIM"])
            st.download_button("Download table (CSV)", df.to_csv(index=False),
                               "comparison.csv", "text/csv")


if __name__ == "__main__":
    main()

