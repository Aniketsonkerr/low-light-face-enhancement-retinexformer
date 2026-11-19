import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer


# ==========================================
# 1. Image Quality Metric Implementations
# ==========================================
def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (Eq. 4) on uint8 BGR images."""
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mse = np.mean((img1_f - img2_f) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) using standard 11x11 Gaussian window."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    ssim_channels = []
    for ch in range(img1_f.shape[2]):
        c1 = img1_f[:, :, ch]
        c2 = img2_f[:, :, ch]

        mu1 = cv2.GaussianBlur(c1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(c2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(c1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(c2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(c1 * c2, (11, 11), 1.5) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        ssim_channels.append(float(np.mean(ssim_map)))

    return float(np.mean(ssim_channels))


def calculate_no_reference_stats(img_bgr: np.ndarray) -> dict:
    """Computes supplementary no-reference metrics for unpaired datasets (Sec. IV-C)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean_brightness = float(np.mean(gray))
    rms_contrast = float(np.std(gray))

    # Shannon Entropy (information richness)
    hist = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).ravel()
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))

    return {
        "brightness": mean_brightness,
        "contrast": rms_contrast,
        "entropy": entropy
    }


# ==========================================
# 2. Baseline & Proposed Methods (Sec. IV-A)
# ==========================================
def apply_gamma_correction(img_bgr: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """Eq. (1): I_enhanced = I^gamma (0 < gamma < 1 brightens dark regions)."""
    norm = img_bgr.astype(np.float32) / 255.0
    corrected = np.power(norm, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_histogram_equalization(img_bgr: np.ndarray) -> np.ndarray:
    """Equalizes luminance channel in YCrCb space to preserve color chromaticity."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def load_retinexformer(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = RetinexFormer(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2]
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def apply_retinexformer(img_bgr: np.ndarray, model: torch.nn.Module, device: torch.device) -> tuple:
    """Runs Retinexformer with 16x16 dynamic padding and precise GPU timing."""
    crop_float = img_bgr.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(crop_float).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = img_tensor.shape[2], img_tensor.shape[3]
    multiple = 16
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_tensor = model(img_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    output_tensor = output_tensor[:, :, :h, :w]
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)

    elapsed_ms = (t_end - t_start) * 1000.0
    return output_img, elapsed_ms


# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================
def main():
    # Dataset paths (supports CustomFaces or any paired dataset like LOL-v1)
    input_dir = os.path.join(BASE_DIR, 'data', 'CustomFaces', 'Test', 'input')
    target_dir = os.path.join(BASE_DIR, 'data', 'CustomFaces', 'Test', 'target')
    save_dir = os.path.join(BASE_DIR, 'results', 'Evaluation_Comparisons')
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(BASE_DIR, 'pretrained_weights', 'LOL_v1.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    # Optional: set USE_FACE_CROP = True to evaluate on detected facial ROIs,
    # or False to evaluate on full images consistently across all methods.
    USE_FACE_CROP = False
    face_cascade = None
    if USE_FACE_CROP:
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)

    print("Loading Retinexformer checkpoint...")
    model = load_retinexformer(weights_path, device)

    # GPU Warmup pass (prevents first-run CUDA initialization spike in timing)
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Collect unique images (skip duplicate .png/.jpeg pairs if same stem exists)
    valid_exts = ('.png', '.jpg', '.jpeg')
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    seen_stems = set()
    image_files = []
    for f in all_files:
        stem = os.path.splitext(f)[0]
        if stem not in seen_stems:
            seen_stems.add(stem)
            image_files.append(f)

    if not image_files:
        print(f"[Error] No images found in {input_dir}")
        return

    metrics = {
        "Original": {"psnr": [], "ssim": [], "time": [], "nr": []},
        "Gamma correction": {"psnr": [], "ssim": [], "time": [], "nr": []},
        "Histogram equalization": {"psnr": [], "ssim": [], "time": [], "nr": []},
        "Retinexformer": {"psnr": [], "ssim": [], "time": [], "nr": []},
    }

    resolutions = []
    valid_pairs_count = 0

    for img_name in image_files:
        in_path = os.path.join(input_dir, img_name)
        img_in = cv2.imread(in_path)
        if img_in is None:
            continue

        # Check for valid, non-duplicated ground-truth target (Sec. III-D & IV-C)
        img_gt = None
        gt_path = os.path.join(target_dir, img_name)
        if os.path.exists(gt_path):
            candidate_gt = cv2.imread(gt_path)
            if candidate_gt is not None and candidate_gt.shape == img_in.shape:
                # Ensure target is NOT just a copied placeholder of the input
                if not np.array_equal(img_in, candidate_gt):
                    img_gt = candidate_gt
                    valid_pairs_count += 1

        # Optional ROI crop applied identically to input (and GT if paired)
        if USE_FACE_CROP and face_cascade is not None and not face_cascade.empty():
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                pad_x, pad_y = int(w * 0.2), int(h * 0.2)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(img_in.shape[1], x + w + pad_x), min(img_in.shape[0], y + h + pad_y)
                img_in = img_in[y1:y2, x1:x2]
                if img_gt is not None:
                    img_gt = img_gt[y1:y2, x1:x2]

        resolutions.append(f"{img_in.shape[1]}x{img_in.shape[0]}")

        # 0. Original
        metrics["Original"]["nr"].append(calculate_no_reference_stats(img_in))
        if img_gt is not None:
            metrics["Original"]["psnr"].append(calculate_psnr(img_in, img_gt))
            metrics["Original"]["ssim"].append(calculate_ssim(img_in, img_gt))

        # 1. Gamma Correction (CPU timing)
        t0 = time.perf_counter()
        out_gamma = apply_gamma_correction(img_in, gamma=0.45)
        t_gamma = (time.perf_counter() - t0) * 1000.0
        metrics["Gamma correction"]["time"].append(t_gamma)
        metrics["Gamma correction"]["nr"].append(calculate_no_reference_stats(out_gamma))
        if img_gt is not None:
            metrics["Gamma correction"]["psnr"].append(calculate_psnr(out_gamma, img_gt))
            metrics["Gamma correction"]["ssim"].append(calculate_ssim(out_gamma, img_gt))

        # 2. Histogram Equalization (CPU timing)
        t0 = time.perf_counter()
        out_he = apply_histogram_equalization(img_in)
        t_he = (time.perf_counter() - t0) * 1000.0
        metrics["Histogram equalization"]["time"].append(t_he)
        metrics["Histogram equalization"]["nr"].append(calculate_no_reference_stats(out_he))
        if img_gt is not None:
            metrics["Histogram equalization"]["psnr"].append(calculate_psnr(out_he, img_gt))
            metrics["Histogram equalization"]["ssim"].append(calculate_ssim(out_he, img_gt))

        # 3. Retinexformer (GPU/CPU synchronized timing)
        out_rf, t_rf = apply_retinexformer(img_in, model, device)
        metrics["Retinexformer"]["time"].append(t_rf)
        metrics["Retinexformer"]["nr"].append(calculate_no_reference_stats(out_rf))
        if img_gt is not None:
            metrics["Retinexformer"]["psnr"].append(calculate_psnr(out_rf, img_gt))
            metrics["Retinexformer"]["ssim"].append(calculate_ssim(out_rf, img_gt))

        # Save side-by-side comparison strip for Visual Comparison (Sec. V)
        strip = np.hstack([img_in, out_gamma, out_he, out_rf])
        cv2.imwrite(os.path.join(save_dir, f"compare_{os.path.splitext(img_name)[0]}.png"), strip)

    # ==========================================
    # 4. Print Reproducibility & Table I Results
    # ==========================================
    print("\n" + "=" * 68)
    print("EXPERIMENTAL SETUP & REPRODUCIBILITY REPORT (Sec. IV-B & V-A)")
    print("=" * 68)
    print(f"Hardware Accelerator : {gpu_name}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"Checkpoint           : pretrained_weights/LOL_v1.pth")
    print(f"Images Evaluated     : {len(image_files)} (Valid GT pairs: {valid_pairs_count})")
    print(f"Image Resolutions    : {', '.join(resolutions)}")
    print(f"Batch Size           : 1 (excludes disk I/O)")
    print(f"Comparison Strips    : {save_dir}")

    print("\n" + "=" * 68)
    print("TABLE I: ENHANCEMENT EVALUATION RESULTS (Markdown)")
    print("=" * 68)
    print("| Method                 | PSNR (dB) | SSIM   | Time (ms) |")
    print("|------------------------|-----------|--------|-----------|")

    latex_rows = []
    for method in ["Gamma correction", "Histogram equalization", "Retinexformer"]:
        data = metrics[method]
        psnr_str = f"{np.mean(data['psnr']):.2f}" if data['psnr'] else "—"
        ssim_str = f"{np.mean(data['ssim']):.4f}" if data['ssim'] else "—"
        time_str = f"{np.mean(data['time']):.2f}" if data['time'] else "—"
        print(f"| {method:<22} | {psnr_str:<9} | {ssim_str:<6} | {time_str:<9} |")
        latex_rows.append(f"{method} & {psnr_str} & {ssim_str} & {time_str} \\\\")

    print("\n" + "=" * 68)
    print("SUPPLEMENTARY NO-REFERENCE QUALITY METRICS (Sec. IV-C)")
    print("=" * 68)
    print("| Method                 | Mean Brightness | RMS Contrast | Entropy (bits) |")
    print("|------------------------|-----------------|--------------|----------------|")
    for method in ["Original", "Gamma correction", "Histogram equalization", "Retinexformer"]:
        nr_list = metrics[method]["nr"]
        b_mean = np.mean([x["brightness"] for x in nr_list])
        c_mean = np.mean([x["contrast"] for x in nr_list])
        e_mean = np.mean([x["entropy"] for x in nr_list])
        print(f"| {method:<22} | {b_mean:<15.2f} | {c_mean:<12.2f} | {e_mean:<14.2f} |")

    print("\n" + "=" * 68)
    print("READY-TO-PASTE LATEX CODE FOR TABLE I")
    print("=" * 68)
    print("\\begin{table}[htbp]")
    print("\\caption{Enhancement Evaluation Results}")
    print("\\begin{center}")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{PSNR (dB)} & \\textbf{SSIM} & \\textbf{Time (ms)} \\\\")
    print("\\hline")
    for row in latex_rows:
        print(row)
        print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab1}")
    print("\\end{center}")
    print("\\end{table}\n")


if __name__ == '__main__':
    main()