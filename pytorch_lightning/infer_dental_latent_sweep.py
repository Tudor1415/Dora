#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from torch_cluster import fps

from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SHARP_DIR = REPO_ROOT / "sharp_edge_sampling"


def parse_int_csv(value: str) -> List[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one integer value.")
    return [int(x) for x in items]


def parse_float_csv(value: str) -> List[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected at least one float value.")
    return [float(x) for x in items]


def sanitize_sigma(sigma: float) -> str:
    return f"{sigma:.5f}".rstrip("0").rstrip(".").replace(".", "p")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_obj_files(data_root: Path) -> List[Path]:
    return sorted(data_root.rglob("*.obj"))


def pick_obj_path(obj_paths: List[Path], seed: int) -> Path:
    rng = random.Random(seed)
    idx = rng.randrange(len(obj_paths))
    return obj_paths[idx]


def expected_npz_path(precomp_root: Path, obj_path: Path) -> Path:
    return precomp_root / obj_path.parent.name / f"{obj_path.stem}.npz"


def ensure_npz_sample(
    obj_path: Path,
    npz_path: Path,
    run_dir: Path,
    point_number: int,
    angle_threshold: int,
    force_resample: bool,
) -> None:
    if npz_path.exists() and not force_resample:
        print(f"[INFO] Using existing sample: {npz_path}")
        return

    preprocess_dir = run_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    sample_root = npz_path.parent.parent
    sharp_point_path = sample_root / "sharp_point_ply"
    sample_root.mkdir(parents=True, exist_ok=True)
    sharp_point_path.mkdir(parents=True, exist_ok=True)

    selected_json = preprocess_dir / "selected_obj.json"
    with selected_json.open("w", encoding="utf-8") as f:
        json.dump([obj_path.as_posix()], f, indent=2)

    cmd = [
        sys.executable,
        str(SHARP_DIR / "sharp_sample.py"),
        "--json_file_path",
        str(selected_json),
        "--point_number",
        str(point_number),
        "--angle_threshold",
        str(angle_threshold),
        "--sharp_point_path",
        str(sharp_point_path),
        "--sample_path",
        str(sample_root),
    ]
    print(f"[INFO] Generating NPZ from OBJ via sharp sampling: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SHARP_DIR), check=True)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Expected sampled NPZ was not created: {npz_path}"
        )


def load_scan_surfaces(npz_path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)

    coarse = data["fps_coarse_surface"]
    sharp = data["fps_sharp_surface"]

    if coarse.ndim == 3:
        coarse = coarse[:, 0, :]
    if sharp.ndim == 3:
        sharp = sharp[:, 0, :]

    coarse = np.nan_to_num(coarse, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    sharp = np.nan_to_num(sharp, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    coarse_t = torch.from_numpy(coarse).unsqueeze(0).to(device)
    sharp_t = torch.from_numpy(sharp).unsqueeze(0).to(device)
    return coarse_t, sharp_t


def _sample_rows(rng: np.random.Generator, arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == 0:
        raise ValueError("Cannot sample from empty array.")
    replace = arr.shape[0] < n
    idx = rng.choice(arr.shape[0], size=n, replace=replace)
    return arr[idx]


def build_supervision_batch(
    npz_path: Path,
    n_supervision: List[int],
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    data = np.load(npz_path)
    sharp_near = data["sharp_near_surface"]
    rand_points = data["rand_points"]

    if rand_points.shape[1] < 4 or sharp_near.shape[1] < 4:
        raise ValueError("Expected sampled arrays with xyz+sdf columns.")

    n_sharp, n_coarse_near, n_coarse_space = n_supervision
    rng = np.random.default_rng(seed)

    coarse_near_pool = rand_points[:400000]
    coarse_space_pool = rand_points[400000:]
    if coarse_space_pool.shape[0] == 0:
        coarse_space_pool = rand_points

    sharp_sel = _sample_rows(rng, sharp_near, n_sharp)
    coarse_near_sel = _sample_rows(rng, coarse_near_pool, n_coarse_near)
    coarse_space_sel = _sample_rows(rng, coarse_space_pool, n_coarse_space)

    merged = np.concatenate([sharp_sel, coarse_near_sel, coarse_space_sel], axis=0)
    points = merged[:, :3].astype(np.float32)
    sdfs = merged[:, 3].astype(np.float32)
    sdfs = np.clip(sdfs, -0.015, 0.015) / 0.015

    points_t = torch.from_numpy(points).unsqueeze(0).to(device)
    targets_t = torch.from_numpy(sdfs).to(device)
    return points_t, targets_t, n_sharp


def _acc_iou(pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, float]:
    accuracy = (pred == label).float().mean().item()
    intersection = (pred * label).sum()
    union = (pred + label).gt(0).sum()
    if union.item() == 0:
        iou = 1e-5
    else:
        iou = (intersection / union).item() + 1e-5
    return accuracy, iou


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, n_sharp: int) -> Dict[str, float]:
    labels = (targets >= 0).float()
    pred = (logits >= 0).float()

    overall_acc, overall_iou = _acc_iou(pred, labels)
    sharp_acc, sharp_iou = _acc_iou(pred[:n_sharp], labels[:n_sharp])
    coarse_acc, coarse_iou = _acc_iou(pred[n_sharp:], labels[n_sharp:])

    return {
        "overall_accuracy": overall_acc,
        "overall_iou": overall_iou,
        "coarse_accuracy": coarse_acc,
        "coarse_iou": coarse_iou,
        "sharp_accuracy": sharp_acc,
        "sharp_iou": sharp_iou,
    }


def save_obj_mesh(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path.as_posix())


def select_fps_indices(points_xyz: torch.Tensor, n_target: int) -> torch.Tensor:
    n_points = points_xyz.shape[0]
    n_target = max(1, min(n_target, n_points))
    if n_target == n_points:
        return torch.arange(n_points, device=points_xyz.device)

    ratio = max(n_target / float(n_points), 1.0 / float(n_points))
    batch = torch.zeros(n_points, dtype=torch.long, device=points_xyz.device)
    idx = fps(points_xyz, batch=batch, ratio=ratio, random_start=False)
    idx = idx[:n_target]

    if idx.numel() < n_target:
        mask = torch.ones(n_points, dtype=torch.bool, device=points_xyz.device)
        mask[idx] = False
        remaining = torch.nonzero(mask, as_tuple=False).flatten()
        idx = torch.cat([idx, remaining[: (n_target - idx.numel())]], dim=0)
    return idx


def encode_with_latent_length(
    model: MichelangeloAutoencoder,
    coarse_surface: torch.Tensor,
    sharp_surface: torch.Tensor,
    latent_length: int,
    sample_posterior: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if latent_length < 2:
        raise ValueError("latent_length must be >= 2.")

    per_branch = max(1, latent_length // 2)
    encoder = model.encoder

    coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
    sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]

    coarse_data = encoder.embedder(coarse_pc)
    if coarse_feats is not None and coarse_feats.shape[-1] > 0:
        if encoder.embed_point_feats:
            coarse_feats = encoder.embedder(coarse_feats)
        coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)
    coarse_data = encoder.input_proj(coarse_data)

    sharp_data = encoder.embedder(sharp_pc)
    if sharp_feats is not None and sharp_feats.shape[-1] > 0:
        if encoder.embed_point_feats:
            sharp_feats = encoder.embedder(sharp_feats)
        sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)
    sharp_data = encoder.input_proj1(sharp_data)

    if coarse_pc.shape[0] != 1 or sharp_pc.shape[0] != 1:
        raise ValueError("This inference script currently expects batch size = 1.")

    idx_coarse = select_fps_indices(coarse_pc[0], per_branch)
    idx_sharp = select_fps_indices(sharp_pc[0], per_branch)

    query_coarse = coarse_data[:, idx_coarse, :]
    query_sharp = sharp_data[:, idx_sharp, :]
    query = torch.cat([query_coarse, query_sharp], dim=1)

    shape_latents = encoder.cross_attn(query, coarse_data)
    shape_latents = shape_latents + encoder.cross_attn1(query, sharp_data)
    shape_latents = encoder.self_attn(shape_latents)
    if encoder.ln_post is not None:
        shape_latents = encoder.ln_post(shape_latents)

    kl_embed, _ = model.encode_kl_embed(shape_latents, sample_posterior=sample_posterior)
    decoded_latents = model.decode(kl_embed)
    return kl_embed, decoded_latents, shape_latents, int(query.shape[1])


def distribute_samples(total: int, n_groups: int) -> List[int]:
    if n_groups <= 0:
        return []
    base = total // n_groups
    rem = total % n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def flatten_record(record: Dict) -> Dict:
    flat = {}
    for k, v in record.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def resolve_config_and_ckpt(config_path: Path, ckpt_path: str) -> Tuple[Path, str]:
    cfg_path = config_path
    if not cfg_path.is_absolute():
        cfg_path = (THIS_DIR / cfg_path).resolve()

    cfg = OmegaConf.load(cfg_path)
    model_ckpt = ckpt_path
    if not model_ckpt:
        model_ckpt = cfg.system.shape_model.pretrained_model_name_or_path
        maybe = Path(model_ckpt)
        if not maybe.is_absolute():
            maybe = (THIS_DIR / maybe).resolve()
            model_ckpt = str(maybe)
    return cfg_path, model_ckpt


def main() -> None:
    scratch_root = os.environ.get("SCRATCH", "")
    default_data_root = str(Path(scratch_root) / "seg_data") if scratch_root else "./seg_data"
    default_precomp_root = str(Path(scratch_root) / "seg_data_precomp_sdf") if scratch_root else "./seg_data_precomp_sdf"

    parser = argparse.ArgumentParser(description="Dental latent-length reconstruction and local latent sampling sweep.")
    parser.add_argument("--config", type=str, default="configs/shape-autoencoder/Dora-VAE-test.yaml")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint override.")
    parser.add_argument("--data_root", type=str, default=default_data_root)
    parser.add_argument("--precomp_root", type=str, default=default_precomp_root)
    parser.add_argument("--out_dir", type=str, default="infer_out")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument("--scan_seed", type=int, default=0)
    parser.add_argument("--latent_lengths", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--sigmas", type=str, default="0.01,0.03")
    parser.add_argument("--local_samples_total", type=int, default=6)
    parser.add_argument("--n_supervision", type=str, default="21384,10000,10000")

    parser.add_argument("--octree_depth", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=262144)
    parser.add_argument("--angle_threshold", type=int, default=15)
    parser.add_argument("--point_number", type=int, default=65536)
    parser.add_argument("--force_resample", action="store_true")
    parser.add_argument("--sample_posterior", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    precomp_root = Path(args.precomp_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    latent_lengths = parse_int_csv(args.latent_lengths)
    sigmas = parse_float_csv(args.sigmas)
    n_supervision = parse_int_csv(args.n_supervision)
    if len(n_supervision) != 3:
        raise ValueError("--n_supervision must contain exactly 3 integers.")

    if args.local_samples_total < 0:
        raise ValueError("--local_samples_total must be >= 0.")
    if any(x <= 0 for x in latent_lengths):
        raise ValueError("All latent lengths must be > 0.")
    if any(s <= 0 for s in sigmas):
        raise ValueError("All sigmas must be > 0.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    set_global_seed(args.scan_seed)

    obj_paths = discover_obj_files(data_root)
    if not obj_paths:
        raise FileNotFoundError(f"No .obj files found under {data_root}")

    selected_obj = pick_obj_path(obj_paths, args.scan_seed)
    npz_path = expected_npz_path(precomp_root, selected_obj)

    run_dir = out_root / f"{selected_obj.parent.name}_{selected_obj.stem}_seed{args.scan_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ensure_npz_sample(
        obj_path=selected_obj,
        npz_path=npz_path,
        run_dir=run_dir,
        point_number=args.point_number,
        angle_threshold=args.angle_threshold,
        force_resample=args.force_resample,
    )

    config_path, resolved_ckpt = resolve_config_and_ckpt(Path(args.config), args.ckpt)
    if resolved_ckpt and not Path(resolved_ckpt).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_ckpt}. Provide --ckpt or ensure config checkpoint exists."
        )

    cfg = OmegaConf.load(config_path)
    shape_cfg = OmegaConf.to_container(cfg.system.shape_model, resolve=True)
    if resolved_ckpt:
        shape_cfg["pretrained_model_name_or_path"] = resolved_ckpt

    model = MichelangeloAutoencoder(shape_cfg).to(device)
    model.eval()

    coarse_surface, sharp_surface = load_scan_surfaces(npz_path, device)
    query_points, supervision_targets, n_sharp = build_supervision_batch(
        npz_path=npz_path,
        n_supervision=n_supervision,
        seed=args.scan_seed,
        device=device,
    )

    summary_rows: List[Dict] = []
    sigma_allocations = distribute_samples(args.local_samples_total, len(sigmas))

    with torch.no_grad():
        for latent_length in latent_lengths:
            latent_dir = run_dir / f"latent_{latent_length}"
            latent_dir.mkdir(parents=True, exist_ok=True)

            base_kl, base_decoded, shape_latents, actual_latent_length = encode_with_latent_length(
                model=model,
                coarse_surface=coarse_surface,
                sharp_surface=sharp_surface,
                latent_length=latent_length,
                sample_posterior=args.sample_posterior,
            )

            records: List[Dict] = []

            def evaluate_variant(variant_name: str, decoded_latents: torch.Tensor, sigma: float, sample_index: int) -> Dict:
                logits = model.query(query_points, decoded_latents).squeeze(0)
                metrics = compute_metrics(logits, supervision_targets, n_sharp)
                mesh_path = latent_dir / f"{variant_name}.obj"

                mesh_v_f, has_surface = model.extract_geometry_by_diffdmc(
                    decoded_latents,
                    octree_depth=args.octree_depth,
                    num_chunks=args.chunk_size,
                    save_slice_dir="",
                )
                if has_surface[0]:
                    verts, faces = mesh_v_f[0]
                    save_obj_mesh(mesh_path, verts, faces)

                record = {
                    "latent_length_target": latent_length,
                    "latent_length_actual": actual_latent_length,
                    "variant": variant_name,
                    "sigma": sigma,
                    "sample_index": sample_index,
                    "has_surface": bool(has_surface[0]),
                    "mesh_path": str(mesh_path) if has_surface[0] else "",
                    "latent_mean": float(decoded_latents.mean().item()),
                    "latent_var": float(decoded_latents.var().item()),
                }
                record.update(metrics)
                return record

            base_record = evaluate_variant(
                variant_name="base",
                decoded_latents=base_decoded,
                sigma=0.0,
                sample_index=0,
            )
            records.append(base_record)

            for sigma, sample_count in zip(sigmas, sigma_allocations):
                for j in range(sample_count):
                    noise_seed = args.scan_seed + latent_length * 1000 + int(sigma * 1e5) + j
                    torch.manual_seed(noise_seed)
                    noise = torch.randn_like(base_kl)
                    local_kl = base_kl + sigma * noise
                    local_decoded = model.decode(local_kl)
                    variant = f"local_sigma{sanitize_sigma(sigma)}_sample{j:02d}"
                    local_record = evaluate_variant(
                        variant_name=variant,
                        decoded_latents=local_decoded,
                        sigma=sigma,
                        sample_index=j,
                    )
                    records.append(local_record)

            payload = {
                "selected_obj": str(selected_obj),
                "selected_npz": str(npz_path),
                "latent_length_target": latent_length,
                "latent_length_actual": actual_latent_length,
                "records": records,
            }
            with (latent_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            for row in records:
                summary_row = {
                    "selected_obj": str(selected_obj),
                    "selected_npz": str(npz_path),
                    "latent_dir": str(latent_dir),
                }
                summary_row.update(row)
                summary_rows.append(summary_row)

    summary_jsonl = run_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = run_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(flatten_record(summary_rows[0]).keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(flatten_record(row))

    meta = {
        "config_path": str(config_path),
        "checkpoint": resolved_ckpt,
        "selected_obj": str(selected_obj),
        "selected_npz": str(npz_path),
        "latent_lengths": latent_lengths,
        "sigmas": sigmas,
        "local_samples_total": args.local_samples_total,
        "octree_depth": args.octree_depth,
        "chunk_size": args.chunk_size,
        "scan_seed": args.scan_seed,
        "output_dir": str(run_dir),
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Done. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
