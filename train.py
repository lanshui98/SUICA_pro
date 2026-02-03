import argparse
from omegaconf import OmegaConf
from utils import pprint_config
from systems import train_embedder, train_inr, fit_griddata, predict_inr
from pathlib import Path
import os, h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['embedder', 'inr', 'grid'], required=True)
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    configs = OmegaConf.load(args.conf)
    print(args.conf)
    pprint_config(configs)

    conf_path = Path(args.conf).resolve()
    conf_dir  = conf_path.parent
    repo_root = Path(__file__).resolve().parent  # train.py 所在目录
    cwd       = Path.cwd()

    def resolve_rel(pth: str) -> Path:
        p = Path(pth)
        if p.is_absolute():
            return p
        # 依次尝试：提交时的工作目录(-cwd) → 仓库根(train.py所在目录) → 配置文件目录
        for base in (cwd, repo_root, conf_dir):
            cand = (base / p)
            if cand.exists():
                return cand.resolve()
        # 都不存在时，按“仓库根”拼出一个候选并报错时打印
        return (repo_root / p).resolve()

    # 规范化路径
    configs.dataset.data_file = str(resolve_rel(str(configs.dataset.data_file)))
    configs.pipeline.optimization.logs = str(resolve_rel(str(configs.pipeline.optimization.logs)))

    print(f"[CONF] conf     = {conf_path}")
    print(f"[CONF] datafile = {configs.dataset.data_file}")
    print(f"[CONF] logs_dir = {configs.pipeline.optimization.logs}")

    p = configs.dataset.data_file
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"data_file 不存在: {p}\n"
            f"建议检查 -cwd 或将 YAML 中 data_file 改为绝对路径。"
        )

    # 轻量预检：是否误传了 .h5ad.gz
    with open(p, "rb") as f:
        if f.read(2) == b"\x1f\x8b":
            raise RuntimeError(f"{p} 看起来是 gzip 压缩包（.h5ad.gz），请先 gunzip 解压为 .h5ad")

    if not h5py.is_hdf5(p):
        raise RuntimeError(f"{p} 不是 HDF5（.h5ad）文件：请检查路径/文件是否损坏或为 zarr 目录")

    if args.mode == "embedder":
        train_embedder(configs)
    elif args.mode == "inr":
        train_inr(configs)
    elif args.mode == "grid":
        fit_griddata(configs)

    pprint_config(configs)