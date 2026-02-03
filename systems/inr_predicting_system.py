import torch
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import anndata as ad
from rich import print
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import numpy as np
import os
import importlib
from pathlib import Path
from omegaconf import OmegaConf

from networks import SirenNet, FourierFeatureNet, NGP
from utils import metrics, plot_ST
from systems.embedder_fitting_system import EmbedderFittingSystem
from datasets import ST2D

def _cpu_workers(default=8):
    """让 DataLoader 的 workers 与 LSF 的 -n 对齐，避免过度并行。"""
    import os
    try:
        n = int(os.environ.get("LSB_DJOB_NUMPROC", ""))
        if n > 1:
            return max(1, min(default, n - 1))
    except Exception:
        pass
    import os as _os
    return min(default, _os.cpu_count() or 2)

def _resolve_path(p, anchors):
    """
    相对→绝对。按 anchors 顺序尝试（例如 [-cwd, 代码目录]）。
    即使不存在也返回一个“最佳猜测”，方便报错打印。
    """
    p = Path(str(p))
    if p.is_absolute():
        return p
    for a in anchors:
        cand = Path(a) / p
        if cand.exists():
            return cand.resolve()
    return (Path(anchors[0]) / p).resolve()

def _fix_lightning_version(path_like):
    """
    若包含 .../lightning_logs/version_X/... 且该路径不存在，
    自动在同级 lightning_logs 下寻找最新的 version_* 并替换。
    """
    p = Path(path_like); s = str(p)
    tag = "/lightning_logs/version_"
    if tag not in s:
        return p
    pre, tail = s.split(tag, 1)
    if "/" not in tail:
        return p
    _, sub = tail.split("/", 1)
    ll = Path(pre) / "lightning_logs"
    if not ll.exists():
        return p
    versions = sorted(
        ll.glob("version_*"),
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
        reverse=True,
    )
    for v in versions:
        cand = v / sub
        if cand.exists():
            return cand.resolve()
    return p

def _ensure_parent_dir(path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


class INRFittingSystem(L.LightningModule):
    def __init__(self, configs, val_pca, decoder=None):
        super().__init__()

        self.save_hyperparameters(ignore=['decoder'])
        
        self.pipeline_configs = configs
        network_configs = self.pipeline_configs.inr

        final_act = "ReLU" if self.pipeline_configs.target == "raw_representations" else "Identity"

        if network_configs.model == "SIREN":
            self.fitting_model = SirenNet(
                    dim_in=network_configs.dim_in,
                    dim_hidden=network_configs.num_hidden_features, 
                    dim_out=network_configs.dim_out, 
                    num_layers=network_configs.num_hidden_layers, 
                    final_activation=final_act
                )
        elif network_configs.model == "FFN":
            # 支持新的编码选项（可选参数，向后兼容）
            encoding_type = getattr(network_configs, 'encoding_type', 'basic')
            mapping_size = getattr(network_configs, 'mapping_size', 256)
            encoding_scales = getattr(network_configs, 'encoding_scales', [1, 10, 100])
            anisotropic_3d = getattr(network_configs, 'anisotropic_3d', False)  # 3D各向异性编码
            z_scales = getattr(network_configs, 'z_scales', None)  # z方向频率
            
            self.fitting_model = FourierFeatureNet(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.num_hidden_features, 
                dim_out=network_configs.dim_out, 
                num_layers=network_configs.num_hidden_layers, 
                final_activation=final_act,
                encoding_type=encoding_type,
                mapping_size=mapping_size,
                encoding_scales=encoding_scales,
                anisotropic_3d=anisotropic_3d,
                z_scales=z_scales,
                network_configs=network_configs  # 传递完整的config以便访问参数
            )
        elif network_configs.model == "NGP":
            self.fitting_model = NGP(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.num_hidden_features, 
                dim_out=network_configs.dim_out, 
                num_layers=network_configs.num_hidden_layers, 
                final_activation=final_act
            )
        else:
            raise NotImplementedError
        
        self.val_pca = val_pca
        self.decoder = decoder
        self.output_cache = []


    def forward(self, x):
        output = self.fitting_model(x)
        return output
    
    
    def dice_loss(self, pred, gt):
        pred_binary = 2.0 * F.sigmoid(pred) - 1
        gt_binary = (gt > 0).float()
        
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        
        # Compute Dice coefficient
        dice_coeff = (2.0 * intersection + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice_coeff
        return dice_loss
    
    def recons_loss(self, pred, gt):
        # Ensure gt is float for loss computation
        gt = gt.float() if not gt.is_floating_point() else gt
        # MSE for non-zero elements
        mse_loss = F.mse_loss(pred[gt != 0], gt[gt != 0])
        # L1 loss to encourage sparsity
        l1_loss = F.l1_loss(pred, gt)

        return mse_loss + l1_loss + 0.01 * self.dice_loss(pred, gt)
    
    def fitting_loss(self, pred, gt):
        # Ensure gt is float for loss computation
        gt = gt.float() if not gt.is_floating_point() else gt
        return F.l1_loss(pred,gt) + F.mse_loss(pred,gt)

    def training_step(self, batch, batch_idx):
        if self.pipeline_configs.target == "embeddings":
            x, gt = batch["coordinates"], batch["embeddings"]
        else:
            x, gt = batch["coordinates"], batch["raw_representations"]
        pred = self.fitting_model(x)
        if self.current_epoch <= self.pipeline_configs.inr.phase:
            fitting_loss = self.fitting_loss(pred, gt)
        else:
            fitting_loss = torch.tensor(0.0, device=self.device)
            for param in self.fitting_model.parameters():
                param.requires_grad = False
        self.log("train/fitting_loss", fitting_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        loss = fitting_loss
        if self.decoder and self.pipeline_configs.inr.decoder.recon_loss:
            if self.current_epoch > self.pipeline_configs.inr.phase: #fine tune the decoder after epoch = phase
                recons = self.decoder(pred)
                recons_loss = self.recons_loss(recons, batch["raw_representations"])
                self.log("train/recons_loss", recons_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
                loss += recons_loss
        self.log("train/loss", loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        results = {}
        if self.pipeline_configs.target == "embeddings":
            x, gt = batch["coordinates"], batch["embeddings"]
            pred = self.fitting_model(x)
            fitting_loss = self.fitting_loss(pred, gt)
            loss = fitting_loss
            if self.decoder:
                raw = batch["raw_representations"]
                recons = self.decoder(pred)
                recons_loss = self.recons_loss(recons, raw)
                self.log("val/recons_loss", recons_loss, on_step=False, on_epoch=True)
                loss += recons_loss
                results["raw"] = raw.detach().cpu().numpy()
                results["recons"] = recons.detach().cpu().numpy()
        else:
            x, gt = batch["coordinates"], batch["raw_representations"]
            pred = self.fitting_model(x)
            fitting_loss = F.mse_loss(pred, gt)
            loss = fitting_loss

        self.log("val/fitting_loss", fitting_loss, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        results["x"] = x.detach().cpu().numpy()
        results["pred"] = pred.detach().cpu().numpy()
        results["gt"] = gt.detach().cpu().numpy()
        self.output_cache.append(results)

    def on_validation_epoch_end(self):
        outputs = self.output_cache
        all_gt = np.concatenate([x["gt"] for x in outputs], axis=0)
        all_pred = np.concatenate([x["pred"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        
        scores_fitting = metrics(all_gt, all_pred, prefix="val_fitting",fast=False)
        self.log_dict(scores_fitting, sync_dist=False)
        print(scores_fitting)

        if self.decoder:
            all_raw = np.concatenate([x["raw"] for x in outputs], axis=0)
            all_recons = np.concatenate([x["recons"] for x in outputs], axis=0)
            scores_recon = metrics(all_raw, all_recons, prefix="val_recon",fast=False)
            self.log_dict(scores_recon, sync_dist=False)
            print(scores_recon)
        # fig = plot_ST(all_x, self.val_pca.transform(all_pred))
        # self.logger.experiment.add_figure("val/pred", fig, self.current_epoch)

        self.output_cache.clear()

    def configure_optimizers(self):
        param_list = list(self.fitting_model.parameters())
        if self.pipeline_configs.inr.decoder.finetune:
            param_list += list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(param_list, lr=self.pipeline_configs.optimization.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        x, raw = batch["coordinates"], batch["raw_representations"]
        
        pred = self.fitting_model(x)
        step_output = {}
        if self.pipeline_configs.target == "embeddings":
            # embd = batch["embeddings"] 
            # step_output["embd"] = embd.detach().cpu().numpy()
            if self.decoder:
                recons = self.decoder(pred)
                step_output["recons"] = recons.detach().cpu().numpy()
        
        step_output["x"] = x.detach().cpu().numpy()
        step_output["raw"] = raw.detach().cpu().numpy()
        step_output["pred"] = pred.detach().cpu().numpy()
        self.output_cache.append(step_output)

    def on_predict_epoch_end(self):
        outputs = self.output_cache
        all_raw = np.concatenate([x["raw"] for x in outputs], axis=0)
        all_pred = np.concatenate([x["pred"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        from scipy.sparse import csr_matrix
        adata = ad.AnnData(X=csr_matrix(all_raw))
        all_pred = csr_matrix(all_pred)
        adata.obsm["spatial"] = all_x
        if self.pipeline_configs.target == "raw_representations":
            adata.obsm["fitted_raw"] = all_pred
            
        elif self.pipeline_configs.target == "embeddings":
            if self.decoder:
                all_recons = np.concatenate([x["recons"] for x in outputs], axis=0)
                adata.obsm["reconstructed_raw"] = all_recons
            # all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
            # adata.obsm["embeddings"] = all_embd
            adata.obsm["fitted_embd"] = all_pred
        
        write_file = os.path.join(self.logger.log_dir, self.pipeline_configs.reconstructed_data)
        print(f"Writing to {write_file} ... It may take some time ...")
        adata.write_h5ad(write_file)
        

def predict_inr(configs):
    # 让 ${case} 等插值立即生效
    OmegaConf.resolve(configs)

    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline

    pl.seed_everything(pipeline_configs.optimization.seed, workers=True)
    torch.set_float32_matmul_precision("highest")

    # ---------- 路径归一化（集群安全） ----------
    repo_root = Path(__file__).resolve().parent  # 当前 systems 文件所在目录
    anchors = [Path.cwd(), repo_root]            # 优先 -cwd，其次代码目录

    # dataset.data_file 可能指向 lightning 产物：做绝对化 + version_* 容错
    if hasattr(dataset_configs, "data_file"):
        df = _resolve_path(dataset_configs.data_file, anchors)
        df = _fix_lightning_version(df)
        dataset_configs.data_file = str(df)

    # 日志目录绝对化
    pipeline_configs.optimization.logs = str(_resolve_path(pipeline_configs.optimization.logs, anchors))
    Path(pipeline_configs.optimization.logs).mkdir(parents=True, exist_ok=True)

    # decoder ckpt（可选）
    decoder = None
    if getattr(pipeline_configs.inr, "decoder", None) and getattr(pipeline_configs.inr.decoder, "ckpt", None):
        dec_ckpt = _fix_lightning_version(_resolve_path(pipeline_configs.inr.decoder.ckpt, anchors))
        decoder = EmbedderFittingSystem.load_from_checkpoint(str(dec_ckpt)).fitting_model.decoder
        print(f"[decoder] loaded from: {dec_ckpt}")
    else:
        # 确保有字段，避免后续 getattr 报错
        pipeline_configs.inr.decoder = getattr(pipeline_configs.inr, "decoder", {"recon_loss": False, "finetune": False})

    # 预测 ckpt（必需）
    if not getattr(pipeline_configs, "prediction", None) or not getattr(pipeline_configs.prediction, "ckpt", None):
        raise ValueError("pipeline.prediction.ckpt 是预测必需项")
    pred_ckpt = _fix_lightning_version(_resolve_path(pipeline_configs.prediction.ckpt, anchors))

    # ---------- 维度推断（来自数据或自定义坐标） ----------
    dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
    dataset = dataset_class(**dataset_configs)
    assert pipeline_configs.target in ["embeddings", "raw_representations"]

    # 输入维度：若 custom 则读自定义坐标；否则从 dataset 样本推断
    if hasattr(pipeline_configs, "custom_coords_file") and pipeline_configs.predict_mode == "custom":
        custom_coords = np.load(str(_resolve_path(pipeline_configs.custom_coords_file, anchors)))
        coord_dim = custom_coords.shape[1]
        pipeline_configs.inr.dim_in = coord_dim
        print(f"Detected custom coord dim: {coord_dim}D")
    else:
        sample = dataset[0]["coordinates"]
        coord_dim = sample.shape[0] if sample.ndim == 1 else sample.shape[1]
        pipeline_configs.inr.dim_in = coord_dim
        print(f"Detected dataset coord dim: {coord_dim}D")

    # 输出维度
    if pipeline_configs.target == "embeddings":
        pipeline_configs.inr.dim_out = dataset.get_embd_dim()
    else:
        pipeline_configs.inr.dim_out = dataset.get_raw_dim()

    print(f"Predicting ST [italic red]{pipeline_configs.target}[/italic red] with INR ...")
    print(f"dim_in={pipeline_configs.inr.dim_in}, dim_out={pipeline_configs.inr.dim_out}")

    # ---------- 从 ckpt 加载 INR（注意参数名！） ----------
    # 你的类 __init__(self, configs, val_pca, decoder=None)，因此用 configs=..., val_pca=None
    fitting_system = INRFittingSystem.load_from_checkpoint(
        str(pred_ckpt),
        configs=pipeline_configs,
        val_pca=None,
        decoder=decoder,
    )
    fitting_system.pipeline_configs = pipeline_configs
    fitting_system.eval()

    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)
    trainer = L.Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    batch_size = pipeline_configs.optimization.batch_size
    nw = _cpu_workers(default=8)

    # ---------- 预测模式 ----------
    if pipeline_configs.predict_mode == "all":
        dl = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=nw, drop_last=False)
        trainer.predict(fitting_system, dl)

    elif pipeline_configs.predict_mode == "val":
        _, val_idx = train_test_split(list(range(len(dataset))), test_size=dataset_configs.val_proportion)
        val_ds = Subset(dataset, val_idx)
        dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=nw, drop_last=False)
        trainer.predict(fitting_system, dl)

    elif pipeline_configs.predict_mode == "custom":
        print("[cyan]Start custom coordinate prediction...[/cyan]")
        coords = custom_coords.astype(np.float32)  # 上面已加载
        class _CustomDataset(torch.utils.data.Dataset):
            def __init__(self, c):
                self.coordinates = c
                self.raw_representations = np.zeros((len(c), 1), dtype=np.float32)
            def __len__(self): return len(self.coordinates)
            def __getitem__(self, i):
                return {
                    "coordinates": self.coordinates[i],
                    "raw_representations": self.raw_representations[i]
                }
        dl = DataLoader(_CustomDataset(coords), shuffle=False, batch_size=batch_size, num_workers=nw, drop_last=False)
        trainer.predict(fitting_system, dl)
        print(f"[green]Custom prediction saved under: {fitting_system.logger.log_dir}[/green]")

    else:
        raise ValueError(f"Unknown predict_mode: {pipeline_configs.predict_mode}")
