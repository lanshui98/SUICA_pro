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

from networks import SirenNet, FourierFeatureNet, NGP
from utils import metrics, plot_ST
from systems.embedder_fitting_system import EmbedderFittingSystem
from datasets import ST2D


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
            self.fitting_model = FourierFeatureNet(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.num_hidden_features, 
                dim_out=network_configs.dim_out, 
                num_layers=network_configs.num_hidden_layers, 
                final_activation=final_act
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
        # MSE for non-zero elements
        mse_loss = F.mse_loss(pred[gt != 0], gt[gt != 0])
        # L1 loss to encourage sparsity
        l1_loss = F.l1_loss(pred, gt)

        return mse_loss + l1_loss + 0.01 * self.dice_loss(pred, gt)
    
    def fitting_loss(self, pred, gt):
        return F.l1_loss(pred,gt) + F.mse_loss(pred,gt)

    def training_step(self, batch, batch_idx):
        if self.pipeline_configs.target == "embeddings":
            x, gt = batch["coordinates"], batch["embeddings"]
        else:
            x, gt = batch["coordinates"], batch["raw_representations"]
        pred = self.fitting_model(x)
        if self.pipeline_configs.inr.phase is not None:
            if self.current_epoch <= self.pipeline_configs.inr.phase:
                fitting_loss = self.fitting_loss(pred, gt)
            else:
                fitting_loss = torch.tensor(0.0, device=self.device)
                for param in self.fitting_model.parameters():
                    param.requires_grad = False
        else:
            fitting_loss = self.fitting_loss(pred, gt)
        self.log("train/fitting_loss", fitting_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        loss = fitting_loss
        if self.decoder and self.pipeline_configs.inr.decoder.recon_loss:
            if self.pipeline_configs.inr.phase is not None:
                if self.current_epoch > self.pipeline_configs.inr.phase:
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
            scores_recon = metrics(all_raw, all_recons, prefix="val_recon", fast=False)
            self.log_dict(scores_recon, sync_dist=False)
            # Save self.decoder
            decoder_path = os.path.join(self.logger.log_dir, "decoder.pth")
            torch.save(self.decoder.state_dict(), decoder_path)
            print(f"Decoder saved to {decoder_path}")
            
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
            embd = batch["embeddings"] 
            step_output["embd"] = embd.detach().cpu().numpy()
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

        adata = ad.AnnData(X=all_raw)
        adata.obsm["spatial"] = all_x
        if self.pipeline_configs.target == "raw_representations":
            adata.obsm["fitted_raw"] = all_pred
            
        elif self.pipeline_configs.target == "embeddings":
            if self.decoder:
                all_recons = np.concatenate([x["recons"] for x in outputs], axis=0)
                adata.obsm["reconstructed_raw"] = all_recons
            all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
            adata.obsm["embeddings"] = all_embd
            adata.obsm["fitted_embd"] = all_pred
        
        write_file = os.path.join(self.logger.log_dir, self.pipeline_configs.reconstructed_data)
        print(f"Writing to {write_file} ... It may take some time ...")
        adata.write_h5ad(write_file)
        

def train_inr(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline

    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally
    torch.set_float32_matmul_precision("highest") # make use of tensorcore, highest = default pytorch

    # Auto-detect dataset type if needed
    if dataset_configs.type == 'auto':
        from datasets import detect_dataset_type, create_dataset
        detected_type = detect_dataset_type(dataset_configs.data_file, **dataset_configs)
        print(f"Auto-detected dataset type: {detected_type}")
        dataset = create_dataset(dataset_configs.data_file, dataset_type=detected_type, **dataset_configs)
    else:
        # dataset configuration
        dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
        dataset = dataset_class(**dataset_configs)
    assert pipeline_configs.target in ["embeddings", "raw_representations"]
    
    pipeline_configs.inr.dim_in = 2
        
    if pipeline_configs.target == "embeddings":
        pipeline_configs.inr.dim_out = dataset.get_embd_dim()
        # val_pca = dataset.embd_pca
    elif pipeline_configs.target == "raw_representations":
        pipeline_configs.inr.dim_out = dataset.get_raw_dim()
        # val_pca = dataset.raw_pca


    print(f"Fitting ST [italic red]{pipeline_configs.target}[/italic red] with INR ...")

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=dataset_configs.val_proportion)
    print(len(train_idx), len(val_idx))

    print(f"{val_idx[:10]=}") # check whether seed works
    train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    batch_size = pipeline_configs.optimization.batch_size
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=False)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
    num_batches_in_epoch = len(train_dataloader)

    # pipeline configuration
    if pipeline_configs.inr.decoder:
        decoder = EmbedderFittingSystem.load_from_checkpoint(pipeline_configs.inr.decoder.ckpt).fitting_model.decoder
        print("[yellow]with pretrained decoder[/yellow]")
    else:
        decoder = None
        pipeline_configs.inr.decoder = {"recon_loss": False, "finetune": False}
        print("[yellow]with no decoder[/yellow]")

    fitting_system = INRFittingSystem(pipeline_configs, val_pca=None, decoder=decoder)
    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch}", save_last=True)
    trainer = L.Trainer(
                #num_sanity_val_steps=0,
                accumulate_grad_batches=num_batches_in_epoch,
                max_epochs=pipeline_configs.optimization.epochs,
                check_val_every_n_epoch=pipeline_configs.optimization.val_freq,
                log_every_n_steps=1, 
                logger=tb_logger, 
                callbacks=[checkpoint_callback],
                devices=1
            )
    trainer.fit(fitting_system, train_dataloader, val_dataloader)
    
    del train_dataset
    del train_dataloader
    import gc
    gc.collect()
    # predict
    if pipeline_configs.predict_mode=="all":
        del val_dataset
        del val_dataloader
        gc.collect()
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, test_dataloader)        
    elif pipeline_configs.predict_mode=="val":
        del dataset
        gc.collect()
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, test_dataloader) 
    else:
        print("End without writing predictions.")
        print("[cyan]Start custom coordinate prediction...[/cyan]")
        del val_dataset
        del val_dataloader
        del dataset
        gc.collect()

        # 1. Get custom coordinates
        if hasattr(pipeline_configs, 'custom_coords_file'):
            # Load coordinates from .npy file
            custom_coords = np.load(pipeline_configs.custom_coords_file)
            print(f"Load custom coordinates from file: {pipeline_configs.custom_coords_file}")
            print(f"Coordinate shape: {custom_coords.shape}")
        elif hasattr(pipeline_configs, 'custom_coordinates'):
            # Get coordinates directly from config
            custom_coords = np.array(pipeline_configs.custom_coordinates)
            print(f"Load custom coordinates from config, shape: {custom_coords.shape}")
        else:
            # Default example coordinates
            print("[yellow]No custom coordinates specified, using default examples[/yellow]")
            custom_coords = np.array([
                [10.0, 20.0, 0.0],
                [30.0, 40.0, 0.0], 
                [50.0, 60.0, 1.0],
                [70.0, 80.0, 1.5]
            ])
        
        # 2. Build custom dataset (reference datasets.py L113)
        class CustomCoordinateDataset:
            """
            Simple custom coordinate dataset
            Only contains coordinate information, references datasets.py implementation
            """
            def __init__(self, coordinates):
                # Ensure coordinates are numpy array with correct shape
                if isinstance(coordinates, list):
                    coordinates = np.array(coordinates)
                assert coordinates.shape[1] == 3, f"Coordinates must be (N, 3) format, current: {coordinates.shape}"
                
                self.coordinates = coordinates.astype(np.float32)
                
                # Create dummy raw_representations (all zeros, won't be used but needs to exist)
                self.raw_representations = np.zeros((len(coordinates), 1), dtype=np.float32)
                
                # If target is embeddings, also need dummy embeddings
                if pipeline_configs.target == "embeddings":
                    # Create dummy embeddings based on INR output dimension
                    embd_dim = pipeline_configs.inr.dim_out
                    self.embeddings = np.zeros((len(coordinates), embd_dim), dtype=np.float32)
                
                print(f"Create custom dataset with {len(self.coordinates)} coordinate points")
            
            def __len__(self):
                return len(self.coordinates)
            
            def __getitem__(self, idx):
                """
                Return data item, format consistent with original dataset
                """
                item = {
                    "coordinates": self.coordinates[idx],
                    "raw_representations": self.raw_representations[idx]
                }
                
                # If target is embeddings, add embeddings field
                if pipeline_configs.target == "embeddings":
                    item["embeddings"] = self.embeddings[idx]
                
                return item
        
        # 3. Create custom dataset and data loader
        custom_dataset = CustomCoordinateDataset(custom_coords)
        custom_dataloader = DataLoader(
            custom_dataset, 
            shuffle=False, 
            batch_size=batch_size, 
            num_workers=8, 
            drop_last=False
        )
        
        print(f"Create custom data loader, batch size: {batch_size}")
        
        # 4. Use trainer for prediction
        print("[cyan]Starting INR prediction...[/cyan]")
        trainer.predict(fitting_system, custom_dataloader)
        
        print(f"[green]Custom coordinate prediction completed! Results saved to: {fitting_system.logger.log_dir}[/green]")


