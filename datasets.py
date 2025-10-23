from torch.utils.data import Dataset
import torch
import numpy as np
from einops import rearrange
import scipy.sparse as sp
from sklearn.decomposition import PCA

from utils import plot_ST, read_anndata

class ST2D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()
        adata = read_anndata(data_file)
        
        self.coordinates = adata.obsm["spatial"][:, :2].astype(float)
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        x_range, y_range = x_max - x_min, y_max - y_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }

# for eval only
class Full2D(Dataset):
    def __init__(self, side_length=200):
        self.side_length = side_length
        linspace = torch.linspace(-1, 1, side_length)
        self.coordinates = torch.cartesian_prod(linspace, linspace)
    
    def __len__(self):
        return self.side_length ** 2
    
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "coordinates": self.coordinates[idx,:].float(),
        }


class ST3D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()

        adata = read_anndata(data_file)

        self.coordinates = adata.obsm["spatial"][:, :3]
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        # self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        # self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min, z_min = list(self.coordinates.min(axis=0))
        x_max, y_max, z_max = list(self.coordinates.max(axis=0))
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
        self.coordinates[:,2] = (self.coordinates[:,2] - z_min) / z_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range, z_range)
            scale_x, scale_y, scale_z = x_range / max_range, y_range / max_range, z_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
            self.coordinates[:,2] *= scale_z
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }

class HRSS(Dataset):
    def __init__(self, data_file, **kwargs):
        super().__init__()
        self.data = np.load(data_file)

        if "embeddings" in self.data.keys():
            self.embeddings = self.data["embeddings"]
            self.raw = self.data["raw"]
            self.coordinates = self.data["spatial"][:, :2]
            self.dataset_length = self.coordinates.shape[0]
            self.n_channels = self.raw.shape[-1]
            self.n_embd = self.embeddings.shape[-1]
            self.embd_pca = PCA(n_components=3, random_state=0)
            self.embd_pca.fit(self.embeddings)

        else:
            self.raw = self.data["raw"]
            
            # switch normalization strategy here
            #self._uint16_normalize()
            self._minmax_normalize()

            # custom slicing of self.raw
            self.raw = self.raw[:,:,::10] # every 10 channels

            self.height, self.width, self.n_channels = self.raw.shape
            self.dataset_length = self.height * self.width
            h_linspace = torch.linspace(-1, 1, self.height) # rows
            w_linspace = torch.linspace(-1, 1, self.width) # columns
            self.coordinates = torch.cartesian_prod(h_linspace, w_linspace)
            self.embeddings = None
            self.raw = rearrange(self.raw, "H W C -> (H W) C")

            

        self.raw_pca = PCA(n_components=3, random_state=0)
        self.raw_pca.fit(self.raw)

    def has_embeddings(self):
        return False if self.embeddings is None else True
    
    def _uint16_normalize(self):
        self.raw = self.raw / 65535

    def _minmax_normalize(self):
        normalized_image = np.zeros_like(self.raw, dtype=float)

        for i in range(self.raw.shape[2]):
            channel = self.raw[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            
            if max_val != min_val:
                # normalized_image[:, :, i] = 2 * (channel - min_val) / (max_val - min_val) - 1
                normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized_image[:, :, i] = 0
        self.raw = normalized_image
    
    def plot_raw_representations(self, train_indices=None, val_indices=None):
        rgb = self.raw_pca.transform(self.raw)
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], rgb[train_indices,:], spot_size=1)
            val_fig = plot_ST(self.coordinates[val_indices,:], rgb[val_indices,:], spot_size=1)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, rgb)
            return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    
    def __len__(self):
        return self.dataset_length
    
    def get_raw_dim(self):
        return self.n_channels
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:]).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:]).float()
            }

class GraphST2D(Dataset):
    def __init__(self, h5ad_file,neighbors,keep_ratio=True, **kwargs):
        super().__init__()
        #if h5ad_file is not str, adata = h5ad_file
        if type(h5ad_file) == str:

            adata = read_anndata(h5ad_file)
        else:
            adata = h5ad_file
        self.sb = True
        
        self.coordinates = adata.obsm["spatial"][:, :2]
        self.neighbors = neighbors
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        self._normalize_coordinates(keep_ratio=keep_ratio)

        self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        self.coordinates = self.coordinates.astype(np.float64)
        x_range, y_range = x_max - x_min, y_max - y_min
        
        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
     
        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __len__(self):
        return self.n_cell
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }

class GraphST3D(Dataset):
    def __init__(self, h5ad_file, neighbors, keep_ratio=True, **kwargs):
        super().__init__()

        adata = read_anndata(h5ad_file) if isinstance(h5ad_file, str) else h5ad_file
        self.sb = True

        self.coordinates = adata.obsm["spatial"][:, :3]
        self.neighbors = neighbors

        self.raw_representations = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        self.n_cell, self.n_gene = self.raw_representations.shape

        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.embeddings.shape[0] == self.n_cell
            self.n_embd = self.embeddings.shape[1]
        else:
            self.embeddings = None

        self._normalize_coordinates(keep_ratio=keep_ratio)

        self.raw_pca = PCA(n_components=3, random_state=0)
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0)
            self.embd_pca.fit(self.embeddings)

    def has_embeddings(self):
        return self.embeddings is not None

    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min, z_min = self.coordinates.min(axis=0)
        x_max, y_max, z_max = self.coordinates.max(axis=0)
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        self.coordinates = self.coordinates.astype(np.float64)
        self.coordinates[:, 0] = (self.coordinates[:, 0] - x_min) / x_range
        self.coordinates[:, 1] = (self.coordinates[:, 1] - y_min) / y_range
        self.coordinates[:, 2] = (self.coordinates[:, 2] - z_min) / z_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio:
            max_range = max(x_range, y_range, z_range)
            scale_x = x_range / max_range
            scale_y = y_range / max_range
            scale_z = z_range / max_range
            self.coordinates[:, 0] *= scale_x
            self.coordinates[:, 1] *= scale_y
            self.coordinates[:, 2] *= scale_z

    def get_raw_dim(self):
        return self.n_gene

    def get_embd_dim(self):
        return self.n_embd if self.has_embeddings() else None

    def __len__(self):
        return self.n_cell

    def __getitem__(self, idx):
        sample = {
            "idx": idx,
            "coordinates": torch.tensor(self.coordinates[idx], dtype=torch.float),
            "raw_representations": torch.tensor(self.raw_representations[idx], dtype=torch.float),
            "neighbors": self.neighbors[idx]
        }
        if self.has_embeddings():
            sample["embeddings"] = torch.tensor(self.embeddings[idx], dtype=torch.float)
        return sample

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        coords = self.coordinates
        if train_indices is not None and val_indices is not None:
            train_fig = plot_ST(coords[train_indices], self.raw_pca.transform(self.raw_representations[train_indices]), spot_size)
            val_fig = plot_ST(coords[val_indices], self.raw_pca.transform(self.raw_representations[val_indices]), spot_size)
            return train_fig, val_fig
        else:
            return plot_ST(coords, self.raw_pca.transform(self.raw_representations), spot_size)

    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "No embeddings available!"
        coords = self.coordinates
        if train_indices is not None and val_indices is not None:
            train_fig = plot_ST(coords[train_indices], self.embd_pca.transform(self.embeddings[train_indices]), spot_size)
            val_fig = plot_ST(coords[val_indices], self.embd_pca.transform(self.embeddings[val_indices]), spot_size)
            return train_fig, val_fig
        else:
            return plot_ST(coords, self.embd_pca.transform(self.embeddings), spot_size)


def detect_dataset_type(data_file, **kwargs):
    """
    Intelligently detect the appropriate dataset type based on data characteristics.
    
    Args:
        data_file: Path to the data file (.h5ad)
        **kwargs: Additional arguments for dataset initialization
    
    Returns:
        str: Detected dataset type ('GraphST2D', 'ST2D', 'GraphST3D', 'ST3D')
    """
    import h5py
    
    try:
        adata = read_anndata(data_file)
        
        # Check spatial dimensions
        spatial_coords = adata.obsm["spatial"]
        coord_dims = spatial_coords.shape[1]
        
        # Check if graph-based features exist
        has_graph_features = False
        if hasattr(adata, 'obsp') and len(adata.obsp) > 0:
            # Check for adjacency matrices or similar graph structures
            has_graph_features = True
        elif hasattr(adata, 'uns') and 'graph' in adata.uns:
            has_graph_features = True
        elif 'neighbors' in kwargs:
            has_graph_features = True
        
        # Determine dataset type based on dimensions and graph features
        if coord_dims == 2:
            if has_graph_features:
                return "GraphST2D"
            else:
                return "ST2D"
        elif coord_dims == 3:
            if has_graph_features:
                return "GraphST3D"
            else:
                return "ST3D"
        else:
            raise ValueError(f"Unsupported coordinate dimensions: {coord_dims}. Expected 2 or 3.")
            
    except Exception as e:
        print(f"Warning: Could not auto-detect dataset type: {e}")
        print("Defaulting to ST2D")
        return "ST2D"

def create_dataset(data_file, dataset_type=None, **kwargs):
    """
    Create a dataset instance with intelligent type detection.
    
    Args:
        data_file: Path to the data file
        dataset_type: Optional explicit dataset type. If None, will auto-detect
        **kwargs: Additional arguments for dataset initialization
    
    Returns:
        Dataset instance
    """
    if dataset_type is None:
        dataset_type = detect_dataset_type(data_file, **kwargs)
    
    print(f"Creating {dataset_type} dataset from {data_file}")
    
    # Import the appropriate dataset class
    if dataset_type == "GraphST2D":
        return GraphST2D(data_file, **kwargs)
    elif dataset_type == "ST2D":
        return ST2D(data_file, **kwargs)
    elif dataset_type == "GraphST3D":
        return GraphST3D(data_file, **kwargs)
    elif dataset_type == "ST3D":
        return ST3D(data_file, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


if __name__ == "__main__":

    ds = ST3D("./data/preprocessed_data/E9_two_slices.h5ad", True, True)
    raw = ds.raw_representations
    print(raw.max(), raw.min())
    sparsity = (raw==0).sum() / (raw>=0).sum()
    print(sparsity)