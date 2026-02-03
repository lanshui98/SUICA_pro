import torch
import torch.nn as nn
import numpy as np

class GaussianEncoding(nn.Module):
    """
    Given an input of size [batches, num_input_channels],
     returns a tensor of size [batches, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._scale = scale
        
        # 初始化 B 矩阵（固定编码矩阵）
        B = torch.randn((num_input_channels, mapping_size)) * scale
        # 固定编码矩阵（使用 register_buffer 以便设备管理）
        self.register_buffer('B', B)

    def forward(self, x):
        x = x @ self.B

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class MultiScaleGaussianEncoding(nn.Module):
    """
    多尺度 Fourier 特征编码 - 使用多个频率范围，提升对不同区域的表达能力
    特别适合处理裂缝等复杂区域
    """
    def __init__(self, num_input_channels, mapping_size=128, scales=[1, 10, 100]):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.scales = scales
        
        self.encodings = nn.ModuleList([
            GaussianEncoding(num_input_channels, mapping_size, scale=s)
            for s in scales
        ])

    def forward(self, x):
        # 对每个尺度进行编码并拼接
        encoded = [enc(x) for enc in self.encodings]
        return torch.cat(encoded, dim=1)


class EnhancedGaussianEncoding(nn.Module):
    """
    增强版 Fourier 特征编码
    - 多尺度频率编码
    - 支持各向异性编码（对z方向使用不同频率）
    """
    def __init__(self, num_input_channels, mapping_size=128, scales=[1, 10, 100], 
                 anisotropic_3d=False, z_scales=None):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.scales = scales
        self.anisotropic_3d = anisotropic_3d and num_input_channels == 3
        
        # 如果是3D且需要各向异性编码，为z方向使用不同的频率
        if self.anisotropic_3d:
            if z_scales is None:
                # 默认z方向使用更低的频率（因为z方向稀疏）
                z_scales = [s * 0.1 for s in scales]  # z方向频率降低10倍
            self.z_scales = z_scales
            
            # xy方向编码（前2维）
            self.xy_encodings = nn.ModuleList([
                GaussianEncoding(2, mapping_size, scale=s)
                for s in scales
            ])
            # z方向编码（第3维）
            self.z_encodings = nn.ModuleList([
                GaussianEncoding(1, mapping_size, scale=s)
                for s in z_scales
            ])
        else:
            # 标准多尺度编码
            self.encodings = nn.ModuleList([
                GaussianEncoding(num_input_channels, mapping_size, scale=s)
                for s in scales
            ])

    def forward(self, x):
        # 各向异性编码（3D，z方向稀疏）
        if self.anisotropic_3d:
            # 分离xy和z坐标
            xy = x[:, :2]  # [batch, 2]
            z = x[:, 2:3]  # [batch, 1]
            
            # xy方向编码
            xy_encoded = [enc(xy) for enc in self.xy_encodings]
            # z方向编码（使用不同频率）
            z_encoded = [enc(z) for enc in self.z_encodings]
            
            # 拼接所有编码
            encoded = xy_encoded + z_encoded
            result = torch.cat(encoded, dim=1)
        else:
            # 标准多尺度编码
            encoded = [enc(x) for enc in self.encodings]
            result = torch.cat(encoded, dim=1)
        
        return result


class FourierFeatureNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, final_activation,
                 encoding_type='basic', mapping_size=256, encoding_scales=[1, 10, 100],
                 anisotropic_3d=False, z_scales=None, network_configs=None):
        super().__init__()
        
        # 如果network_configs提供了，优先从中获取参数
        if network_configs is not None:
            anisotropic_3d = getattr(network_configs, 'anisotropic_3d', anisotropic_3d)
            z_scales = getattr(network_configs, 'z_scales', z_scales)
        
        # 选择编码类型
        if encoding_type == 'basic':
            self.transform = GaussianEncoding(dim_in, mapping_size, scale=10)
            encoding_dim = mapping_size * 2
        elif encoding_type == 'multiscale':
            self.transform = MultiScaleGaussianEncoding(dim_in, mapping_size, scales=encoding_scales)
            encoding_dim = mapping_size * 2 * len(encoding_scales)
        elif encoding_type == 'enhanced':
            self.transform = EnhancedGaussianEncoding(
                dim_in, mapping_size, scales=encoding_scales,
                anisotropic_3d=anisotropic_3d,
                z_scales=z_scales
            )
            
            # 计算编码维度
            if anisotropic_3d and dim_in == 3:
                # xy方向 + z方向的编码（每个都有mapping_size*2维）
                encoding_dim = mapping_size * 2 * len(encoding_scales) * 2  # xy和z各一套
            else:
                encoding_dim = mapping_size * 2 * len(encoding_scales)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
        
        self.encoding_dim = encoding_dim
        final_activation = getattr(nn, final_activation)()

        self.layers = nn.Sequential(
                        nn.Linear(encoding_dim, dim_hidden),
                        nn.ReLU(),
                        *[nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers)],
                        nn.Linear(dim_hidden, dim_out),
                        final_activation
                    )
        
    
    def forward(self, x):
        ff = self.transform(x)
        y = self.layers(ff)
        return y


if __name__ == "__main__":
    # 测试基本版本
    transform_basic = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2, 
                                       final_activation="Identity", encoding_type='basic')
    x = torch.ones([100, 2])
    y = transform_basic(x)
    print(f"Basic encoding output shape: {y.shape}")
    
    # 测试多尺度版本
    transform_multiscale = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2,
                                            final_activation="Identity", encoding_type='multiscale',
                                            mapping_size=128, encoding_scales=[1, 10, 100])
    y = transform_multiscale(x)
    print(f"Multiscale encoding output shape: {y.shape}")
    
    # 测试增强版本（推荐用于裂缝区域）
    transform_enhanced = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2,
                                          final_activation="Identity", encoding_type='enhanced',
                                          mapping_size=128, encoding_scales=[1, 10, 100])
    y = transform_enhanced(x)
    print(f"Enhanced encoding output shape: {y.shape}")