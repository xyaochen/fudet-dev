import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner.base_module import BaseModule
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class FutrAttentionV2(BaseModule):
    """
    Add cross attention between modality.
    An attention module used in Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 embed_dims=256,
                 radar_dims=64,
                 num_cams=6,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 pc_range=None,
                 rad_cuda=True,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.pc_range = pc_range
        self.num_cams = num_cams
        self.rad_cuda = rad_cuda

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.fused_embed = 0
        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims
        
        if self.use_camera:
            self.img_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.img_attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_heads * num_levels * num_points)

            self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
            self.weight_dropout = nn.Dropout(0.0)
            self.fused_embed += embed_dims
        if self.use_radar:
            self.rad_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_points * 2)
            self.rad_attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_points)
            self.rad_value_proj = nn.Linear(radar_dims, radar_dims)
            self.rad_output_proj = nn.Linear(radar_dims, radar_dims)
            self.fused_embed += radar_dims

        if self.fused_embed > embed_dims:
            self.modality_fusion_attn = nn.MultiheadAttention(embed_dims, 8, batch_first=True)
            self.fusion_drop = nn.Dropout(0.1)
            self.fusion_norm = nn.LayerNorm(self.embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_radar:
            constant_init(self.rad_sampling_offsets, 0.)
            self.rad_sampling_offsets.bias.data = grid_init[:, 0].reshape(-1)
            constant_init(self.rad_attention_weights, val=0., bias=0.)
            xavier_init(self.rad_value_proj, distribution='uniform', bias=0.)
            xavier_init(self.rad_output_proj, distribution='uniform', bias=0.)
        if self.use_camera:
            constant_init(self.img_sampling_offsets, 0.)
            self.img_sampling_offsets.bias.data = grid_init.reshape(-1)
            constant_init(self.img_attention_weights, val=0., bias=0.)
        self._is_init = True

    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pts_feats=None,
                img_feats=None,
                rad_feats=None,
                rad_key_padding_mask=None,
                rad_spatial_shapes=None,
                rad_level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        if self.use_lidar:
            value = pts_feats
            bs, num_value, _ = value.shape
            
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

            value = self.value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.num_levels,
                                                    self.num_points)
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
            # ref_points = reference_points
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = ref_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            
            # [bs, C, num_query, num_points*num_levels]
            output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)
            # [bs, num_query, num_points*num_levels, C]
            output = output.permute(0, 2, 3, 1)
            pts_output = self.output_proj(output)

        if self.use_camera:
            # (B, 1, num_query, num_cams, num_points, num_levels)
            img_attention_weights = self.img_attention_weights(query).view(
                bs, num_query, self.num_cams, self.num_heads, self.num_levels * self.num_points)
            img_attention_weights = img_attention_weights.softmax(-1)

            img_attention_weights = img_attention_weights.view(bs, num_query, self.num_cams,
                                                    self.num_heads, 1, 
                                                    self.num_levels,
                                                    self.num_points)
            img_attention_weights = img_attention_weights.permute(0, 3, 4, 1, 2, 6, 5)
            # img_attention_weights = self.img_attention_weights(query).view(
            #    bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
            # reference_points: (bs, num_query, 3)
            
            sampling_offsets = self.img_sampling_offsets(query).view(
                bs, num_query, 1, self.num_heads, self.num_levels, self.num_points, 2)
            # output (B, num_heads, dim, num_query, num_cam, num_points, len(mlvl_feats))
            reference_points_3d, img_output, mask = feature_sampling(
                img_feats, reference_points, sampling_offsets, self.pc_range, kwargs['img_metas'])
            img_output = torch.nan_to_num(img_output)
            # mask = torch.nan_to_num(mask)

            img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * mask
            img_output = img_output * img_attention_weights
            img_output = img_output.view(bs, self.embed_dims, num_query, \
                                            self.num_cams*self.num_points*self.num_levels)
            
            # output (B, num_query, num_cams*num_points*num_levels, emb_dims)
            img_output = img_output.permute(0, 2, 3, 1)

            img_output = self.img_output_proj(img_output)

        if self.use_radar:
            if self.rad_cuda:
                value = rad_feats
                bs, num_value, _ = value.shape
                
                assert (rad_spatial_shapes[:, 0] * rad_spatial_shapes[:, 1]).sum() == num_value

                value = self.rad_value_proj(value)
                if rad_key_padding_mask is not None:
                    value = value.masked_fill(rad_key_padding_mask[..., None], 0.0)
                value = value.view(bs, num_value, self.num_heads, -1)
                
                sampling_offsets = self.rad_sampling_offsets(query).view(
                    bs, num_query, self.num_heads, 1, self.num_points, 2)
                attention_weights = self.rad_attention_weights(query).view(
                    bs, num_query, self.num_heads, self.num_points)
                attention_weights = attention_weights.softmax(-1)

                attention_weights = attention_weights.view(bs, num_query,
                                                        self.num_heads,
                                                        1,
                                                        self.num_points)
                ref_points = reference_points.unsqueeze(2).expand(-1, -1, 1, -1)
                # ref_points = reference_points
                ref_points = ref_points[..., :2]
                if ref_points.shape[-1] == 2:
                    offset_normalizer = torch.stack(
                        [rad_spatial_shapes[..., 1], rad_spatial_shapes[..., 0]], -1)
                    sampling_locations = ref_points[:, :, None, :, None, :] \
                        + sampling_offsets \
                        / offset_normalizer[None, None, None, :, None, :]
                else:
                    raise ValueError(
                        f'Last dim of reference_points must be'
                        f' 2, but get {reference_points.shape[-1]} instead.')
                if ((IS_CUDA_AVAILABLE and value.is_cuda)
                        or (IS_MLU_AVAILABLE and value.is_mlu)):
                    output = MultiScaleDeformableAttnFunction.apply(
                        value, rad_spatial_shapes, rad_level_start_index, sampling_locations,
                        attention_weights, self.im2col_step)
                else:
                    output = multi_scale_deformable_attn_pytorch(
                        value, rad_spatial_shapes, sampling_locations, attention_weights)

                radar_output = self.rad_output_proj(output)
            else:
                radar_output= feature_sampling_3D(
                    rad_feats, reference_points, self.pc_range)
                radar_output = torch.nan_to_num(radar_output)

                radar_output = radar_output.sum(-1).sum(-1)
                radar_output = radar_output.permute(0, 2, 1)

                radar_output = self.radar_output_proj(radar_output)

        if self.use_lidar and self.use_camera:
            pts_output = pts_output.view(bs*num_query, self.num_levels*self.num_points, self.embed_dims)
            img_output = img_output.view(bs*num_query, self.num_cams*self.num_levels*self.num_points, self.embed_dims)
            attn_out, _ = self.modality_fusion_attn(pts_output, img_output, img_output)
            # output = torch.cat((img_output, pts_output), dim=2)
            # output = self.modality_fusion_layer(output)
            output = pts_output + self.fusion_drop(attn_out)
            output = self.fusion_norm(output)
            output = output.sum(1).view(bs, num_query, self.embed_dims)
            
        elif self.use_camera and self.use_radar:
            output = torch.cat((img_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar:
            output = pts_output
        elif self.use_camera:
            output = img_output
        elif self.use_radar:
            output = img_output

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        if self.use_lidar:
            return self.dropout(output) + identity

        return self.dropout(output) + identity + self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)


def feature_sampling(mlvl_feats, reference_points, sampling_offsets, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query, _, num_heads, num_levels, num_points = sampling_offsets.shape[:6]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    # [B, num_cams, num_query, 2]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    # [bs, num_query, num_cams, num_heads, num_levels, num_points, 2]
    sampling_offsets[..., 0] /= img_metas[0]['img_shape'][0][1]
    sampling_offsets[..., 1] /= img_metas[0]['img_shape'][0][0]
    # [bs, num_cams, num_heads, num_query, num_levels, num_points, 2]
    sampling_offsets = sampling_offsets.permute(0, 2, 3, 1, 4, 5, 6)

    reference_points_cam = reference_points_cam[:, :, None, :, None, None, :] + sampling_offsets
                            
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = mask[:, :, None, :, None, None, :]
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    
    mask = mask.view(B, num_cam, num_heads, 1, num_query, num_levels, num_points).permute(0, 2, 3, 4, 1, 6, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        dim = C // num_heads
        feat = feat.view(B*N*num_heads, dim, H, W)
        reference_points_cam_lvl = reference_points_cam[:, :, :, :, lvl]
        reference_points_cam_lvl = reference_points_cam_lvl.reshape(B*N*num_heads, num_query, num_points, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, num_heads, dim, num_query, num_points).permute(0, 2, 3, 1, 4, 5)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, num_heads, dim, num_query, num_cam, num_points, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),
    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).view(bs, num_heads * embed_dims,
                                              num_queries, num_levels * num_points)
    return output