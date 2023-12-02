import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class ResBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = SynchronizedBatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)

        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        m['bn2'] = SynchronizedBatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        self.group1 = nn.Sequential(m)


    def forward(self, x):
        out = self.group1(x) + residual
        out = self.relu(out)

        return out

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels,mid_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GLA(nn.Module):

    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=default_conv, res_scale=1):
        super(GLA,self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels, channels//reduction, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.conv_assembly = BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.conv_assembly_fc = BasicBlock(conv, channels, channels, k_size, bn=False, act=nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(channels, chunk_size),
            nn.ReLU(inplace=True),
            nn.Linear(chunk_size, chunk_size)
        )

    # Super-Bit Locality-Sensitive Hashing
    def SBLSH(self, hash_buckets, x):
        #x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
        # assert rotations_shape[1] > rotations_shape[2]*rotations_shape[3]
        random_rotations = torch.nn.init.orthogonal_(torch.empty(x.shape[-1], hash_buckets))
        for _ in range(self.n_hashes-1):
            random_rotations = torch.cat([random_rotations, torch.nn.init.orthogonal_(torch.empty(x.shape[-1],hash_buckets))], dim=-1)
        # Training under multi-gpu: random_rotations.cuda() -> random_rotations.to(x.device) (suggested by Breeze-Zero from github: https://github.com/laoyangui/DLSN/issues/2)
        random_rotations = random_rotations.reshape(rotations_shape[0], rotations_shape[1], rotations_shape[2], hash_buckets).expand(N, -1, -1, -1).cuda() #[N, C, n_hashes, hash_buckets]
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets]
        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N,n_hashes,H*W]
        #add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes*H*W]
        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)
        return torch.cat([x, x_extra_back,x_extra_forward], dim=3)

    def forward(self, input):
        input1 = input.permute(0, 3, 1, 2).contiguous()
        N,_,H,W = input1.shape
        x_embed = self.conv_match(input1).view(N,-1,H*W).contiguous().permute(0,2,1)
        y_embed = self.conv_assembly(input1).view(N,-1,H*W).contiguous().permute(0,2,1)
        fc_embed = self.conv_assembly_fc(input1).view(N,-1,H*W).contiguous().permute(0,2,1)
        x_embed_extra_index = torch.arange(H * W).unsqueeze(0).unsqueeze(0).permute(0, 2, 1).cuda() # [1, HW, 1]

        L,C = x_embed.shape[-2:]

        #number of hash buckets/hash bits
        hash_buckets = min(L//self.chunk_size + (L//self.chunk_size)%2, 128)

        #get assigned hash codes/bucket number
        hash_codes = self.SBLSH(hash_buckets, x_embed) #[N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        #group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order
        mod_indices = (indices % L) #now range from (0->H*W)

        x_embed_sorted = batched_index_select(x_embed, mod_indices) #[N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices) #[N,n_hashes*H*W,C]
        fc_embed_embed_sorted = batched_index_select(fc_embed, mod_indices) #[N,n_hashes*H*W,C]

        #pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L%self.chunk_size if L%self.chunk_size!=0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_embed_embed_sorted, (N, self.n_hashes,-1, C*self.reduction))

        if padding:
            pad_x = x_att_buckets[:,:,-padding:,:].clone()
            pad_y = y_att_buckets[:,:,-padding:,:].clone()
            pad_fc = fc_att_buckets[:,:,-padding:,:].clone()
            x_att_buckets = torch.cat([x_att_buckets,pad_x],dim=2)
            y_att_buckets = torch.cat([y_att_buckets,pad_y],dim=2)
            fc_att_buckets = torch.cat([fc_att_buckets,pad_fc],dim=2)

        x_att_buckets = torch.reshape(x_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C] # q
        y_att_buckets = torch.reshape(y_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))
        fc_att_buckets = torch.reshape(fc_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1,eps=5e-5)

        #allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match) #[N, n_hashes, num_chunks, chunk_size*3, C]  # k
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
        fc_att_buckets = self.add_adjacent_buckets(fc_att_buckets)
        fc_raw_score = self.fc(fc_att_buckets).permute(0,1,2,4,3) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match) + fc_raw_score #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        #softmax    self.sigmoid2(self.fc2(self.sigmoid1(self.fc1(x_att_buckets))))
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score) #(after softmax)

        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C*self.reduction]
        bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1])
        ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction))

        #if padded, then remove extra elements
        if padding:
            ret = ret[:,:,:-padding,:].clone()
            bucket_score = bucket_score[:,:,:-padding].clone()

        #recover the original order
        ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)#[N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*H*W]
        #weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score,dim=1)
        ret = torch.sum(ret * probs, dim=1)
        ret =  ret.view(N,H,W,-1).contiguous()
        ret =ret *self.res_scale+input
        return ret

class GLAM(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, chunk_size, n_hashes):
        super(GLAM, self).__init__()
        modules_body = []
        modules_body.append(GLA(channels=n_feat, chunk_size=chunk_size, n_hashes=n_hashes, reduction=reduction, res_scale=1))
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x)
        res = res.permute(0, 3, 1, 2).contiguous()
        res = self.conv1(res)
        res = res.mul(self.res_scale)
        res = res.permute(0, 2, 3, 1).contiguous()
        res += x
        return res

class GlobalTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,n_feat=96):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # (conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, chunk_size, n_hashes):
        self.glam1=GLAM(default_conv, n_feat, 3, 16, act=nn.ReLU(True), res_scale=1, n_resblocks=20, chunk_size=144, n_hashes=4)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x = self.glam1(x)
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,n_feat=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            GlobalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 n_feat=n_feat)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
                x.cuda()
            else:
                x = blk(x)
                x.cuda()
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C


        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Generator(nn.Module):
    def __init__(self, img_size=448, patch_size=4, in_chans=3, num_classes=6,
                 embed_dim=96, depths=[1, 1, 1, 1], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "Generator expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # layer1
        self.resblock1 = ResBlock(int(embed_dim * 1),int(embed_dim * 1))
        self.layer1 = BasicLayer(dim=int(embed_dim * 1),
                                 input_resolution=(patches_resolution[0] // 1,
                                                   patches_resolution[1] // 1),
                                 depth=depths[0],
                                 num_heads=num_heads[0],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                                 norm_layer=norm_layer,
                                 downsample=PatchMerging if (0 < self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint,
                                 n_feat=int(embed_dim * 1))

        self.resblock2 = ResBlock(int(embed_dim * 2),int(embed_dim * 2))
        self.layer2 = BasicLayer(dim=int(embed_dim * 2),
                                 input_resolution=(patches_resolution[0] // 2,
                                                   patches_resolution[1] // 2),
                                 depth=depths[1],
                                 num_heads=num_heads[1],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                                 norm_layer=norm_layer,
                                 downsample=PatchMerging if (1 < self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint,
                                 n_feat=int(embed_dim * 2))

        # layer3
        self.resblock3 = ResBlock(int(embed_dim * 4), int(embed_dim * 4))
        self.layer3 = BasicLayer(dim=int(embed_dim * 4),
                                 input_resolution=(patches_resolution[0] // 4,
                                                   patches_resolution[1] // 4),
                                 depth=depths[2],
                                 num_heads=num_heads[2],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                                 norm_layer=norm_layer,
                                 downsample=PatchMerging if (2 < self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint,
                                 n_feat=int(embed_dim * 4))

        # layer4
        self.resblock4 = ResBlock(int(embed_dim * 8), int(embed_dim * 8))
        self.layer4 = BasicLayer(dim=int(embed_dim * 8),
                                 input_resolution=(patches_resolution[0] // 8,
                                                   patches_resolution[1] // 8),
                                 depth=depths[3],
                                 num_heads=num_heads[3],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                                 norm_layer=norm_layer,
                                 downsample=PatchMerging if (3 < self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint,
                                 n_feat=int(embed_dim * 8))

        self.up1 = Up(768, True)
        self.conv1 = DoubleConv(192 + 96, 192, 96)
        self.up2 = Up(96, True)
        self.conv2 = DoubleConv(96 + 48, 96, 96)
        self.up3 = Up(192, True)
        self.conv3 = DoubleConv(96 + 24, 48, 48)
        self.up4 = Up(48, True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )

        # build decoder layers 2
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.type(torch.FloatTensor)  # 转Float
        x = x.cuda()  # 转cuda
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        x_downsample.append(x)
        x = self.resblock1(x)
        x = self.layer1(x)
        x_downsample.append(x)
        x = self.resblock2(x)
        x = self.layer2(x)
        x_downsample.append(x)
        x = self.resblock3(x)
        x = self.layer3(x)
        x_downsample.append(x)
        x = self.resblock4(x)
        x = self.layer4(x)
        x = self.norm(x)  # B L C
        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        B, L, C = x.shape
        x = x.view(B, 28, 28, -1)
        x = x.permute(0, 3, 1, 2)
        x1 = self.up1(x)
        B3, L3, C3 = x_downsample[2].shape
        downsample3 = x_downsample[2].view(B3, 56, 56, -1)
        downsample3 = downsample3.permute(0, 3, 1, 2)
        x1 = torch.cat([downsample3, x1], dim=1)
        x1 = self.conv1(x1)
        x2 = self.up2(x1)
        B2, L2, C2 = x_downsample[1].shape
        downsample2 = x_downsample[1].view(B2, 112, 112, -1)
        downsample2 = downsample2.permute(0, 3, 1, 2)
        x2 = torch.cat([downsample2, x2], dim=1)
        x2 = self.conv2(x2)
        x3 = self.up3(x2)
        B1, L1, C1 = x_downsample[0].shape
        downsample1 = x_downsample[0].view(B1, 224, 224, -1)
        downsample1 = downsample1.permute(0, 3, 1, 2)
        x3 = torch.cat([downsample1, x3], dim=1)
        x3 = self.conv3(x3)
        x4 = self.up4(x3)
        x4 = self.conv4(x4)
        return x4

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class Discriminator(nn.Module):
    def __init__(self, num_classes=3, ndf = 64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, num_classes, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x)
        return x

from thop import profile

if __name__ == "__main__":
    import time
    model = Discriminator()
    device=torch.device('cuda:0')
    model.to(device)
    for idx, m in enumerate(model.modules()):
        print(idx, "-", m)
    s = time.time()
    rgb = torch.ones(1, 3, 448, 448, dtype=torch.float, requires_grad=False)
    rgb=rgb.type(torch.FloatTensor).cuda()
    out = model(rgb)
    out.to(device)
    flops, params = profile(model, inputs=(rgb,))
    print('parameters:', params)
    print('flops', flops)
    print('time: {:.4f}ms'.format((time.time()-s)*10))