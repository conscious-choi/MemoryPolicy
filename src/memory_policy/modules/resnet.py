import torch
import torch.nn as nn
import torchvision.models as tvm

def _build_resnet(name: str, pretrained: bool, replace_stride_with_dilation):
    fn = getattr(tvm, name)
    # torchvision 버전 호환 처리
    try:
        # 최신 API
        weights = getattr(tvm, f"{name.capitalize()}_Weights").DEFAULT if pretrained else None
        return fn(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
    except Exception:
        # 구 API
        return fn(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)

class ResNetPatchEmbed(nn.Module):
    """
    ResNet 백본으로 만든 패치 임베딩
    forward(x) -> tokens: (B, N, C), grid_size: (H, W)
    """
    def __init__(
        self,
        resnet: str = "resnet18",         # resnet18, resnet34, resnet50 등
        pretrained: bool = True,
        out_stride: int = 16,             # 32 또는 16 권장
        embed_dim: int = 768,
        norm_layer: str = "ln",           # "ln" 또는 "none"
        freeze_bn: bool = True,
        flatten: bool = False
    ):
        super().__init__()
        assert out_stride in (16, 32), "out_stride은 16 또는 32만 권장"

        # dilation 세팅
        if out_stride == 32:
            replace = [False, False, False] # : 기본 다운샘플링 (더 작고 추상적)
        else:  # 16
            replace = [False, False, True] # → stride를 없애고 dilation=2로 대체 # dilation으로 공간 해상도 유지 (더 촘촘하고 spatial 정보 많음)

        backbone = _build_resnet(resnet, pretrained, replace)
        # stem + layer1..4만 사용
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        # in_channels는 fc in_features로 얻을 수 있음
        in_ch = backbone.fc.in_features
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer == "ln" else nn.Identity()

        # 안 쓰는 모듈 정리
        del backbone.avgpool
        del backbone.fc

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        self.flatten = flatten

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W), [0,1] 정규화 가정
        returns:
          tokens: (B, N, C)
          grid_size: (h, w)
        """
        feat = self.stem(x)          # (B, Cb, h, w)
        feat = self.proj(feat)       # (B, C,  h, w)
        B, C, h, w = feat.shape

        if not self.flatten:
            # 🔹 spatial map 유지
            return feat, (h, w)

        tokens = feat.flatten(2).transpose(1, 2)  # (B, N=h*w, C)
        tokens = self.norm(tokens)
        return tokens, (h, w)
