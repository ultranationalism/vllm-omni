import torch
from nunchaku import NunchakuZImageTransformer2DModel
from diffusers import ZImagePipeline
os.environ["HF_HOME"] = os.path.expanduser("~/hf_home_internal")
# 1. 加载双截棍量化变体 Transformer (内存中加载，不占用额外磁盘空间)
# 这里会根据你的 GPU 自动选择 int4 或 fp4
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    "nunchaku-tech/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors"
)
hf_token=""
# 2. 整合进原版 Pipeline
# 这会自动从 Tongyi-MAI 仓库拉取 VAE、Text Encoder 和 Scheduler 等组件
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", 
    transformer=transformer, 
    torch_dtype=torch.bfloat16
)

# 3. 推送到 Hub，并标注变体
# 使用 variant="nunchaku" 后，Transformer 权重会被保存为 diffusion_pytorch_model.nunchaku.safetensors
# 这样你就得到了一个包含完整组件且带量化变体的仓库
pipe.push_to_hub("ultranationalism", variant="nunchaku",token=hf_token)