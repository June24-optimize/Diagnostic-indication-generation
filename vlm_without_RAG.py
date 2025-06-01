import torch
import torch.nn as nn
import numpy as np
import math
import nibabel as nib
from transformers import AutoModel
biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


class PatchEmbedding(nn.Module): # This converts an input image into a sequence of patch embeddings (similar to word embeddings in NLP).
    def __init__(self, in_channel = 72, patch_size = 16, emb_dim = 768, image_height=280, image_width=320):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_height // patch_size) * (image_width // patch_size) # Using a Conv2D to split image into patches and project to embedding dim
        self.proj = nn.Conv2d(in_channel, emb_dim, kernel_size = 16, stride = patch_size)

    def forward(self, x): # x shape: [batch_size, C, H, W]
        x = self.proj(x) # -> [B, emb_dim, H/patch_size, W/patch_size]
        x = x.flatten(2) # -> [B, emb_dim, num_patches]
        x = x.transpose(1, 2) # -> [B, num_patches, emb_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    # A single block of a Transformer encoder, composed of:
    # LayerNorm → Multi-head Self-Attention → Residual
    # LayerNorm → Feedforward Network → Residual

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # Residual connections prevent vanishing gradients.
        # LayerNorm is applied before attention/MLP (pre-norm).
        # GELU is a smoother alternative to ReLU.

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim),
                                 nn.Dropout(dropout)
                                 )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] # Self-attention block with residual connection
        x = x + self.mlp(self.norm2(x)) # Feedforward block with residual connection
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_channels=72, num_heads=12, emb_dim=768, depth=12, mlp_dim=3072, dropout=0.1, image_height=280, image_width=320):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, image_height, image_width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim)) # learnable [CLS] tokens
        self.num_patches = (image_height // patch_size) * (image_width // patch_size) # 340
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim)) # [1, 341, 768]
        self.pos_drop = nn.Dropout(p = dropout)
        self.visual_proj = nn.Linear(emb_dim, 1024)  # match language model (BioGPT) embedding size  

        # Stack of Transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize parameters from truncated normal
        # samples values from a normal distribution but discards values beyond a certain range (typically ±2 standard deviations). This prevents:
        # Vanishing gradients: Extremely small initial weights can cause gradients to vanish during backpropagation.
        # Exploding gradients: Extremely large weights can destabilize training early on.
        # For ViTs, this is especially important because the self-attention mechanism is sensitive to the scale of initial weights.

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0] # [1, 72, 280, 320]
        x = self.patch_embed(x)  # [B, num_patches, emb_dim] [1, 401, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, emb_dim] [1, 341, 768]
        x = x + self.pos_embed[:, :x.size(1)] # Add positional encoding
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        visual_token = self.visual_proj(x[:, 0])  # [B, 768]
        return visual_token

subject = 'RS104'
data_dir = './data/' + f'{subject}.nii.gz'
image = nib.load(data_dir)
image = image.get_fdata() # (280, 320, 72)
image = np.expand_dims(image, axis=0) 
if isinstance(image, np.ndarray):
    image = torch.from_numpy(image).float()  # Convert to float32 tensor
if image.ndim == 4:
    image = image.permute(0, 3, 1, 2)   # CNN shape: (B, C, H, W)

image_height, image_width = image.shape[-2], image.shape[-1]

encoder = VisionTransformer(
    patch_size=16,
    in_channels=72,
    emb_dim=768,
    depth=12,
    num_heads=12,
    mlp_dim=3072,
    dropout=0.1, 
    image_height=image_height, 
    image_width=image_width)

visual_features = encoder(image)  # [8, 10]
print('visual feature generated', visual_features.shape)



# === BioBERT text decoder with visual prefix ===

# Load tokenizer and BioBERT
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
# bert = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load tokenizer and BioGPT (or another decoder-only model)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
language_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

prompt = "Describe the finding:"
inputs = tokenizer(prompt, return_tensors="pt")
visual_prefix = visual_features.unsqueeze(1)  # [1, 1, 768]

# Generate output tokens
outputs = language_model.generate(
    inputs_embeds=torch.cat([visual_prefix, language_model.get_input_embeddings()(inputs.input_ids)], dim=1),
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated description:", output_text)


