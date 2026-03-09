from run.vision_language_pretraining import MiniGPT4Module
import hydra
import torch
import sys
from PIL import Image
import numpy as np
from huggingface_hub import login, HfApi
import scipy
import textwrap
from transformers import PreTrainedModel, PretrainedConfig

class RetinaVLMConfig(PretrainedConfig):
    model_type = "RetinaVLM"
    def __init__(self, torch_dtype="float32", **kwargs):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.__dict__.update(kwargs)

class RetinaVLM(PreTrainedModel):
    config_class = RetinaVLMConfig

    def __init__(self, config):
        super().__init__(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniGPT4Module(config, device=device).model.eval()

    @property
    def all_tied_weights_keys(self):
        return {}

    def convert_any_image_to_normalized_tensor(self, image_input):
        # Convert input to numpy array if it's a PIL Image
        if isinstance(image_input, Image.Image):
            image_input = np.array(image_input)
            if image_input.ndim == 3:  # If it has channels
                if image_input.shape[2] == 4:  # If RGBA, drop alpha channel
                    image_input = image_input[:, :, :3]
                # Permute from W x H x C to C x W x H for numpy processing
                image_input = np.transpose(image_input, (2, 0, 1))

        # Convert input to numpy array if it's a PyTorch tensor, assuming it is already C x W x H
        elif isinstance(image_input, torch.Tensor):
            # Make sure it's on CPU and convert to numpy
            image_input = image_input.cpu().numpy()

        # Check if the input is now a numpy array
        elif not isinstance(image_input, np.ndarray):
            raise TypeError("Unsupported image type. Ensure input is a PIL Image, NumPy array, or PyTorch tensor.")

        # If input is a 2D grayscale numpy array, add a channel dimension
        if image_input.ndim == 2:
            image_input = image_input[np.newaxis, :, :]

        # Resize the image if not already 192x192
        if image_input.shape[1] != 192 or image_input.shape[2] != 192:
            # Resize using scipy to maintain the channel-first format
            zoom_factors = [1, 192 / image_input.shape[1], 192 / image_input.shape[2]]
            image_input = scipy.ndimage.zoom(image_input, zoom_factors, order=1)  # Bilinear interpolation

        # Normalize the pixel values to 0-1 if the dtype indicates they are in the range 0-255
        if not np.issubdtype(image_input.dtype, np.floating):
            image_input = image_input.astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(image_input)

        return img_tensor

    def forward(self, images, queries, max_new_tokens=750):
        answer_preambles = [''] * len(images)
        if isinstance(images, torch.Tensor) and len(images.shape) == 2:
            images = [images]
        
        # Get the dtype and device of the model to match input
        model_param = next(self.model.parameters())
        model_dtype = model_param.dtype
        model_device = model_param.device

        images = [self.convert_any_image_to_normalized_tensor(image) for image in images]
        images = torch.stack(images, dim=0).to(device=model_device, dtype=model_dtype)
        
        outputs, samples = self.model.query(images, queries, answer_preamble=answer_preambles, max_new_tokens=max_new_tokens, output_only=True, return_samples=True)
        return outputs

def load_retinavlm(config):
    rvlm_config = RetinaVLMConfig.from_pretrained("saved_models/RetinaVLM-Specialist")
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None
    model = RetinaVLM.from_pretrained("saved_models/RetinaVLM-Specialist", config=rvlm_config, low_cpu_mem_usage=False, _fast_init=False).eval()
    return model

# def load_retinavlm_specialist_from_hf(config):
#     rvlm_config = RetinaVLMConfig.from_pretrained("RobbieHolland/RetinaVLM/RetinaVLM-Specialist")
#     rvlm_config.update(config)
#     rvlm_config.model.checkpoint_path = None
#     model = RetinaVLM.from_pretrained("RobbieHolland/RetinaVLM/RetinaVLM-Specialist", config=rvlm_config).eval()
#     return model

def load_retinavlm_specialist_from_hf(config):
    import torch
    import json
    import os
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file as safetensors_load_file

    # 1. 설정 로드
    rvlm_config = RetinaVLMConfig.from_pretrained("RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist")
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None

    # 2. 모델을 직접 생성 (from_pretrained 우회 → meta device 문제 없음)
    print("Creating RetinaVLM model...")
    model = RetinaVLM(rvlm_config)

    # 3. HuggingFace에서 체크포인트 다운로드
    print("Downloading checkpoint from HuggingFace...")
    repo_id = "RobbieHolland/RetinaVLM"
    subfolder = "RetinaVLM-Specialist"

    # sharded safetensors index 다운로드
    index_path = hf_hub_download(repo_id, f"{subfolder}/model.safetensors.index.json",
                                  cache_dir=config.pretrained_model_dir)
    with open(index_path) as f:
        index = json.load(f)

    # 필요한 shard 파일들 다운로드 및 로드
    shard_files = set(index["weight_map"].values())
    full_state_dict = {}
    for shard_file in sorted(shard_files):
        print(f"  Loading shard: {shard_file}")
        shard_path = hf_hub_download(repo_id, f"{subfolder}/{shard_file}",
                                      cache_dir=config.pretrained_model_dir)
        shard_dict = safetensors_load_file(shard_path)
        full_state_dict.update(shard_dict)

    print(f"  Total keys in checkpoint: {len(full_state_dict)}")

    # 4. 8-bit 양자화 가중치를 float16으로 역양자화(dequantize)
    #    체크포인트에 SCB(scale) + int8 weight가 있으면 float16으로 변환
    dequantized_dict = {}
    scb_keys = {k for k in full_state_dict if k.endswith('.SCB')}
    weight_format_keys = {k for k in full_state_dict if k.endswith('.weight_format')}
    skip_keys = scb_keys | weight_format_keys

    for key, tensor in full_state_dict.items():
        if key in skip_keys:
            continue

        # 8-bit 양자화된 가중치 확인 (int8 + SCB 존재)
        scb_key = key.rsplit('.', 1)[0] + '.SCB' if '.' in key else key + '.SCB'
        if tensor.dtype == torch.int8 and scb_key in full_state_dict:
            scb = full_state_dict[scb_key]
            # int8 역양자화: float_weight = int8_weight * scale / 127
            dequantized = tensor.float() * scb.float().unsqueeze(0) / 127.0
            dequantized_dict[key] = dequantized.half()
        else:
            dequantized_dict[key] = tensor

    print(f"  Dequantized {len([k for k in full_state_dict if full_state_dict[k].dtype == torch.int8])} int8 tensors to float16")

    # 5. state_dict 로드 (strict=False로 누락/추가 키 허용)
    missing, unexpected = model.load_state_dict(dequantized_dict, strict=False)

    print(f"\n  LOAD RESULT:")
    print(f"    Loaded keys: {len(dequantized_dict) - len(unexpected)}")
    print(f"    Missing keys: {len(missing)}")
    print(f"    Unexpected keys: {len(unexpected)}")
    if missing:
        # 카테고리별 missing 키 요약
        missing_cats = {}
        for k in missing:
            cat = k.split('.')[1] if '.' in k else k
            missing_cats[cat] = missing_cats.get(cat, 0) + 1
        for cat, count in sorted(missing_cats.items()):
            print(f"      Missing [{cat}]: {count} keys")

    return model.eval()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def debug(config):
    from dataset.retinal_text_dataset import RetinalTextDataset

    sys.path.append(config['octlatent_dir'])
    dataset = RetinalTextDataset(config, set_='validation')
    sample = dataset.__getitem__(2)
    images = [sample[0]]
    print(config.images_for_figures_dir + '/' + sample[1]['ImageId'])

    query = textwrap.dedent(f'''
                            Write an extensive report describing the OCT image and listing any present biomarkers or other observations. Do not provide a disease stage, or referral recommendation yet.
                            ''')
    queries = [query]

    retinavlm = RetinaVLM(config).eval()
    output = retinavlm.forward(images, queries)
    print(output[0])

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def save_model(config):
    api = HfApi()
    retinavlm = RetinaVLM(config).eval()

    retinavlm.save_pretrained("saved_models/RetinaVLM-Base")

    api.upload_folder(
        folder_path="saved_models/RetinaVLM-Base",
        path_in_repo="RetinaVLM-Base",
        repo_id="RobbieHolland/RetinaVLM",
        token=config.hf_write_token,
    )

    # api.upload_folder(
    #     folder_path="saved_models/RetinaVLM-Specialist",
    #     path_in_repo="RetinaVLM-Specialist",
    #     repo_id="RobbieHolland/RetinaVLM",
    #     token=config.hf_write_token,
    # )

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test_load(config):
    model = load_retinavlm(config)
    return model

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def load_from_api(config):
    model = load_retinavlm_specialist_from_hf(config)
    return model

if __name__ == "__main__":
    # save_model()
    # test_load()
    debug()
    # load_from_api()