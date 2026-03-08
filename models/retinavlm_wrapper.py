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
        hf_config = RetinaVLMConfig()
        print(hf_config)
        super().__init__(hf_config)

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
        
        # Get the dtype of the model to match input
        model_dtype = next(self.model.parameters()).dtype
        
        images = [self.convert_any_image_to_normalized_tensor(image) for image in images]
        images = torch.stack(images, dim=0).to(device=self.model.device, dtype=model_dtype)
        
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
    from huggingface_hub import hf_hub_download
    import torch

    # 1. 설정 로드
    rvlm_config = RetinaVLMConfig.from_pretrained("RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist")
    rvlm_config.update(config)
    rvlm_config.model.checkpoint_path = None

    # 2. 모델 인스턴스 직접 생성
    model = RetinaVLM(rvlm_config)

    # 3. 가중치 파일 다운로드 및 로드
    print("Downloading weights from HuggingFace...")
    repo_id = "RobbieHolland/RetinaVLM"
    
    # 가중치 파일 후보들 (서버 경로이므로 반드시 '/' 사용)
    filenames = ["RetinaVLM-Specialist/pytorch_model.bin", "RetinaVLM-Specialist/model.safetensors"]
    
    archive_file = None
    for filename in filenames:
        try:
            archive_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=config.pretrained_model_dir)
            break
        except Exception:
            continue
            
    if archive_file is None:
        raise OSError(f"Could not find weights in {repo_id} for {filenames}")

    print(f"Loading weights from {archive_file}...")
    if archive_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(archive_file, device="cpu")
    else:
        state_dict = torch.load(archive_file, map_location="cpu")

    # 가중치 주입
    model.load_state_dict(state_dict, strict=False)

    return model


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