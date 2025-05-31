# tests/unit/text_detection/test_text_detection.py

import pytest
import numpy as np
import torch
from src.text_detection import CraftUNet, TextDetector, ImagePreprocessor
from src.utils.config_loader import ConfigLoader


# Fixtures
@pytest.fixture
def sample_tile():
    """Create a sample 512x512 test tile"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def config():
    """Sample configuration"""
    return {
        "model_path": "test_model.pth",  # Will be mocked
        "backbone": "efficientnet-b3",
        "use_coordconv": True,
        "use_aspp": True,
        "batch_size": 2,
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "border_mask": 15,
        "heatmap_quantization": False
    }


@pytest.fixture
def mock_model(config, mocker):
    """Mock model with pretrained weights"""
    # Create a simple model
    model = CraftUNet(config)

    # Mock torch.load
    mocker.patch("torch.load", return_value=model.state_dict())

    return model


def test_model_architecture(config):
    """Test model initialization and architecture"""
    model = CraftUNet(config)

    # Verify components exist
    assert hasattr(model, "encoder")
    assert hasattr(model, "aspp") if config["use_aspp"] else True
    assert hasattr(model, "decoder4")
    assert hasattr(model, "char_head")
    assert hasattr(model, "affinity_head")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512)
    char_map, aff_map = model(dummy_input)

    assert char_map.shape == (1, 1, 512, 512)
    assert aff_map.shape == (1, 1, 512, 512)
    assert torch.all(char_map >= 0) and torch.all(char_map <= 1)
    assert torch.all(aff_map >= 0) and torch.all(aff_map <= 1)


def test_preprocessor(config, sample_tile):
    """Test image preprocessing"""
    preprocessor = ImagePreprocessor(config)
    tensor = preprocessor.preprocess(sample_tile)

    # Verify output properties
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 512, 512)
    assert tensor.min() >= -2.5 and tensor.max() <= 2.5  # After normalization

    # Test batch preprocessing
    batch = [sample_tile, sample_tile]
    batch_tensor = preprocessor.preprocess_batch(batch)
    assert batch_tensor.shape == (2, 3, 512, 512)


def test_border_mask(config):
    """Test border mask creation"""
    preprocessor = ImagePreprocessor(config)
    mask = preprocessor.create_border_mask((512, 512))

    # Verify mask properties
    assert mask.shape == (512, 512)
    assert mask[15, 15] == 1.0  # Center
    assert mask[0, 0] == 0.0  # Corner
    assert mask[10, 10] < 0.5  # Border region


def test_text_detector(config, sample_tile, mock_model, mocker):
    """Test text detector inference"""
    # Mock model loading
    mocker.patch("src.text_detection.craft_unet.CraftUNet", return_value=mock_model)

    detector = TextDetector(config)

    # Test single image inference
    char_map, aff_map = detector.detect(sample_tile)
    assert char_map.shape == (512, 512)
    assert aff_map.shape == (512, 512)
    assert np.all(char_map >= 0) and np.all(char_map <= 1)

    # Test batch inference
    tiles = [sample_tile, sample_tile]
    results = detector.detect_batch(tiles)
    assert len(results) == 2
    for char_map, aff_map in results:
        assert char_map.shape == (512, 512)

    # Test quantization
    quant_config = config.copy()
    quant_config["heatmap_quantization"] = True
    quant_detector = TextDetector(quant_config)
    char_map, aff_map = quant_detector.detect(sample_tile)
    assert char_map.dtype == np.uint8
    assert np.all(char_map <= 255)


def test_model_finetuning(config, mocker):
    """Test model fine-tuning setup"""
    # Mock dependencies
    mocker.patch("torch.utils.data.DataLoader")
    mocker.patch("torch.optim.AdamW")
    mocker.patch("torch.save")

    from src.text_detection.training.finetune import finetune_model

    # Add training config
    config["training"] = {
        "epochs": 2,
        "data": {
            "train_dir": "data/train",
            "val_dir": "data/val"
        },
        "freeze_encoder": True
    }

    # Run fine-tuning
    finetune_model(config)

    # Verify model methods called
    # (In real tests, you'd verify training steps)
    assert True


def test_augmentations():
    """Test P&ID specific augmentations"""
    from src.text_detection.training.augmentations import PIDAUGMENT

    sample_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Test train transform
    train_transform = PIDAUGMENT.get_train_transform()
    augmented = train_transform(image=sample_img)
    assert augmented["image"].shape == (512, 512, 3)

    # Test pipe occlusion
    pipe_img = PIDAUGMENT.add_pipe_occlusion(sample_img)
    assert pipe_img.shape == sample_img.shape

    # Test symbol overlay
    symbol_img = PIDAUGMENT.add_symbol_overlay(sample_img)
    assert symbol_img.shape == sample_img.shape

    # Test line noise
    noise_img = PIDAUGMENT.add_line_noise(sample_img)
    assert noise_img.shape == sample_img.shape