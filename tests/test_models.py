cat <<EOF > tests/test_models.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_Weights

def test_faster_rcnn_load():
    # Test if Faster R-CNN can be initialized
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval()
        print("Faster R-CNN loaded successfully!")
        assert model is not None
    except Exception as e:
        assert False, f"Faster R-CNN failed to load: {e}"

def test_retinanet_load():
    # Test if RetinaNet can be initialized
    try:
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        model.eval()
        print("RetinaNet loaded successfully!")
        assert model is not None
    except Exception as e:
        assert False, f"RetinaNet failed to load: {e}"

def test_dummy_inference():
    # Ensure models can handle a dummy tensor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model.eval()
    dummy_img = [torch.rand(3, 300, 300)]
    output = model(dummy_img)
    assert len(output) > 0
    print("Dummy inference test passed!")
EOF