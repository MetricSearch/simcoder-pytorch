import torch
import torch.nn as nn

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def load_model(model_name: str) -> nn.Module:
    match model_name:
        case "AlexNet_FC6":
            model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
            model.eval()
            model.classifier[6] = nn.Identity()
            preprocess = Compose(
                [
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    return model, preprocess
