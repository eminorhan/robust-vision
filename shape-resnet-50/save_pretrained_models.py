"""Load and save pre-trained Shape-ResNet-50 models.
"""
import argparse
from collections import OrderedDict
import torch
import torchvision.models
from torch.utils import model_zoo


def load_model(model_name):
    model_urls = {
        'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }

    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = model_zoo.load_url(model_urls[model_name])
    model.load_state_dict(checkpoint["state_dict"])
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load and save pre-trained Shape-ResNet-50 models')
    parser.add_argument('--model_save_dir', type=str,
                        help='directory where the pre-trained Shape-ResNet model will be stored')

    args = parser.parse_args()

    # Abbreviations:
    # SIN = Stylized-ImageNet
    # IN = ImageNet

    model_A = "resnet50_trained_on_SIN"
    model_B = "resnet50_trained_on_SIN_and_IN"
    model_C = "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"
    model = load_model(model_C)  # change to different model as desired

    for k, v in model.module.state_dict().items():
        print(k)

    print("Model loading completed.")

    # Save model locally
    torch.save(model.state_dict(), args.model_save_dir + 'resnet50_sin_in_in.pth.tar')
    print("Model saving completed.")
