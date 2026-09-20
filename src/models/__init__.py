from src.models.module_img import ImgModule
try:
    from src.models.module_tab import TabModule
except ImportError:
    TabModule = None


def get_model(config, device):
    if config["MODEL"]["architecture"] in ["TSViT", "ConvLSTM", "ResNet", "ViT", "ViTFacto"]:
        return ImgModule(config).to(device)

    elif config["MODEL"]["architecture"] in [
        "TabTSViT",
        "TabConvLSTM",
        "MixResNet",
        "EnvResNet",
        "EnvViTFactorizeModel",
        "TabResNetConvLSTM",
        "TabViTFactorizeModel",
        "MultiViTFactorizeModel",
        "UNet",
        "UTAE",
    ]:
        return TabModule(config).to(device)

    else:
        raise NameError(f"Model architecture {config['MODEL']['architecture']} not found")
