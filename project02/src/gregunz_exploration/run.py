from unet import UNet

unet = UNet(n_channels=1)
unet.train(batch_size=20, epochs=100)