import time

import torch
from torch.autograd import Variable
import super_resolution.model_g
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from d2l import torch as d2l
from PIL import Image
from img_transferrer import Transfer
import os
from learn_gan.CycleGan.model import Generator
from super_resolution import model_g

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_img(file_path, max_size=400, shape=None):
    img = Image.open(file_path).convert('RGB')
    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    return in_transform(img)[:3, :, :].unsqueeze(0)


def tensor_to_img(img):
    image = img.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def super_resolution(img_path):
    hr_img = Image.open(img_path)
    device = torch.device('cuda:0')
    generator = model_g.Generator()
    param_dict = torch.load('../super_resolution/output/dicts_p.pt')
    generator.load_state_dict(param_dict['generator'])
    generator.to(device)
    generator.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(hr_img).to(device)
    pre = generator(img[None, ...])[0]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        pre[i] = std[i] * pre[i] + mean[i]

    pre = pre.cpu().clamp(0.0, 1.0)
    sr_img = transforms.ToPILImage()(pre)
    sr_img.show()
    print(sr_img.size)
    if os.path.exists(img_path):
        os.remove(img_path)
    sr_img.save(img_path)


class ImgTransfer:
    def __init__(self, content_path, style_path, device=d2l.try_gpu(), style_weights=None, content_weight=1,
                 style_weight=1e6):
        self.content = load_img(content_path).to(device)
        self.style = load_img(style_path, shape=self.content.shape[-2:]).to(device)
        if style_weights is None:
            self.style_weights = {'conv1_1': 1.,
                                  'conv2_1': 0.8,
                                  'conv3_1': 0.5,
                                  'conv4_1': 0.3,
                                  'conv5_1': 0.1}
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = device

    def transfer(self, epochs=2000, lr=0.003):
        transfer = Transfer(self.device,
                            self.content,
                            self.style,
                            self.style_weights,
                            self.content_weight,
                            self.style_weight)
        img = transfer.train(epochs, lr)
        path = '../web_window/static/res_img/res' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.jpg'
        plt.imsave(path, tensor_to_img(img))
        return path

    @staticmethod
    def transfer_GAN(img_path, isSR=True, mode=None):
        if mode is None:
            print("gan method mode is none")
            return
        device = torch.device('cuda:0')
        batch_size = 1
        input_c, output_c = 3, 3
        size = 256
        param_dict = torch.load('../learn_gan/CycleGan/output/'+mode+'/param.pt')
        netG_B2A = Generator(output_c, input_c).cuda().to(device)
        netG_B2A.load_state_dict(param_dict['netG_B2A'])
        netG_B2A.eval()
        Tensor = torch.cuda.FloatTensor
        input_B = Tensor(batch_size, output_c, size, size)

        transforms_ = [transforms.Resize((size, size)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transforms_ = transforms.Compose(transforms_)

        img_b = Image.open(img_path).convert('RGB')
        item_B = transforms_(img_b)
        real_B = Variable(input_B.copy_(item_B))
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        transforms_ = transforms.Compose([
            transforms.Resize((img_b.size[1], img_b.size[0])),
            transforms.ToPILImage()
        ])
        res = transforms_(fake_A.squeeze(0))
        path = '../web_window/static/res_img/res' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.jpg'
        res.save(path)
        if isSR:
            super_resolution(path)
        return path
