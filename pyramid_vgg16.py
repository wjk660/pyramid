import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import faulthandler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class PyramidVgg16(nn.Module):

    def __init__(self, n_layers, input_image_channels=3, output_channels=1, loss_weights=None):  # n_layers=5 5层网络
        super(PyramidVgg16, self).__init__()
        # fixed number of channels throughout the network 网络中固定的通道数量
        self.no_channels = 32
        self.no_rc_per_block = 4

        # first convolution sets the image to the 'correct' number of channels
        self.conv1 = nn.Conv2d(input_image_channels, self.no_channels, (3, 3), padding=1)

        # create network structure创建网络结构
        self.upsample_blocks = []
        self.downsample_blocks = []
        self.pre_loss_convs = []
        for i in range(n_layers):
            self.upsample_blocks.append(self._create_rc_block(self.no_rc_per_block))
            self.downsample_blocks.append(self._create_rc_block(self.no_rc_per_block))
            self.pre_loss_convs.append(nn.Sequential( #todo:为什么用这个计算loss
                nn.ReLU(inplace=False),
                nn.Conv2d(self.no_channels, output_channels, 3, padding=1)  # todo 1x1 conv instead?
            ))
        # add one more upsample block 上采样块多加了一层，也就是说最上面的A在了上采样的过程中
        self.upsample_blocks.append(self._create_rc_block(self.no_rc_per_block))

        self.upsample_blocks = nn.ModuleList(self.upsample_blocks)
        self.downsample_blocks = nn.ModuleList(self.downsample_blocks)
        self.pre_loss_convs = nn.ModuleList(self.pre_loss_convs)

        # have one loss per output 损失 BCEWithLogitsLoss是sigmoid加上交叉熵BCE
        self.losses = [nn.BCEWithLogitsLoss(pos_weight=lw, reduction='none') for lw in loss_weights] #todo:?
        self.losses = nn.ModuleList(self.losses)
        # self.loss = nn.CrossEntropyLoss(weight=loss_weights)  # softmax inside

    def forward(self, x):
        x = self.conv1(x)

        # downsample blocks take as input prev step and upsample one as in unet
        downsampled = []
        for i, layer in enumerate(self.downsample_blocks):
            x = layer(x)
            downsampled.append(x)
            # max pooling must be done separately so that x can be re-used later
            x = nn.MaxPool2d(2, stride=2)(x)

        # keep track of multi-scale prediction of network
        multiscale_predictions = []
        # first upsample block has no summing of map from downsampled
        x = self.upsample_blocks[0](x)
        x = nn.Upsample(scale_factor=2.0, mode='nearest')(x)
        # [print(d.shape) for d in downsampled]
        for i, layer in enumerate(self.upsample_blocks[1:]):# upsample_blocks[0]是最顶端的结构，没有求和操作
            # sum map coming from correspondent downsample layer  来自对应的下采样层
            # print(x.shape, downsampled[-i-1].shape)
            x1=x
            x2=downsampled[-i - 1]  # -1,-2,-3,-4,-5
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # x = x + downsampled[-i - 1]  # todo concat here?
            x = x1 + x2  # todo concat here?
            x = layer(x)
            # store current resolution prediction (apply RC block first) 存储当前分辨率下的预测
            multiscale_predictions.append(self.pre_loss_convs[i](x))
            x = nn.Upsample(scale_factor=2.0, mode='nearest')(x)

        # no need to squash values in between 0-1, sigmoid is inside loss
        return multiscale_predictions

    # rc stands for ReLU followed by a 3x3 conv as depicted in paper
    def _create_rc_block(self, number_of_rc):
        block = []
        for i in range(number_of_rc):
            block.append(nn.LeakyReLU(inplace=False))
            block.append(nn.Conv2d(self.no_channels, self.no_channels, (3, 3), padding=1))

        return nn.Sequential(*block)

    # Computes multi-scale loss given a list of predictions and a list of matching size targets; loss at different scale is summed up.
    # A mask is applied to the loss so that unlabeled pixels are ignored.将掩码应用于损失，以便忽略未标记的像素。
    def compute_multiscale_loss(self, multiscale_prediction, multiscale_targets, multiscale_masks):
        # reduction logic: mask is applied afterwards on the non-reduced loss
        losses = [torch.sum(self.losses[i](x, y) * mask) / torch.sum(mask) for i, (x, y, mask) in  # loss计算：loss(input, target)
                  enumerate(zip(multiscale_prediction, multiscale_targets, multiscale_masks))]# todo：为什么这么计算loss
        # here sum will call overridden + operator
        return sum(losses)


if __name__ == '__main__':
    model=PyramidVgg16(5, loss_weights=[torch.tensor([0.1]), torch.tensor([0.4]), torch.tensor([1.]),
                                      torch.tensor([4]), torch.tensor([10])])
    print(model)
    #model 为你要打印的参数模型
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    # faulthandler.enable()
    # net = PyramidNet(5, loss_weights=[torch.tensor([0.1]), torch.tensor([0.4]), torch.tensor([1.]),
    #                                   torch.tensor([4]), torch.tensor([10])])
    # writer = SummaryWriter('runs/pyramid_network')
    # writer.add_graph(net)
    # # print(net)
    # with torch.no_grad():
    #     x = torch.randn((2, 3, 128, 128), requires_grad=True)
    #     targets = [torch.ones((2, s, s), requires_grad=True).unsqueeze(1) for s in [8, 16, 32, 64, 128]]
    #     masks = [torch.ones((2, s, s), requires_grad=True).unsqueeze(1) for s in [8, 16, 32, 64, 128]]
    #     ys = net(x)
    #     [print(y.shape) for y in ys]
    #     [print(y.shape) for y in targets]
    #     for y in ys:
    #         print(y.max().item(), y.min().item())
    #         plt.imshow(y[0].squeeze().numpy(), cmap='gray')
    #         plt.show()
    #     loss = net.compute_multiscale_loss(ys, targets, masks)
    #     # loss.backward()
