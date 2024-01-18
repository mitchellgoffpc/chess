import flax.linen as nn

class Downsample(nn.Module):
  channels: int
  stride: int
  enabled: bool

  def setup(self):
    if self.enabled:
      self.conv = nn.Conv(self.channels, kernel_size=(1, 1), strides=self.stride, use_bias=False)
      self.bn = nn.BatchNorm()

  def __call__(self, x, train=False):
    if self.enabled:
      return self.bn(self.conv(x), use_running_average=not train)
    else:
      return x

class BasicBlock(nn.Module):
  channels: int
  stride: int
  expansion: int
  downsample: bool = False
  
  def setup(self):
    self.conv1 = nn.Conv(self.channels, kernel_size=(3, 3), strides=self.stride, padding=1, use_bias=False)
    self.bn1 = nn.BatchNorm()
    self.conv2 = nn.Conv(self.channels * self.expansion, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)
    self.bn2 = nn.BatchNorm()
    self.conv_down = Downsample(self.channels * self.expansion, self.stride, self.downsample)

  def __call__(self, x, train=False):
    y = nn.relu(self.bn1(self.conv1(x), use_running_average=not train))
    y = self.bn2(self.conv2(y), use_running_average=not train)
    return nn.relu(y + self.conv_down(x, train=train))

class Bottleneck(nn.Module):
  channels: int
  stride: int
  expansion: int
  downsample: bool = False
  
  def setup(self):
    # NOTE: This uses the ResNet V1.5 architecture, so downsampling is done in the 3x3 convs instead of the 1x1s
    self.conv1 = nn.Conv(self.channels, kernel_size=(1, 1), use_bias=False)
    self.bn1 = nn.BatchNorm()
    self.conv2 = nn.Conv(self.channels, kernel_size=(3, 3), strides=self.stride, padding=1, use_bias=False)
    self.bn2 = nn.BatchNorm()
    self.conv3 = nn.Conv(self.channels * self.expansion, kernel_size=(1, 1), use_bias=False)
    self.bn3 = nn.BatchNorm()
    self.conv_down = Downsample(self.channels * self.expansion, self.stride, self.downsample)

  def __call__(self, x, train=False):
    y = nn.relu(self.bn1(self.conv1(x), use_running_average=not train))
    y = nn.relu(self.bn2(self.conv2(y), use_running_average=not train))
    y = self.bn3(self.conv3(y), use_running_average=not train)
    return nn.relu(y + self.conv_down(x, train=train))


class ResNet(nn.Module):
  STRIDES = [1, 2, 2, 2]
  CHANNELS = [64, 128, 256, 512]
  CONFIGS = {
    18: (BasicBlock, 1, [2, 2, 2, 2]),
    34: (BasicBlock, 1, [3, 4, 6, 3]),
    50: (Bottleneck, 4, [3, 4, 6, 3]),
    101: (Bottleneck, 4, [3, 4, 23, 3]),
    152: (Bottleneck, 4, [3, 8, 36, 3])}

  size: int

  def setup(self):
    assert self.size in ResNet.CONFIGS, f"Invalid size ({size}), choices are {list(ResNet.CONFIGS.keys())}"
    self.embed = nn.Embed(13, 16)
    self.conv1 = nn.Conv(64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False)
    self.bn1 = nn.BatchNorm()

    layers = []
    Block, expansion, blocks_per_layer = ResNet.CONFIGS[self.size]
    for out_channels, stride, num_blocks in zip(ResNet.CHANNELS, ResNet.STRIDES, blocks_per_layer):
      blocks = [Block(out_channels, stride, expansion, downsample=True)]
      blocks += [Block(out_channels, 1, expansion) for _ in range(1, num_blocks)]
      layers.append(nn.Sequential(blocks))

    self.layers = layers

  def __call__(self, x, train=False):
    x = self.embed(x)
    x = nn.relu(self.bn1(self.conv1(x), use_running_average=not train))
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
    for block in self.layers:
      x = block(x, train=train)
    return x


class ChessModel(nn.Module):
  def setup(self):
    self.resnet = ResNet(50)
    self.conv = nn.Conv(16, kernel_size=(1, 1))
    self.value = nn.Dense(1)
    self.policy = nn.Dense(64 * 144)

  def __call__(self, x, train=False):
    x = self.resnet(x)
    x = x.mean((1, 2))
    v = self.value(x)
    p = self.policy(x)
    return v, p
