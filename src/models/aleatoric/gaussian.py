import torch.nn as nn
import torch.nn.functional as F
from models.aleatoric.modules.encoder import Encoder
from models.aleatoric.modules.decoder import Decoder

def make_model(args):
    return ALEATORIC_GAUSSIAN(args)


class ALEATORIC_GAUSSIAN(nn.Module):
    def __init__(self, config):
        super(ALEATORIC_GAUSSIAN, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        out_channels = 1
        filter_config = (64, 128)

        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_var = nn.ModuleList()

        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],) # 128,64,64

        for i in range(0, 2):
            # encoder architecture
            self.encoders.append(Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders_mean.append(Decoder(decoder_filter_config[i],
                                               decoder_filter_config[i + 1],
                                               decoder_n_layers[i]))

            # decoder architecture
            self.decoders_var.append(Decoder(decoder_filter_config[i],
                                              decoder_filter_config[i + 1],
                                              decoder_n_layers[i]))

        # final classifier (equivalent to a fully connected layer)
        self.classifier_mean = nn.Conv2d(filter_config[0], out_channels, 3, 1, 1)
        self.classifier_var = nn.Conv2d(filter_config[0], out_channels, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 2):
            (feat, ind), size = self.encoders[i](feat)
            if i == 1:
                feat = F.dropout(feat, p=self.drop_rate)
            indices.append(ind)
            unpool_sizes.append(size)

        feat_mean = feat
        feat_var = feat
        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 2):
            feat_mean = self.decoders_mean[i](feat_mean, indices[1 - i], unpool_sizes[1 - i])
            feat_var = self.decoders_var[i](feat_var, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:
                feat_mean = F.dropout(feat_mean, p=self.drop_rate)
                feat_var = F.dropout(feat_var, p=self.drop_rate)

        output_mean = self.classifier_mean(feat_mean)
        output_var = self.classifier_var(feat_var)

        results = {'mean': output_mean, 'var': output_var}
        return results




