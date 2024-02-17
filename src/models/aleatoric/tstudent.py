import torch.nn as nn
import torch.nn.functional as F
import torch

from models.aleatoric.modules.encoder import Encoder
from models.aleatoric.modules.decoder import Decoder

def make_model(args):
    return ALEATORIC_TSTUDENT(args)
    
class Reduction(nn.Module):
    def __init__(self):
        super(Reduction, self).__init__()
    def forward(self, x):
        return 5 * torch.sigmoid(x) + 2
    
class ALEATORIC_TSTUDENT(nn.Module):
    def __init__(self, config):
        super(ALEATORIC_TSTUDENT, self).__init__()
        self.drop_rate = config.drop_rate
        in_channels = config.in_channels
        filter_config = (64, 128)

        self.encoders = nn.ModuleList()
        self.decoders_mean = nn.ModuleList()
        self.decoders_t = nn.ModuleList()
        self.decoders_v = nn.ModuleList()
  
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

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
            self.decoders_t.append(Decoder(decoder_filter_config[i],
                                                decoder_filter_config[i + 1],
                                                decoder_n_layers[i]))
            
            # decoder architecture
            self.decoders_v.append(Decoder(decoder_filter_config[i],
                                              decoder_filter_config[i + 1],
                                              decoder_n_layers[i]))

        # final classifier (equivalent to a fully connected layer)
        self.classifier_mean = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)
        self.classifier_t = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)
        self.classifier_v = nn.Conv2d(filter_config[0], in_channels, 3, 1, 1)

        self.filter_v = Reduction()



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
        feat_t = feat
        feat_v = feat
        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 2):
            feat_mean = self.decoders_mean[i](feat_mean, indices[1 - i], unpool_sizes[1 - i])
            feat_t = self.decoders_t[i](feat_t, indices[1 - i], unpool_sizes[1 - i])
            feat_v = self.decoders_v[i](feat_v, indices[1 - i], unpool_sizes[1 - i])
            if i == 0:  
                feat_mean = F.dropout(feat_mean, p=self.drop_rate)
                feat_t = F.dropout(feat_t, p=self.drop_rate)
                feat_v = F.dropout(feat_v, p=self.drop_rate)

        output_mean = self.classifier_mean(feat_mean)
        output_t = self.classifier_t(feat_t)
        output_v = self.classifier_v(feat_v)

        output_v = self.filter_v(output_v)
        
        results = {'mean': output_mean, 't': output_t, 'v': output_v}
        return results



