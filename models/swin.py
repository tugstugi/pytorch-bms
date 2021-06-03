#
# Original code: https://www.kaggle.com/c/bms-molecular-translation/discussion/231190
# Updated for SWIN
#

import numpy as np
import timm
from torch.nn.utils.rnn import pack_padded_sequence

from .fairseq_transformer import *
from .fairseq_transformer import PositionEncode1D, TransformerDecode
from .vit import VitDecoder


class SwinEncoder(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()

        self.e = timm.create_model(backbone, pretrained=True)

    def forward(self, x):
        x = self.e.patch_embed(x)
        if self.e.absolute_pos_embed is not None:
            x = x + self.e.absolute_pos_embed
        x = self.e.pos_drop(x)
        x = self.e.layers(x)
        x = self.e.norm(x)
        return x


class Swin(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.encoder = SwinEncoder(backbone, args)
        self.decoder = VitDecoder(args)

    def _forward(self, x, encoded_captions, caption_lengths):
        encoder_out = self.encoder(x)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = self.decoder(encoder_out, encoded_captions, caption_lengths)
        targets = encoded_captions[:, 1:]
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # targets = encoded_captions
        return targets, predictions

    def _predict(self, x, max_length, tokenizer):
        encoder_out = self.encoder(x)
        predictions = self.decoder.predict(encoder_out)  # , max_length, tokenizer)
        # predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        predicted_captions = tokenizer.predict_captions(predictions.detach().cpu().numpy())
        predicted_captions = ['InChI=1S/' + p for p in predicted_captions]
        return predicted_captions

    def forward(self, x, predict, encoded_captions, caption_lengths, max_length, tokenizer):
        if predict:
            return self._predict(x, max_length, tokenizer)
        else:
            return self._forward(x, encoded_captions, caption_lengths)
