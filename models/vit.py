#
# Original code: https://www.kaggle.com/c/bms-molecular-translation/discussion/231190
# Updated for VIT
#

import numpy as np
import timm
from torch.nn.utils.rnn import pack_padded_sequence

from .fairseq_transformer import *
from .fairseq_transformer import PositionEncode1D, TransformerDecode


class VitEncoder(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()

        self.e = timm.create_model(backbone, pretrained=True)

    def forward(self, x):
        x = self.e.patch_embed(x)
        cls_token = self.e.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.e.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.e.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.e.pos_drop(x + self.e.pos_embed)
        x = self.e.blocks(x)
        x = self.e.norm(x)
        # tiny 16, 197, 192
        # small 16, 197, 384
        # base 16, 197, 768
        # base384 16, 577, 768
        return x


class VitDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.max_length = 300  # args.max_token
        self.embed_dim = args.embed_dim

        self.image_encode = nn.Identity()
        self.text_pos = PositionEncode1D(self.embed_dim, self.max_length)
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.text_decode = TransformerDecode(self.embed_dim,
                                             ff_dim=args.ff_dim,
                                             num_head=args.num_head,
                                             num_layer=args.num_layer)

        # ---
        self.logit = nn.Linear(self.embed_dim, self.vocab_size)

        # ----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)

    @torch.jit.unused
    def forward(self, image_embed, token, length):
        device = image_embed.device
        # 16, 577, 768
        image_embed = self.image_encode(image_embed).permute(1, 0, 2).contiguous()
        # (T,N,E) expected

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1, 0, 2).contiguous()

        text_mask_max_length = length.max()  # max_length
        text_mask = np.triu(np.ones((text_mask_max_length, text_mask_max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(device)

        # ----
        # <todo> mask based on length of token?
        # <todo> perturb mask as aug

        x = self.text_decode(text_embed, image_embed, text_mask)
        x = x.permute(1, 0, 2).contiguous()

        logit = self.logit(x)
        return logit

    @torch.jit.export
    def predict(self, image):
        STOI = {
            '<sos>': 190,
            '<eos>': 191,
            '<pad>': 192,
        }

        # ---------------------------------
        device = image.device
        batch_size = len(image)

        # image_embed = self.cnn(image)
        image_embed = self.image_encode(image).permute(1, 0, 2).contiguous()

        token = torch.full((batch_size, self.max_length), STOI['<pad>'], dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:, 0] = STOI['<sos>']

        # -------------------------------------
        eos = STOI['<eos>']
        pad = STOI['<pad>']

        # incremental_state = {}
        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        for t in range(self.max_length - 1):
            # last_token = token [:,:(t+1)]
            # text_embed = self.token_embed(last_token)
            # text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

            last_token = token[:, t]
            text_embed = self.token_embed(last_token)
            text_embed = text_embed + text_pos[:, t]  #
            text_embed = text_embed.reshape(1, batch_size, self.embed_dim)

            x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
            x = x.reshape(batch_size, self.embed_dim)
            # print(incremental_state.keys())

            l = self.logit(x)
            k = torch.argmax(l, -1)  # predict max
            token[:, t + 1] = k
            if ((k == eos) | (k == pad)).all():  break

        predict = token[:, 1:]
        return predict


class Vit(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.encoder = VitEncoder(backbone, args)
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
