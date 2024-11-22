
from xformers.factory.model_factory import xFormer, xFormerConfig
import torch
from typing import Any, Dict, List, Optional, Union

from pdb import set_trace as pb

class MyxFormer(xFormer):

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        encoder_input_mask: Optional[torch.Tensor] = None,
        decoder_input_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:

        # Encode to latent space if encoder is present
        if len(list(self.encoders.parameters())) > 0:
            encoders = self.encoders
            memory = src.clone()
            if isinstance(encoders, torch.nn.ModuleList):
                for encoder in encoders:
                    memory = encoder(memory, input_mask=encoder_input_mask)
            else:
                if self.rev_enc_pose_encoding:
                    memory = self.rev_enc_pose_encoding(src)

                # Reversible Encoder
                x = torch.cat([memory, memory], dim=-1)

                # Apply the optional input masking
                if encoder_input_mask is not None:
                    if x.dim() - encoder_input_mask.dim() > 1:
                        encoder_input_mask.unsqueeze(0)
                    x += encoder_input_mask.unsqueeze(-1)

                x = encoders(x)
                memory = torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

            if not self.decoders:
                return memory
        else:
            memory = src.clone()

        # If decoder: either use the encoder ouput, or just decode, both options are possible
        if len(self.decoders) > 0:
            tgt = src.clone() if tgt is None else tgt

            for decoder in self.decoders:
                tgt = decoder(
                    target=tgt,
                    # pyre-fixme[61]: `memory` is not always initialized here.
                    memory=memory,
                    input_mask=decoder_input_mask,
                )

            return tgt

        return None