import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder_lane)
        init.initialize_decoder(self.decoder_depth)
        init.initialize_head(self.segmentation_head_lane)
        init.initialize_head(self.segmentation_head_depth)
        if self.classification_head_lane is not None:
            if self.classification_head_depth is not None:
                init.initialize_head(self.classification_head_lane)
                init.initialize_head(self.classification_head_depth)


    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
		
		# lane branch
        decoder_output_lane = self.decoder_lane(*features)
        masks_lane = self.segmentation_head_lane(decoder_output_lane)
				
		#depth branch
        decoder_output_depth = self.decoder_depth(*features)
        masks_depth = self.segmentation_head_depth(decoder_output_depth)

		#classification head
        if self.classification_head_lane is not None:
            if self.classification_head_depth is not None:
                labels_lane = self.classification_head_lane(features[-1])
                labels_depth = self.classification_head_depth(features[-1])
                return masks_lane, labels_lane, masks_depth, labels_depth

        return masks_lane, masks_depth

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x, y = self.forward(x)

        return x, y
