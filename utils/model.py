import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.layers import (GlobalAveragePooling2D,
                                     GlobalMaxPooling2D,
                                     Dense)
from tensorflow.keras.layers import (
    Reshape,
    BatchNormalization,
    Multiply,
)


# 4 Layers U-Net with SE and CBAM attention mechanism
# CBAM: Convolutional Block Attention Module
def spatial_attention_block(inputs):
    """
    Spatial Attention Module:
    This module applies spatial attention by aggregating feature maps
    across channels
    and generating a spatial attention map that highlights important regions.

    :param inputs: Input feature map (tensor)
    :return: Feature map refined with spatial attention
    """

    # Calculate the average and maximum poolings of the input feature map
    max_pool = GlobalMaxPooling2D(keepdims=True)(inputs)
    avg_pool = GlobalAveragePooling2D(keepdims=True)(inputs)

    # Concatenate pooling results along
    # the channel dimension and apply a 1x1 convolution
    concat = Concatenate(axis=-1)([max_pool, avg_pool])
    attention_map = Conv2D(1, kernel_size=7, padding="same",
                           activation="sigmoid")(
                            concat
                            )

    # Multiply the attention map with the input feature map to refine
    # important spatial locations
    return Multiply()([inputs, attention_map])


# Defining U-Net model classes
class UNet:
    """
    U-Net with Attention Mechanism:
    A modified U-Net model incorporating channel
    and spatial attention mechanisms.

    :param input_shape: Shape of the input image (e.g., (384, 384, 3))
    :param num_filters: Number of convolutional filters for the
                        first convolution layer,
                        which doubles in subsequent layers.
    """

    def __init__(self, input_shape, num_filters):
        """
        Initialise model parameters
        :param input_shape: shape of the input data, e.g. (384, 384, 3)
        :param num_filters: number of convolution kernels
                            for the first
        convolution layer, doubled for subsequent layers
        """
        self.input_shape = input_shape
        self.num_filters = num_filters

    def build_model(self):
        """
        Constructs the U-Net model with an attention mechanism.

        :return: Compiled U-Net model
        """
        # Input layer
        inputs = Input(self.input_shape)

        # Encoder (Downsampling path)
        c1, p1 = self._conv_block(inputs, self.num_filters)  # First block
        c2, p2 = self._conv_block(p1, self.num_filters * 2)  # Second block
        c3, p3 = self._conv_block(p2, self.num_filters * 4)  # Third block
        c4, p4 = self._conv_block(p3, self.num_filters * 8)  # Fourth block

        # Bottleneck (Deepest part of U-Net)
        # bottleneck
        c5 = self._conv_block(p4, self.num_filters * 16, pool=False)

        # Decoder (Upsampling path with attention mechanism)
        # Upsampling and concatenation of the encoded feature maps
        # First decoding block
        u1 = UpSampling2D((2, 2))(c5)
        c4_attention = self._attention_block(c4)  # Apply attention mechanism
        u1 = Concatenate()([u1, c4_attention])
        c6 = self._conv_block(u1, self.num_filters * 8, pool=False)

        # Second decoding block
        u2 = UpSampling2D((2, 2))(c6)
        c3_attention = self._attention_block(c3)  # Apply attention mechanism
        u2 = Concatenate()([u2, c3_attention])
        c7 = self._conv_block(u2, self.num_filters * 4, pool=False)

        # Third decoding block
        u3 = UpSampling2D((2, 2))(c7)
        c2_attention = self._attention_block(c2)  # Apply attention mechanism
        u3 = Concatenate()([u3, c2_attention])
        c8 = self._conv_block(u3, self.num_filters * 2, pool=False)

        # Fourth decoding block
        u4 = UpSampling2D((2, 2))(c8)
        c1_attention = self._attention_block(c1)  # Apply attention mechanism
        u4 = Concatenate()([u4, c1_attention])
        c9 = self._conv_block(u4, self.num_filters, pool=False)

        # Output layer with a single channel and relu activation
        outputs = Conv2D(1, (1, 1), activation="relu")(c9)

        outputs = Reshape((outputs.shape[1], outputs.shape[2]))(
            outputs
        )  # Reshape output to remove the last single channel dimension

        # Build the U-Net model
        return Model(inputs, outputs)

    def _conv_block(self, inputs, filters, pool=True):
        """
        Defines a convolutional block with two convolution layers
        and an optional pooling layer.

        :param inputs: Input tensor
        :param filters: Number of convolutional filters
        :param pool: Whether to apply max pooling
        :return: Output tensor of the block
        (and the pooled tensor if pooling is enabled)
        """
        # First convolutional layer
        c = Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)

        # Batch normalization
        c = BatchNormalization()(c)

        # Second convolutional layer
        c = Conv2D(filters, (3, 3), activation="relu", padding="same")(c)

        # Batch normalization
        c = BatchNormalization()(c)

        if pool:
            p = MaxPooling2D((2, 2))(c)
            return c, p  # Return both convolution output and pooled output
        else:
            return c  # Return only convolution output

    # SE and CBAM Attention mechanism
    def _attention_block(self, inputs):
        """
        Attention Block (SE Block-like mechanism):
        Applies channel and spatial attention mechanisms
        to enhance important features.

        :param inputs: Input tensor
        :return: Feature map refined with attention
        """
        # Channel Attention using Global Average Pooling
        # Get the number of channels
        channels = tf.keras.backend.int_shape(inputs)[-1]

        # Apply Global Average Pooling
        gap = GlobalAveragePooling2D()(inputs)

        # Reduce dimensionality
        gap = Dense(channels // 4, activation="relu")(gap)

        # Restore dimensionality
        gap = Dense(channels, activation="sigmoid")(gap)

        # Reshape to match input dimensions
        gap = Reshape((1, 1, channels))(gap)

        # Apply SE attention
        channel_attention = Multiply()([inputs, gap])

        # Apply CBAM attention
        spatial_attention = spatial_attention_block(channel_attention)

        return spatial_attention
