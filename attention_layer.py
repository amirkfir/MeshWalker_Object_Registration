import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

@register_keras_serializable(package="Custom", name="Attention")
class Attention(Layer):
    def __init__(
        self,
        use_scale=False,
        score_mode="dot",
        dropout=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_scale = use_scale
        self.score_mode = score_mode
        self.dropout = dropout
        self.seed = seed
        self.dropout_layer = tf.keras.layers.Dropout(dropout, seed=seed)

        if self.score_mode not in ["dot", "concat"]:
            raise ValueError(
                "Invalid value for argument score_mode. "
                "Expected one of {'dot', 'concat'}. "
                f"Received: score_mode={score_mode}"
            )

    def build(self, input_shape):
        self.scale = None
        self.concat_score_weight = None
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        if self.score_mode == "concat":
            self.concat_score_weight = self.add_weight(
                name="concat_score_weight",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        super().build(input_shape)

    def _calculate_scores(self, query, key):
        if self.score_mode == "dot":
            scores = tf.matmul(query, key, transpose_b=True)
            if self.scale is not None:
                scores *= self.scale
        elif self.score_mode == "concat":
            q_reshaped = tf.expand_dims(query, axis=-2)
            k_reshaped = tf.expand_dims(key, axis=-3)
            concat = q_reshaped + k_reshaped
            if self.scale is not None:
                concat *= self.scale
            scores = self.concat_score_weight * tf.reduce_sum(tf.tanh(concat), axis=-1)
        return scores

    def _apply_scores(self, scores, value, scores_mask=None, training=False):
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            max_val = tf.constant(65504.0 if scores.dtype == tf.float16 else 1.0e9, dtype=scores.dtype)
            scores -= max_val * tf.cast(padding_mask, dtype=scores.dtype)

        weights = tf.nn.softmax(scores, axis=-1)
        if training and self.dropout > 0:
            weights = self.dropout_layer(weights, training=training)

        return tf.matmul(weights, value), weights

    def _calculate_score_mask(self, scores, v_mask, use_causal_mask):
        if use_causal_mask:
            shape = tf.shape(scores)
            Tq, Tv = shape[-2], shape[-1]
            causal_mask = tf.linalg.band_part(tf.ones((Tq, Tv), dtype=tf.bool), -1, 0)
            causal_mask = tf.reshape(causal_mask, (1, Tq, Tv))
            if v_mask is not None:
                v_mask = tf.expand_dims(v_mask, axis=-2)
                return tf.logical_and(v_mask, causal_mask)
            return causal_mask
        else:
            return v_mask

    def call(
        self,
        inputs,
        mask=None,
        training=False,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        self._validate_inputs(inputs, mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None

        scores = self._calculate_scores(q, k)
        scores_mask = self._calculate_score_mask(scores, v_mask, use_causal_mask)
        result, attention_scores = self._apply_scores(scores, v, scores_mask, training)

        if q_mask is not None:
            q_mask = tf.expand_dims(tf.cast(q_mask, dtype=result.dtype), axis=-1)
            result *= q_mask

        if return_attention_scores:
            return result, attention_scores
        return result

    def compute_mask(self, inputs, mask=None):
        if mask is None or mask[0] is None:
            return None
        return tf.convert_to_tensor(mask[0])

    def compute_output_shape(self, input_shape):
        return (*input_shape[0][:-1], input_shape[1][-1])

    def _validate_inputs(self, inputs, mask=None):
        if not isinstance(inputs, list):
            raise ValueError("Attention expects a list of inputs (query, value[, key])")
        if len(inputs) not in [2, 3]:
            raise ValueError("Attention input list must have 2 or 3 tensors")
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError("Attention expects a list of masks (query_mask, value_mask)")
            if len(mask) < 2 or len(mask) > 3:
                raise ValueError("Mask list must have length 2 or 3")

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_scale": self.use_scale,
            "score_mode": self.score_mode,
            "dropout": self.dropout,
            "seed": self.seed,
        })
        return config
