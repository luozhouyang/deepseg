import abc


class AbstractModel(abc.ABC):
    """Model interface."""

    def input_fn(self, params):
        """Input fn for estimator.

        Args:
            params: A dict, storing hyper params

        Returns:
            A instance of tf.data.Dataset.
        """
        raise NotImplementedError()

    def model_fn(self, features, labels, mode, params, config):
        """Build model fn for estimator.

        Args:
            features: Features data
            labels: Labels data
            mode: Mode
            params: Hyper params
            config: Run config
        """
        raise NotImplementedError()

    def decode(self, output, nwords, params):
        """Decode the outputs of BiLSTM.

        Args:
            output: A tensor, output of BiLSTM
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            Logits and predict ids.
        """
        raise NotImplementedError()

    def compute_loss(self, logits, labels, nwords, params):
        """Compute loss.

        Args:
            logits: A tensor, logits from decode
            labels: A tensor, the ground truth label
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            The loss of this network.
        """
        raise NotImplementedError()

    def build_predictions(self, predict_ids, params):
        """Build predictions for predict mode.

        Args:
            predict_ids: A tensor, predict ids from decode.
            params: A dict, storing hyper params

        Returns:
            A dict, storing predictions you want
        """
        raise NotImplementedError()

    def build_eval_metrics(self, predict_ids, labels, nwords, params):
        """Build evaluation ops for eval mode.

        Args:
            predict_ids: A tensor, predict ids from decode
            labels: A tensor, the ground truth labels
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            A metrics op for eval mode.
        """
        raise NotImplementedError()

    def build_train_op(self, loss, params):
        """Build train op for train mode.

        Args:
            loss: A tensor, loss of this network
            params: A dict, storing hyper params

        Returns:
            A train op for train mode.
        """
        raise NotImplementedError()
