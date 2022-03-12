import os
import numpy as np
import mxnet as mx
from collections import namedtuple


def get_resnext_fc(network_path, network_epoch, input_height=224, input_width=224, batch_size=16):
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    batch_def = namedtuple("Batch", ["data"])
    sym, arg_params, aux_params = mx.model.load_checkpoint(network_path, network_epoch)

    network = mx.mod.Module(symbol=sym.get_internals()["flatten0_output"], label_names=None, context=mx.gpu())
    network.bind(for_training=False, data_shapes=[("data", (1, 3, input_height, input_width))])
    network.set_params(arg_params, aux_params)

    def fc(video):
        outputs = []
        for i in range(0, len(video), batch_size):
            batch = video[i:i + batch_size]
            batch = batch.astype(np.float32) - np.array([[[[123.68, 116.779, 103.939]]]], dtype=np.float32)
            batch = np.transpose(batch, [0, 3, 1, 2])
            inputs = batch_def([mx.nd.array(batch)])

            network.forward(inputs)
            output = network.get_outputs()[0].asnumpy()
            outputs.append(output)
        return np.concatenate(outputs, 0)

    return fc
