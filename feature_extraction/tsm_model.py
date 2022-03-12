import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def get_tsm_model(batch_size=128):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])

    tf.disable_eager_execution()
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    
    model = hub.Module("https://tfhub.dev/deepmind/mmv/tsm-resnet50/1")
    input_frames = tf.placeholder(tf.float32, shape=(1, batch_size, 200, 200, 3))
    _ = model(input_frames, signature="video", as_dict=True)

    graph = tf.get_default_graph()
    last_conv = graph.get_tensor_by_name(
        "module_apply_video/text_audio_video_2/visual_module_1/resnet_v2_50_inception_preprocessed/postnorm/Relu:0")
    sess = tf.train.MonitoredSession()

    def fc(video):
        outputs = []
        for i in range(0, len(video), batch_size):
            batch = video[i:i + batch_size]
            if len(batch) % batch_size != 0:
                batch = np.concatenate([batch] + [batch[-1:]] * (batch_size - len(batch) % batch_size), 0)
            batch = batch.astype(np.float32) / 255.

            out = sess.run(last_conv, feed_dict={input_frames: batch[np.newaxis]})
            outputs.append(out)

        # /2 to compensate for dropout probably?
        outputs = np.concatenate(outputs, 0)[:len(video)] / 2
        return outputs.mean(axis=(1, 2))
    return fc
