import sys
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print("GPUs found:", gpus)
        if gpus:
            for i,g in enumerate(gpus):
                print(f" GPU[{i}] name:", g.name if hasattr(g,'name') else g)
    except Exception as e:
        print("Error listing GPUs:", repr(e))
except Exception as e:
    print("TensorFlow import failed:", repr(e))
    sys.exit(1)
