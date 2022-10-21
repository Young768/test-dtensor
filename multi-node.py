import tensorflow as tf
from tensorflow.experimental import dtensor
import os

rank=int(os.environ['DTENSOR_CLIENT_ID'])
size=int(os.environ['DTENSOR_NUM_CLIENTS'])

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[rank], 'GPU')
visible_devices = tf.config.experimental.get_visible_devices()

tf.experimental.dtensor.initialize_multi_client(enable_coordination_service=True)
mesh_1d = dtensor.create_distributed_mesh([('x', size)], device_type='GPU')
print('mesh_1d',mesh_1d)
layout = dtensor.Layout(['x', dtensor.UNSHARDED], mesh_1d)

local_component = tf.range(rank*2, 2+rank*2)
local_component = tf.reshape(local_component, [1, 2])
print('local_component', local_component)

my_dtensor = dtensor.pack([local_component],layout)
print('my_dtensor', my_dtensor)

fully_replicated_layout = dtensor.Layout([dtensor.UNSHARDED, 'x'], mesh_1d)
replicated_dtensor = dtensor.relayout(my_dtensor, fully_replicated_layout)
#print('fully replicated', replicated_dtensor)