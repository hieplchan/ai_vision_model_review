import sys
sys.path.append('..')

from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import time

from utils import OPENVINO_MODELS_DIR

CHECKPOINT_NAME = 'mobilenetv3_small'
model_xml = OPENVINO_MODELS_DIR + '/' + CHECKPOINT_NAME + '/' + CHECKPOINT_NAME +'.xml'
model_bin = OPENVINO_MODELS_DIR + '/' + CHECKPOINT_NAME + '/' + CHECKPOINT_NAME + '.bin'

plugin = IEPlugin(device='CPU')
net = IENetwork(model=model_xml, weights=model_bin)
n, c, h, w = net.inputs['0'].shape
exec_net = plugin.load(network = net, num_requests = 8)
out_key = ''
for key in net.outputs:
    out_key = key

def models_inspect():
    print('-------------- MODELS INSPECTION --------------')
    print('Plugin version: ' + plugin.version)

    print('Input shape: ' + str(net.inputs['0'].shape))
    print('Input precision: ' + str(net.inputs['0'].precision))
    print('Input layout: ' + str(net.inputs['0'].layout))

    print('Batch size: ' + str(net.batch_size))

    # print('Output shape: ' + str(net.outputs['749'].shape))
    print('Output shape: ' + str(net.outputs[out_key].shape))

    print('-------------- PERFORMANCE MEASURES --------------')

    # Queries performance measures per layer to get feedback of what is the most time consuming layer
    input = np.random.rand(2, 3, 224, 224)
    exec_net.requests[0].infer({'0': input})
    perf_counts = exec_net.requests[0].get_perf_counts()
    print("{:<30} {:<15} {:<30} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
    for layer, stats in perf_counts.items():
        print("{:<30} {:<15} {:<30} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                          stats['status'], stats['real_time']))

    print('-----------------------------------------------')

if __name__ == "__main__":
    # models_inspect()

    input = np.random.rand(2, 3, 224, 224)
    while True:
        start_time_mark = time.time()

        exec_net.requests[0].infer({'0': input})

        end_time_mark = time.time()
        process_time = (end_time_mark - start_time_mark)*1000

        if (process_time > 0.1):
            print(str(process_time) + ',0.0')
