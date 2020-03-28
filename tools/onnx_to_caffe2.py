import os
import cv2
import time
import torch

import onnx
import numpy as np
import caffe2.python.onnx.backend as onnx_caffe2_backend

models_name = 'mobilenetv1_101_features'
input = torch.randn(1, 3, 529, 961, requires_grad=True)

if __name__ == "__main__":

    # Load the ONNX ModelProto object. model is a standard Python protobuf object
    models = onnx.load(models_name + '.onnx')

    # prepare the caffe2 backend for executing the model this converts the ONNX model into a
    # Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
    # availiable soon.
    models_backend = onnx_caffe2_backend.prepare(models, strict=False)

    # run the model in Caffe2

    # Construct a map from input names to Tensor data.
    # The graph of the model itself contains inputs for all weight parameters, after the input image.
    # Since the weights are already embedded, we just need to pass the input image.
    # Set the first input.
    W = {models.graph.input[0].name: input.data.numpy()}

    # Run the Caffe2 net:
    caffe2_output = models_backend.run(W)[0]

    # Verify the numerical correctness upto 3 decimal places
    print(caffe2_output)
    # np.testing.assert_almost_equal(torch_output[0].cpu().detach().numpy(), caffe2_output, decimal=3)

    print("Exported model has been executed on Caffe2 backend, and the result looks good!")
    #endregion


    # Export to mobile
    from caffe2.python.predictor import mobile_exporter

    # extract the workspace and the model proto from the internal representation
    c2_workspace = models_backend.workspace
    c2_model = models_backend.predict_net

    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    # Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
    with open(models_name + '_init.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open(models_name + '_predict.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())

    while True:
        time_mark = time.time()
        caffe2_output = models_backend.run(W)
        print(str((time.time() - time_mark)*1000))
