import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.models import resnet50
import torch 
def quantize_model_ptq(model_to_be_quantized,test_loader):
    fp32_model = model_to_be_quantized.eval()
    model = copy.deepcopy(fp32_model)
    model = model.to('cpu')
    # `qconfig` means quantization configuration, it specifies how should we
    # observe the activation and weight of an operator
    # `qconfig_dict`, specifies the `qconfig` for each operator in the model
    # we can specify `qconfig` for certain types of modules
    # we can specify `qconfig` for a specific submodule in the model
    # we can specify `qconfig` for some functioanl calls in the model
    # we can also set `qconfig` to None to skip quantization for some operators
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    example_inputs = (torch.randn(1, 3, 224, 224),)
    # example_inputs = example_inputs.to('cpu')
    # `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
    model_prepared = prepare_fx(model, qconfig_dict,example_inputs)
    # calibration runs the model with some sample data, which allows observers to record the statistics of
    # the activation and weigths of the operators
    calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
    length_test_dataset= len(test_loader)
    for i,(data, target) in enumerate(test_loader):  
        data =data.to('cpu')  
        model_prepared(data)
        if i>0.1*length_test_dataset:
            break #use only 10% of the test to calibrate
    # `convert_fx` converts a calibrated model to a quantized model, this includes inserting
    # quantize, dequantize operators to the model and swap floating point operators with quantized operators
    model_quantized = convert_fx(copy.deepcopy(model_prepared))
    return model_quantized