import torch


def test_run(net, x):
    y = net(x)
    print("input: ", x.shape)
    print("output: ", y.shape)


def export_jit_script(net, root='.'):
    net_script = torch.jit.script(net)
    torch.jit.save(net_script, f'{root}/{net.__class__.__name__}.jit.script')
    torch.jit.load(f'{root}/{net.__class__.__name__}.jit.script')


def export_onnx(net, x, root='.'):
    torch.onnx.export(net,
                      x,
                      f"{root}/{net.__class__.__name__}.onnx",  # 输出的 ONNX 文件名
                      export_params=True,  # 存储训练参数
                      opset_version=14,  # 指定 ONNX 操作集版本
                      do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入张量的名称
                      output_names=['output'],  # 输出张量的名称
                      dynamic_axes={'input': {0: 'batch_size'},  # 可变维度的字典
                                    'output': {0: 'batch_size'}})


def statistics(net, no_batch_input_shape):
    from ptflops import get_model_complexity_info
    res = get_model_complexity_info(net, no_batch_input_shape)
    print(res)
