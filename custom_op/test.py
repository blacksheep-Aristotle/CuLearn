import paddle
from custom_setup_ops import custom_relu

x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_relu(x)

