该仓库是基于Paddle的custom op
使用方式：
1:编译
cd custom_op
python setup_cuda.py install 
2:使用
import paddle
from custom_setup_ops import custom_relu

x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_relu(x)
