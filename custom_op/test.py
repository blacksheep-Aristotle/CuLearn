import paddle
from custom_setup_op_test import *

def string_to_callable(func_path):
    """
    支持更复杂的模块路径，如 "paddle.nn.functional.relu" , "custom_setup_op_test.custom_relu"
    """
    if func_path is None:
        return None
    parts = func_path.split('.')
    
    # 尝试逐级导入
    for i in range(len(parts)-1, 0, -1):
        module_path = '.'.join(parts[:i])
        func_name = parts[i]
        
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            
            # 检查剩余部分是否是属性
            for attr_name in parts[i+1:]:
                func = getattr(func, attr_name)
                
            return func
        except (ImportError, AttributeError):
            continue
    
    raise ValueError(f"无法解析函数路径: {func_path}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CUDA算子性能对比基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python benchmark.py custom_relu paddle.relu
  python benchmark.py my_ops.custom_conv paddle.nn.functional.conv2d --input-shapes 1,64,224,224 1,128,112,112
  python benchmark.py torch.relu paddle.relu --warmup 5 --runs 50 --output report.xlsx
        """
    )
    
    parser.add_argument('custom_op', default=None,help='自定义算子路径 (如: custom_ops.my_relu)')
    parser.add_argument('reference_op', default=None,help='参考算子路径 (如: paddle.relu, torch.relu)')
    
    parser.add_argument('--input-shapes', nargs='+', 
                       default=['1024', '2048', '4096', '8192'],
                       help='输入形状列表 (默认: 1024 2048 4096 8192)')
    
    parser.add_argument('--warmup', type=int, default=10,
                       help='预热运行次数 (默认: 10)')
    
    parser.add_argument('--runs', type=int, default=100,
                       help='基准测试运行次数 (默认: 100)')
    
    parser.add_argument('--output', type=str, 
                       help='输出文件路径 (Excel格式)')
    
    parser.add_argument('--plot', type=str,
                       help='保存图表文件路径')
    
    parser.add_argument('--check_correctness', type=bool,
                        default=False,
                       help='保存图表文件路径')
    
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='运行设备 (默认: cuda)')
    
    return parser.parse_args()

def benchmark_op(op_name,reference_op_name,input_shapes,config):
    """使用示例"""
    
    # 创建测试运行器
    runner = BenchmarkRunner()
    
    # 基础配置
    test_op_func = string_to_callable(op_name)
    reference_op_func = string_to_callable(reference_op_name)
    if reference_op_func is None:
        # 测试单个算子
        print("🧪 测试单个CUDA算子")
        result = runner.benchmark.benchmark_operator(
            op_func=your_custom_op,
            input_shapes=input_shapes,  
            config=config,
            reference_func=reference_op_func,
            op_name=op_name
        )
    
    # # 运行缩放测试
    # scaling_results = runner.run_scaling_test(
    #     op_func=your_custom_op,
    #     base_shapes=[(256, 256), (256, 256)],
    #     scale_factors=[0.5, 1.0, 2.0, 4.0],
    #     op_name="my_custom_cuda_op"
    # )
    # 比较多个算子（如果有多个实现）
    operators_to_compare = {
        op_name : test_op_func,
        reference_op_name : reference_op_func, 
    }

    comparison = runner.compare_operators(
        operators=operators_to_compare,
        input_shapes=input_shapes,
        config=config
    )
    
    # # 生成报告和图表
    # runner.generate_report("benchmark_report.csv")
    # runner.plot_results("benchmark_results.png")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA不可用，切换到CPU模式")
        args.device = 'cpu'
    
    print(f"🚀 开始基准测试: {args.custom_op} vs {args.reference_op}")
    print(f"📟 设备: {args.device}")
    print(f"📊 输入形状: {args.input_shapes}")
    print(f"⚙️  预热: {args.warmup} 次, 测试: {args.runs} 次")
    
    # 解析输入形状
    input_shapes = []
    for shape_str in args.input_shapes:
        try:
            if ',' in shape_str:
                # 多维形状，如 "1,64,224,224"
                shape = tuple(map(int, shape_str.split(',')))
            else:
                # 一维形状，如 "1024"
                shape = (int(shape_str),)
            input_shapes.append(shape)

    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        sync_cuda=True,
        check_correctness=args.check_correctness,
        memory_stats=False
    )
    benchmark_op(args.custom_op,args.reference_op,input_shapes,config)