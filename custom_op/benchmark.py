from log import logger
from timer import get_timers
import paddle
import time
import numpy as np
from typing import Dict, List, Callable, Optional
import warnings
import gc
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    warmup_runs: int = 10
    benchmark_runs: int = 100
    sync_cuda: bool = True
    check_correctness: bool = True
    memory_stats: bool = True

class CUDABenchmark:
    """CUDA算子基准测试类"""
    
    def __init__(self, device: str = "cuda", dtype: paddle.dtype = paddle.float32):
        self.device = device
        self.dtype = dtype
        self.results = {}
        self.timers = get_timers()
        
    def _sync_if_needed(self):
        """如果需要，同步CUDA设备"""
        if paddle.cuda.is_available():
            paddle.cuda.synchronize(self.device)
    
    def _generate_test_inputs(self, input_shapes, low=-1.0, high=1.0):
        """生成测试输入张量"""
        inputs = []
        for shape in input_shapes:
            if isinstance(shape, tuple):
                tensor = paddle.rand(*shape, device=self.device, dtype=self.dtype) * (high - low) + low
            else:
                # 假设是单个整数，创建1D张量
                tensor = paddle.rand(shape, device=self.device, dtype=self.dtype) * (high - low) + low
            inputs.append(tensor)
        return inputs
    
    def _get_memory_stats(self):
        """获取内存统计"""
        if paddle.cuda.is_available():
            return {
                'allocated': paddle.cuda.memory_allocated(self.device),
                'reserved': paddle.cuda.memory_reserved(self.device),
                'max_allocated': paddle.cuda.max_memory_allocated(self.device)
            }
        return {}
    
    def benchmark_operator(
        self,
        op_func: Callable,
        input_shapes: List,
        config: BenchmarkConfig = BenchmarkConfig(),
        reference_func: Optional[Callable] = None,
        op_name: str = "custom_op"
    ) -> Dict:
        """
        基准测试单个算子
        
        Args:
            op_func: 要测试的算子函数
            input_shapes: 输入张量形状列表
            config: 基准测试配置
            reference_func: 参考实现（用于正确性检查）
            op_name: 算子名称
            
        Returns:
            包含性能指标的字典
        """
        print(f"🏃‍♂️ 开始基准测试: {op_name}")
        
        # 清理内存
        if paddle.cuda.is_available():
            paddle.cuda.empty_cache()
        
        # 生成测试输入
        inputs = self._generate_test_inputs(input_shapes)
        
        # 预热运行
        print(f"  🔥 预热运行 ({config.warmup_runs} 次)...")
        for _ in range(config.warmup_runs):
            outputs = op_func(*inputs)
            if config.sync_cuda:
                self._sync_if_needed()
        
        # 正确性检查（如果有参考实现）
        correctness_passed = True
        if config.check_correctness and reference_func is not None:
            print("  ✅ 运行正确性检查...")
            try:
                ref_outputs = reference_func(*[x.cpu() for x in inputs])
                test_outputs = op_func(*inputs)
                if config.sync_cuda:
                    self._sync_if_needed()
                
                # 转换到CPU比较
                test_outputs_cpu = test_outputs.cpu() if test_outputs.is_cuda else test_outputs
                
                # 检查数值接近程度
                if isinstance(test_outputs_cpu, torch.Tensor):
                    diff = torch.abs(test_outputs_cpu - ref_outputs)
                    max_diff = diff.max().item()
                    avg_diff = diff.mean().item()
                    
                    correctness_passed = max_diff < 1e-4
                    print(f"    最大差异: {max_diff:.6f}, 平均差异: {avg_diff:.6f}")
                    
                    if not correctness_passed:
                        warnings.warn(f"正确性检查失败! 最大差异: {max_diff}")
            except Exception as e:
                print(f"     ⚠️ 正确性检查出错: {e}")
                correctness_passed = False
        
        # 基准测试运行
        print(f"   ⏱️ 基准测试运行 ({config.benchmark_runs} 次)...")
        timings = []
        memory_stats = []
        
        for i in range(config.benchmark_runs):
            # 记录内存使用前状态
            if config.memory_stats:
                memory_before = self._get_memory_stats()
            
            # 计时开始
            self.timers("read-data").start()
            
            # 运行算子
            outputs = op_func(*inputs)
            
            # 同步设备
            if config.sync_cuda:
                self._sync_if_needed()
            
            # 计时结束
            self.timers("read-data").stop()
            
            # 记录内存使用后状态
            if config.memory_stats:
                memory_after = self._get_memory_stats()
                memory_stats.append({
                    'allocated_diff': memory_after['allocated'] - memory_before['allocated'],
                    'iteration': i
                })
        
        # 统计分析
        timer_info = self.timers.log(self.timers.timers.keys(), reset=True)
        
        if config.memory_stats and memory_stats:
            memory_diffs = [m['allocated_diff'] for m in memory_stats]
            stats['memory_allocated_avg'] = np.mean(memory_diffs)
            stats['memory_allocated_max'] = np.max(memory_diffs)
            stats['timer_info'] = timer_info
        
        self.results[op_name] = stats

        return stats

class BenchmarkRunner:
    """批量基准测试运行器"""
    
    def __init__(self):
        self.benchmark = CUDABenchmark()
        self.comparison_results = []
    
    def run_scaling_test(
        self,
        op_func: Callable,
        base_shapes: List,
        scale_factors: List[float],
        op_name: str = "custom_op"
    ):
        """运行缩放测试（不同输入大小）"""
        print(f"\n📈 运行缩放测试: {op_name}")
        print("=" * 60)
        
        scaling_results = []
        
        for scale in scale_factors:
            # 缩放输入形状
            scaled_shapes = []
            for shape in base_shapes:
                if isinstance(shape, tuple):
                    scaled_shape = tuple(int(dim * scale) for dim in shape)
                else:
                    scaled_shape = int(shape * scale)
                scaled_shapes.append(scaled_shape)
            
            print(f"  缩放因子: {scale:.1f}, 输入形状: {scaled_shapes}")
            
            # 运行基准测试
            result = self.benchmark.benchmark_operator(
                op_func, scaled_shapes, op_name=f"{op_name}_scale_{scale}"
            )
            
            scaling_results.append({
                'scale_factor': scale,
                'input_shapes': scaled_shapes,
                **result
            })
        
        return scaling_results
    
    def compare_operators(
        self,
        operators: Dict[str, Callable],
        input_shapes: List,
        config: BenchmarkConfig = BenchmarkConfig()
    ):
        """比较多个算子的性能"""
        print(f"\n⚖️ 比较算子性能")
        print("=" * 60)
        
        comparison = []
        
        for op_name, op_func in operators.items():
            print(f"\n测试算子: {op_name}")
            result = self.benchmark.benchmark_operator(
                op_func, input_shapes, config=config, op_name=op_name
            )
            comparison.append(result)
            
            # 打印简要结果
            logger.info(f"Profile op_name {op_name} : {result}")
        
        self.comparison_results = comparison
        return comparison

    def generate_report(self, filename: Optional[str] = None):
        """生成详细测试报告"""
        if not self.benchmark.results:
            print("没有测试结果可报告")
            return
        
        print("\n" + "="*80)
        print("📊 基准测试报告")
        print("="*80)
        
        # 创建数据框用于显示
        report_data = []
        for op_name, result in self.benchmark.results.items():
            report_data.append({
                'Operator': op_name,
                'Input Shapes': str(result['input_shapes']),
                'Mean (ms)': f"{result['mean_ms']:.4f}",
                'Std (ms)': f"{result['std_ms']:.4f}",
                'Min (ms)': f"{result['min_ms']:.4f}",
                'Max (ms)': f"{result['max_ms']:.4f}",
                'Throughput (ops/sec)': f"{result['throughput']:.2f}",
                'P95 (ms)': f"{result['p95_ms']:.4f}",
                'Correctness': '✅' if result.get('correctness_passed', True) else '❌'
            })
        
        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))
        
        # 保存到文件
        if filename:
            df.to_csv(filename, index=False)
            print(f"\n报告已保存到: {filename}")
        
        return df
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制测试结果图表"""
        if not self.benchmark.results:
            print("没有测试结果可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 平均时间比较
        operators = list(self.benchmark.results.keys())
        means = [self.benchmark.results[op]['mean_ms'] for op in operators]
        
        bars = ax1.bar(operators, means)
        ax1.set_title('平均执行时间比较')
        ax1.set_ylabel('时间 (ms)')
        ax1.bar_label(bars, fmt='%.3f')
        
        # 2. 吞吐量比较
        throughputs = [self.benchmark.results[op]['throughput'] for op in operators]
        bars = ax2.bar(operators, throughputs)
        ax2.set_title('吞吐量比较')
        ax2.set_ylabel('操作/秒')
        ax2.bar_label(bars, fmt='%.0f')
        
        # 3. 时间分布（箱线图）
        timing_data = [self.benchmark.results[op]['timings_ms'] for op in operators]
        ax3.boxplot(timing_data, labels=operators)
        ax3.set_title('执行时间分布')
        ax3.set_ylabel('时间 (ms)')
        
        # 4. 百分位数比较
        percentiles_data = {
            'P50': [self.benchmark.results[op]['p50_ms'] for op in operators],
            'P95': [self.benchmark.results[op]['p95_ms'] for op in operators],
            'P99': [self.benchmark.results[op]['p99_ms'] for op in operators]
        }
        
        x = np.arange(len(operators))
        width = 0.25
        multiplier = 0
        
        for attr, measurement in percentiles_data.items():
            offset = width * multiplier
            rects = ax4.bar(x + offset, measurement, width, label=attr)
            ax4.bar_label(rects, fmt='%.3f', padding=3, fontsize=8)
            multiplier += 1
        
        ax4.set_title('百分位数比较')
        ax4.set_ylabel('时间 (ms)')
        ax4.set_xticks(x + width, operators)
        ax4.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
