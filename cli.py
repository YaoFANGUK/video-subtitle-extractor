#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : Modified for CLI multi-file support
@FileName: cli.py
@desc: 命令行版本的字幕提取器，支持多文件处理
"""
import os
import sys
import argparse
import multiprocessing
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from typing import List, Tuple, Optional

# 添加项目根目录和backend路径到sys.path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

import config
from backend.main import SubtitleExtractor


class MultiFileSubtitleExtractor:
    """多文件字幕提取器"""
    
    def __init__(self, 
                 video_files: List[str],
                 subtitle_area: Optional[Tuple[int, int, int, int]] = None,
                 output_dir: str = "output",
                 max_workers: int = 1,
                 language: str = None,
                 mode: str = None,
                 non_interactive: bool = False,
                 use_vsf: bool = False):
        """
        初始化多文件字幕提取器
        
        Args:
            video_files: 视频文件路径列表
            subtitle_area: 字幕区域 (ymin, ymax, xmin, xmax)
            output_dir: 输出目录
            max_workers: 最大并发数
            language: 识别语言
            mode: 识别模式 (fast/accurate/auto)
            non_interactive: 非交互模式
            use_vsf: 是否使用VSF算法进行帧提取
        """
        self.video_files = [str(Path(f).resolve()) for f in video_files]
        self.subtitle_area = subtitle_area
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.non_interactive = non_interactive
        self.use_vsf = use_vsf
        self.results = []
        
        # 设置语言和模式
        if language:
            self._set_language(language)
        if mode:
            self._set_mode(mode)
            
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _set_language(self, language: str):
        """设置识别语言"""
        # 更新配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'settings.ini')
        import configparser
        settings = configparser.ConfigParser()
        if os.path.exists(config_path):
            settings.read(config_path, encoding='utf-8')
        else:
            settings['DEFAULT'] = {}
        
        settings['DEFAULT']['Language'] = language
        
        with open(config_path, 'w', encoding='utf-8') as f:
            settings.write(f)
        
        # 重新加载配置
        importlib.reload(config)
        
    def _set_mode(self, mode: str):
        """设置识别模式"""
        # 更新配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'settings.ini')
        import configparser
        settings = configparser.ConfigParser()
        if os.path.exists(config_path):
            settings.read(config_path, encoding='utf-8')
        else:
            settings['DEFAULT'] = {}
        
        settings['DEFAULT']['Mode'] = mode
        
        with open(config_path, 'w', encoding='utf-8') as f:
            settings.write(f)
        
        # 重新加载配置
        importlib.reload(config)
    
    def process_single_video(self, video_path: str) -> dict:
        video_path = Path(video_path)
        result = {
            'video_path': str(video_path),
            'video_name': video_path.stem,
            'status': 'processing',
            'start_time': time.time(),
            'error': None,
            'srt_path': None,
            'txt_path': None
        }

        try:
            print(f"\n开始处理: {video_path.name}")

            # 创建字幕提取器
            extractor = SubtitleExtractor(str(video_path), self.subtitle_area)
            
            # 设置是否使用VSF算法
            if self.use_vsf:
                extractor.use_vsf = True
                print(f"  使用VSF算法进行帧提取")

            # 运行提取（传递非交互模式参数）
            extractor.run(non_interactive=self.non_interactive)

            # 检查输出文件
            srt_path = video_path.with_suffix('.srt')
            txt_path = video_path.with_suffix('.txt')

            # 目标子文件夹
            target_dir = self.output_dir / video_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            new_srt_path = target_dir / (video_path.stem + '.srt')
            new_txt_path = target_dir / (video_path.stem + '.txt')

            # 移动到子文件夹（无论原始srt在哪里，都强制移动到目标）
            if srt_path.exists() and srt_path != new_srt_path:
                srt_path.replace(new_srt_path)
                srt_path = new_srt_path
            elif new_srt_path.exists():
                srt_path = new_srt_path  # 已经在目标位置

            if txt_path.exists() and txt_path != new_txt_path:
                txt_path.replace(new_txt_path)
                txt_path = new_txt_path
            elif new_txt_path.exists():
                txt_path = new_txt_path

            if srt_path.exists():
                result['srt_path'] = str(srt_path)
                result['status'] = 'success'
                print(f"✓ 成功: {video_path.name} -> {srt_path}")
            else:
                result['status'] = 'failed'
                result['error'] = 'SRT文件未生成'
                print(f"✗ 失败: {video_path.name} - SRT文件未生成")

            if txt_path.exists():
                result['txt_path'] = str(txt_path)

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            print(f"✗ 错误: {video_path.name} - {e}")

        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']

        return result
    
    def run(self):
        """运行多文件处理"""
        if config.USE_GPU and self.max_workers > 1:
            print("警告: 使用GPU时，max_workers >1 可能导致资源竞争。建议设置为1。")
        print(f"开始处理 {len(self.video_files)} 个视频文件")
        print(f"字幕区域: {self.subtitle_area if self.subtitle_area else '自动检测'}")
        print(f"识别语言: {config.REC_CHAR_TYPE}")
        print(f"识别模式: {config.MODE_TYPE}")
        print(f"最大并发数: {self.max_workers}")
        print("-" * 60)
        
        start_time = time.time()
        
        if self.max_workers == 1:
            # 单线程处理
            for video_path in self.video_files:
                result = self.process_single_video(video_path)
                self.results.append(result)
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_video = {
                    executor.submit(self.process_single_video, video_path): video_path 
                    for video_path in self.video_files
                }
                
                # 收集结果
                for future in as_completed(future_to_video):
                    result = future.result()
                    self.results.append(result)
        
        # 生成报告
        self._generate_report(start_time)

    def _generate_report(self, start_time: float):
        """生成处理报告"""
        total_time = time.time() - start_time
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        failed_count = len(self.results) - success_count
        
        print("\n" + "=" * 60)
        print("处理完成报告")
        print("=" * 60)
        print(f"总文件数: {len(self.video_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {total_time/len(self.video_files):.2f}秒/文件")
        
        if failed_count > 0:
            print("\n失败的文件:")
            for result in self.results:
                if result['status'] == 'failed':
                    print(f"  - {Path(result['video_path']).name}: {result['error']}")
        
        # 保存详细报告
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_files': len(self.video_files),
                    'success_count': success_count,
                    'failed_count': failed_count,
                    'total_time': total_time,
                    'average_time': total_time/len(self.video_files)
                },
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_path}")


def validate_video_files(video_files: List[str]) -> List[str]:
    """验证视频文件"""
    valid_files = []
    for file_path in video_files:
        path = Path(file_path)
        if not path.exists():
            print(f"警告: 文件不存在 - {file_path}")
            continue
        if not path.is_file():
            print(f"警告: 不是文件 - {file_path}")
            continue
        # 检查视频文件扩展名
        if path.suffix.lower() not in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
            print(f"警告: 可能不是视频文件 - {file_path}")
        valid_files.append(str(path))
    return valid_files


def parse_subtitle_area(area_str: str) -> Optional[Tuple[int, int, int, int]]:
    """解析字幕区域字符串"""
    if not area_str:
        return None
    
    try:
        parts = area_str.split(',')
        if len(parts) != 4:
            raise ValueError("字幕区域格式错误")
        
        ymin, ymax, xmin, xmax = map(int, parts)
        return (ymin, ymax, xmin, xmax)
    except Exception as e:
        print(f"字幕区域解析错误: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="视频字幕提取器 - 命令行版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python cli.py video1.mp4
  
  # 处理多个文件
  python cli.py video1.mp4 video2.mp4 video3.mp4
  
  # 使用VSF算法提高识别准确性
  python cli.py video1.mp4 --use-vsf
  
  # 指定字幕区域
  python cli.py video1.mp4 --subtitle-area "558,693,0,975"
  
  # 指定输出目录和并发数
  python cli.py *.mp4 --output-dir ./subtitles --max-workers 4
  
  # 指定语言和模式
  python cli.py *.mp4 --language ch --mode accurate
  
  # 综合使用VSF算法和其他参数
  python cli.py *.mp4 --use-vsf --no-interactive --max-workers 2
        """
    )
    
    parser.add_argument(
        'video_files',
        nargs='+',
        help='要处理的视频文件路径 (支持通配符)'
    )
    
    parser.add_argument(
        '--subtitle-area',
        type=str,
        help='字幕区域坐标 (ymin,ymax,xmin,xmax) 例如: "558,693,0,975"'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='最大并发数 (默认: 1)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        choices=['ch', 'en', 'japan', 'korea', 'chinese_cht'],
        help='识别语言 (默认: 从配置文件读取)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'accurate', 'auto'],
        help='识别模式 (默认: 从配置文件读取)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='非交互模式，跳过所有用户交互'
    )
    
    parser.add_argument(
        '--use-vsf',
        action='store_true',
        help='使用VSF (VideoSubFinder) 算法进行帧提取，提高字幕识别准确性'
    )
    
    parser.add_argument(
        '--list-languages',
        action='store_true',
        help='列出支持的语言'
    )
    
    args = parser.parse_args()
    
    # 列出支持的语言
    if args.list_languages:
        print("支持的语言:")
        print("  ch - 简体中文")
        print("  en - 英文")
        print("  japan - 日文")
        print("  korea - 韩文")
        print("  chinese_cht - 繁体中文")
        return
    
    # 处理通配符
    import glob
    video_files = []
    for pattern in args.video_files:
        print(f"处理路径/模式: {pattern}")  # 添加日志
        matched = glob.glob(pattern)
        if not matched and Path(pattern).exists():  # 如果不是模式但文件存在，直接添加
            matched = [pattern]
        video_files.extend(matched)
        if matched:
            print(f"  匹配到 {len(matched)} 个文件")
        else:
            print(f"  警告: 未匹配到文件")
    
    if not video_files:
        print("错误: 没有找到匹配的视频文件")
        return
    
    # 验证文件
    valid_files = validate_video_files(video_files)
    if not valid_files:
        print("错误: 没有有效的视频文件")
        return
    
    # 解析字幕区域
    subtitle_area = parse_subtitle_area(args.subtitle_area)
    
    # 设置多进程启动方法
    multiprocessing.set_start_method("spawn", force=True)
    
    # 创建提取器并运行
    extractor = MultiFileSubtitleExtractor(
        video_files=valid_files,
        subtitle_area=subtitle_area,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        language=args.language,
        mode=args.mode,
        non_interactive=args.no_interactive,
        use_vsf=args.use_vsf
    )
    
    try:
        extractor.run()
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()