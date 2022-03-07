import os
import sys
import pprint

import logging
from config import cfg
import utils.distributed as dist

'''
Python logging模块
主要包括四部分：
Loggers: 可供程序直接调用的接口，app通过调用提供的api来记录日志
Handlers: 决定将日志记录分配至正确的目的地
Filters:对日志信息进行过滤， 提供更细粒度的日志是否输出的判断
Formatters: 制定最终记录打印的格式布局
'''


def setup_default_logging():
    # 使用工厂方法返回一个Logger实例
    logger = logging.getLogger(cfg.logger_name)
    # 设置日志的级别
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    # 设置日志的级别
    ch.setLevel(logging.INFO)
    # 设置打印格式
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(cfg.root_dir, cfg.logger_name + '.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))

    return logger


def logger_info(data):
    # 判断当前线程是否是主线程
    if dist.is_master_proc():
        logger = logging.getLogger(cfg.logger_name)
        logger.info(data)

            