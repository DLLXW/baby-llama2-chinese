import logging
from os.path import dirname, abspath
import os
import colorlog 
import time

# 自定义日志格式
class Logger(object):
    def __init__(self, logger_name: str, level=logging.DEBUG, std_out: bool=True, save2file: bool=False, file_name: str=None) ->None:
        super().__init__()

        if std_out == False and save2file == False:
            raise ValueError('args: [std_out, save2file], at less one of them must be True')

        # 默认的格式化
        datefmt = "%Y-%m-%d %H:%M:%S"
        
        # 输出到控制台
        if std_out:
            
            std_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(log_color)s%(message)s"

            self.stdout_logger = logging.getLogger('{}_std'.format(logger_name))
            self.stdout_logger.setLevel(level)

             # 彩色输出格式化
            log_colors_config = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }
            formatter = colorlog.ColoredFormatter(
                        fmt=std_logfmt,
                        datefmt=datefmt,
                        log_colors=log_colors_config,
                        )
            
            sh = logging.StreamHandler()
            sh.setLevel(level)        
            sh.setFormatter(formatter)
            
            self.stdout_logger.addHandler(sh)
       
                    
         # 输出到文件
        if save2file:

            file_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s"

            self.file_logger = logging.getLogger('{}_file'.format(logger_name))
            self.file_logger.setLevel(level)

            base_dir ='./logs' # 获取上级目录的绝对路径
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            
            log_file = ''
            if file_name is not None:
                log_file = file_name
            else:
                log_file = base_dir + '/' + logger_name  + '-' + str(time.strftime('%Y%m%d', time.localtime())) +'.log'

            fh = logging.FileHandler(filename=log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            save_formatter =  logging.Formatter(
                fmt=file_logfmt,
                datefmt=datefmt,
                )
            fh.setFormatter(save_formatter)
            self.file_logger.addHandler(fh)

    def info(self, message: str, std_out: bool=True, save_to_file: bool=False) -> None:
        if std_out:
            self.stdout_logger.info(message)
        if save_to_file:
            self.file_logger.info(message)

    def debug(self, message: str, std_out: bool=True, save_to_file: bool=False) -> None:
        if std_out:
            self.stdout_logger.debug(message)
        if save_to_file:
            self.file_logger.debug(message)

    def warning(self, message: str, std_out: bool=True, save_to_file: bool=False) -> None:
        if std_out:
            self.stdout_logger.warning(message)
        if save_to_file:
            self.file_logger.warning(message)

    def error(self, message: str, std_out: bool=True, save_to_file: bool=False) -> None:
        if std_out:
            self.stdout_logger.error(message)
        if save_to_file:
            self.file_logger.error(message)

if __name__ == "__main__":
    log = Logger('test', std_out=True, save2file=True, file_name='../logs/test.log')
    # log = Logger('test', save2file=True)
    log.info('test info')
    log.info('test file log', save_to_file=True)