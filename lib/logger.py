import argparse
import os
import logging
import sys
from datetime import datetime


def get_logger(args):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = '{}-{}-{}-{}.log'.format(args.exp_id,
                                            args.model,args.disc, time)
    logfilepath = os.path.join(args.log_dir, log_filename)

    logger = logging.getLogger(args.model)

    # critical > error > warning > info > debug > notset
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', args.log_dir)
    return logger


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--log_dir', default='E:\Python\GNN\Traffic Codes\ModelFlow\experiments\PEMS', type=str)
    args.add_argument('--debug', default=False, type=eval)
    args.add_argument('--model', default='AGCRN', type=str)
    args.add_argument('--exp_id', default=1, type=int)
    args.add_argument('--dataset', default='PEMS04', type=str)
    args = args.parse_args()
    logger = get_logger(args)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.warning('Gradient explosion detected. Ending...')
    print(logger)
