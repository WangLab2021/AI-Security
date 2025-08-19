from pathlib import Path
import shutil
from loguru import logger
import sys

__all__ = ['setup_logger', 'move_files', 'copy_files', "logger"]

def setup_logger(log_file=None, level='DEBUG'):
    logger.remove()
    if log_file is not None:
        f = Path(log_file)
        f.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            f,
            rotation='1 day',
            retention='10 days',
            level=level,
            # format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}',
        )
    logger.add(sys.stdout, level=level)


def move_files(srcs, dsts):
    """
    Move files from srcs to dsts, creating directories as needed.
    """
    for src, dst in zip(srcs, dsts):
        src, dst = Path(src).expanduser().absolute(), Path(dst).expanduser().absolute()
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Moving {src} to {dst}")
            if not src.exists():
                logger.info(f"Source file {src} does not exist, skipping copy.")
                continue
            shutil.move(src, dst)
            
def copy_files(srcs, dsts):
    """
    Copy files from srcs to dsts, creating directories as needed.
    """
    for src, dst in zip(srcs, dsts):
        src, dst = Path(src).expanduser().absolute(), Path(dst).expanduser().absolute()
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying {src} to {dst}")
            if not src.exists():
                logger.info(f"Source file {src} does not exist, skipping copy.")
                continue
            shutil.copy(src, dst)