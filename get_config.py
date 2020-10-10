import configparser


def get_config(cfg):
    data_path = cfg.get('PATH', 'data_path')



cfg = configparser.ConfigParser()
cfg.read("./config.ini",encoding='utf-8')
get_config(cfg)

