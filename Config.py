import yaml


def getConfig(configFilePath, configName):
    configFileOpen = open(configFilePath, 'r', encoding='utf-8')
    configFile = configFileOpen.read()
    config = yaml.load(configFile, Loader=yaml.SafeLoader)
    return config[configName]


def getMainConfig(configName):
    return getConfig("config\\MainConfig.yml", configName)


def getDEConfig(configName):
    return getConfig("config\\DifferentialEvolution.yml", configName)


def getPretrainConfig(configName):
    return getConfig("config\\PretrainConfig.yml", configName)
