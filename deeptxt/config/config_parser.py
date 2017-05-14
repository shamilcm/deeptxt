import configparser
import sys
from ast import literal_eval

class TrainingHyperparams(object):
    pass

class NetworkHyperparams(object):
    pass


class IniParser:
    ''' 
    Parse config from an ini file.
    
    # Arguments
        config_file: path to the configuration file.

    '''
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        if 'model' in config:
            self.model = config['model']['type']
        else:
            print >> sys.stderr, "Missing [model] section in config file!"
            sys.exit()

        if 'training' in config:
            self.train_params = TrainingHyperparams()
            for kk, vv in config['training'].iteritems():
                try:
                    vv = literal_eval(vv)
                except (SyntaxError,ValueError): # fails when having strings or other characters like '/'
                    pass
                setattr(self.train_params, kk, vv)
        else:
            print >> sys.stderr, "Missing [training] section in config file!"
            sys.exit()

        if 'network' in config:
            self.network_params = NetworkHyperparams()
            for kk, vv in config['network'].iteritems():
                try:
                    vv = literal_eval(vv)
                except (SyntaxError,ValueError):
                    pass
                setattr(self.network_params, kk, vv)
        else:
            print >> sys.stderr, 'Missing [network] section in config file!'
            sys.exit()


    # TODO: complete this in a better way
    # TODO: calling a parameter, and if it fails, throw an error
    