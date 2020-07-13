from fairseq import data, options
from load_weiqi_single_model import main
from load_weiqi_single_model import load_weiqi_single

import pickle
if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    df2 = open('weiqi_single_args_2.pt', 'wb')
    pickle.dump(args, df2)
    df2.close()
    main(args)
    # model1 = load_weiqi_single(args)
    # model1.forward()#
