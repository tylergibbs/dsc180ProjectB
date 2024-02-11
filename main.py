
import logging

import numpy as np

def main():
    # Get arguments parsed
    #args = get_args()

    pass
    # Setup for logging
    #output_dir = 'output/{}'.format(get_datetime_str())
    #create_dir(output_dir)    # Create directory to save log files and outputs
    #LogHelper.setup(log_path='{}/training.log'.format(output_dir), level='INFO')
    #_logger = logging.getLogger(__name__)
    #_logger.info("Finished setting up the logger.")

    # Save configs
    #save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    #set_seed(args.seed)

    # Load dataset
    #dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree,
     #                          args.noise_type, args.B_scale, args.seed)
    #_logger.info("Finished loading the dataset.")

    # Load B_init for initialization
    #if args.init:
     #   if args.init_path is None:
      #      args.init_path = get_init_path('output/')
       # B_init = np.load('{}'.format(args.init_path))
        #_logger.info("Finished loading B_init from {}.".format(args.init_path))
    #else:
     #   B_init = None

    #Method
    #B_est=None
    #if args.method == 'GOLEM':
     #  print("GOLEMTS")
      # B_est = golem(dataset.X, args.lambda_1, args.lambda_2, args.equal_variances, args.num_iter,
       #           args.learning_rate, args.seed, args.checkpoint_iter, output_dir, B_init)
    #elif args.method == "DAGMA" and args.loss is not None:
     #  print("DYDAGMA")

      # B_est = dagma_linear(dataset.X, args.lambda_1, args.early_stop_delta, .1, args.num_iter, 
       #        args.checkpoint_iter, args.loss, False) 

    #elif args.method == "NOTEARS" and args.loss is not None:
     #  print("DYNOTEARS")
       #B_est = notears_linear(dataset.X, args.lambda_1, args.loss, .1, args.num_iter,
        #                     args.checkpoint_iter, args.early_stop_delta , False)

    #else:
     #  print(args.method + "requires --loss to be set")
       #_logger.info(args.method + "requires --loss to be set")
      # return 

    #_logger.info("Finished training the model.")

    # Post-process estimated solution
    #B_processed = postprocess(B_est, args.graph_thres)
    #_logger.info("Finished post-processing the estimated graph.")

    # Checkpoint
    #checkpoint_after_training(output_dir, dataset.X, dataset.B, B_init,
     #                         B_est, B_processed, _logger.info)


if __name__ == '__main__':
    main()

