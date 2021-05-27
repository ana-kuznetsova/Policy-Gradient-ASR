from model import train
from model import predict
from data import preproc
import argparse


def main(args):
   if args.mode=='train':
      train(args.corpus_path, args.model_path, args.num_epochs, args.batch_size, args.device)
   elif args.mode=='predict':
      predict(args.test_path, args.aud_path, args.alphabet, args.model_path, 
              args.batch_size, args.maxlen, args.maxlent)
   elif args.mode=='preproc':
      preproc(args.corpus_path)



if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--corpus_path', type=str, help='Directory where the corpus is stored')
   parser.add_argument('--model_path', type=str, help="Directory where model logs and checkpoints will be saved.")
   parser.add_argument('--num_epochs', nargs='?', type=int, default=10, help="Number of epochs")
   parser.add_argument('--batch_size', nargs='?', type=int, default=32, help='Batch size')
   parser.add_argument('--mode', type=str, help="Select mode: train, predict", required=True)
   parser.add_argument('--device', type=int, help="GPU id", required=True)
   args = parser.parse_args()
   main(args)