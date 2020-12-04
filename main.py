from model import train
from model import predict
from data import find_maxlen
import argparse


def main(args):
   if args.mode=='train':
      train(args.train_path, args.dev_path, args.aud_path, args.alphabet,
            args.model_path, args.num_epochs, args.batch_size)
   elif args.mode=='predict':
      predict(args.test_path, args.aud_path, args.alphabet, args.model_path)


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--train_path', type=str, help='Path to train csv', required=True)
   parser.add_argument('--dev_path', type=str, help="Path to dev csv", required=True)
   parser.add_argument('--test_path', type=str, help='Path to test csv', required=True)
   parser.add_argument('--model_path', type=str, help="Directory where model logs and checkpoints will be saved.", required=True)
   parser.add_argument('--aud_path', type=str, help='Path to audio files', required=True)
   parser.add_argument('--alphabet', type=str, help='Path to alphabet file in .txt format', required=True)
   parser.add_argument('--num_epochs', nargs='?', const=10, type=int, default=10, help="Number of epochs")
   parser.add_argument('--batch_size', nargs='?', const=32, type=int, default=32, help='Batch size')
   parser.add_argument('--mode', type=str, help="Select mode: train, predict", required=True)
   args = parser.parse_args()
   main(args)