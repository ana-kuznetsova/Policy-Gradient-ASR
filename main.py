from model import train
from data import find_maxlen
import argparse


def main(args):
   train(args.csv_path, args.aud_path, args.alphabet)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--csv_path', type=str, help='Path to csv')
   parser.add_argument('--aud_path', type=str, help='Path to audio files')
   parser.add_argument('--alphabet', type=str, help='Path to alphabet file in .txt format')
   args = parser.parse_args()
   main(args)