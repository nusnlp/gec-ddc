import argparse
# import random

# random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('-fo', '--fairseq-output', required=True, help='fairseq output path')
# parser.add_argument('--rand-from-n', type=int, default=1, help='choose random from top N (default N=1)')
args = parser.parse_args()

def print_sample(curidx, source, hyps, args):
	# chosen_hyp = random.sample(hyps[:args.rand_from_n],1)[0]
	print("{}\t{}\t{}".format(curidx,source, hyps[0]))
	#print(curidx, source, hyps)

with open(args.fairseq_output) as f:
	curidx=-1
	for line in f:
		line = line.strip()
		if not (line.startswith('S') or line.startswith('H')):
			continue
		
		pieces = line.split('\t')
		c, idx = pieces[0].split('-')
		if c == 'S':
			if curidx != -1:
				print_sample(curidx, source, hyps, args)
			curidx = int(idx)
			hyps = []
			source=pieces[1]
		elif c == 'H':
			hyps.append(pieces[-1])
	print_sample(curidx, source, hyps, args)

