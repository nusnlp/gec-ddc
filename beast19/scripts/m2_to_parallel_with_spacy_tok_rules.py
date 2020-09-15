from collections import defaultdict
import re
import spacy
import en_core_web_sm
nlp=en_core_web_sm.load()
import argparse



def prepare_src_trg(m2file, outprefix, copy_source=False, cleanup=False):
    def spacytok1word(w):
        return ' '.join([tok.text for tok in list(nlp(w))])

    def do_cleanup(s):
        if cleanup == False:
            return s
        s = s.replace("`", "'").replace("''",'"')
        s = s.replace("' ve ","'ve ").replace("' t ", "'t ").replace("' t ","'t ").replace("' d ","'d ")
        s = s.replace("' m ","'m ").replace("' ll ","'ll ").replace("' re ","'re ")
        s = s.replace("'. . .","...")
        s = s.replace("n 't"," n't")
        s = re.sub(r'(\d+) %',r'\1%', s)
        s = ' '.join([spacytok1word(tok) if '-' in tok else tok for tok in s.split() ])
        return s


    words = []
    sid = eid = 0
    prev_sid = prev_eid = -1
    pos = 0
    output_src_path=outprefix+'.src'
    output_tgt_path=outprefix+'.trg'
    print("src: {} , trg: {}".format(output_src_path,output_tgt_path))
    skip_list = ["UNK","Um","noop"]
    with open(m2file) as fin, open(output_src_path, 'w') as output_src_file, open(output_tgt_path, 'w') as output_tgt_file:
        for line in fin:
            line = line.strip()
            if line.startswith('S'):
                line = line[2:]
                source = line
                annots_set = defaultdict(list)
                targets = dict()
            elif line.startswith('A'):
                line = line[2:]
                # add annotations to dictionary
                annot = line.split("|||")
                ann_id = annot[-1]
                annots_set[ann_id].append(annot)
                # initialize corrected sents with source sentence preceded by <S>
                swords = source.split()
                target = ['<S>'] + swords[:]
                targets[ann_id] = target
            else:
                if annots_set:
                    for ann_id, annots in annots_set.items():
                        for annot in annots:
                            sid, eid = annot[0].split()
                            sid = int(sid) + 1; eid = int(eid) + 1;
                            error_type = annot[1]
                            if error_type in skip_list:
                                continue
                            for idx in range(sid, eid):
                                targets[ann_id][idx] = ""
                            if sid == eid:
                                if sid == 0: continue   # Originally index was -1, indicating no op
                                if sid != prev_sid or eid != prev_eid:
                                    pos = len(targets[ann_id][sid-1].split())
                                    cur_words = targets[ann_id][sid-1].split()
                                    cur_words.insert(pos, annot[2])
                                    pos += len(annot[2].split())
                                    targets[ann_id][sid-1] = " ".join(cur_words)
                            else:
                                targets[ann_id][sid] = annot[2]
                            pos = 0
                            prev_sid = sid
                            prev_eid = eid
                        target_sentence = ' '.join([word for word in targets[ann_id] if word != ""])
                        assert target_sentence.startswith('<S>'), '(' + target_sentence + ')'
                        target_sentence = target_sentence[4:]
                        output_src_file.write(do_cleanup(source) + '\n')
                        output_tgt_file.write(do_cleanup(target_sentence) + '\n')
                        prev_sid = -1
                        prev_eid = -1
                        pos = 0
                else:
                    if copy_source:
                        output_src_file.write(do_cleanup(source) + '\n')
                        output_tgt_file.write(do_cleanup(source) + '\n')


parser = argparse.ArgumentParser()
parser.add_argument('-m2', required=True, help='path to input lang-8 m2 file')
parser.add_argument('-out', required=True, help='output prefix')
parser.add_argument('--cleanup', action='store_true')
parser.add_argument('--copy-unchanged-source', action='store_true')
args = parser.parse_args()

prepare_src_trg(args.m2, args.out, copy_source=args.copy_unchanged_source, cleanup=args.cleanup)