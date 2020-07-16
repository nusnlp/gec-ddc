from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch, os, codecs, math


class departureLossRL_from_file(nn.Module):  # parsers
    def __init__(self, device, word_alphabet, vocab_size, args, src_dict):
        super(departureLossRL_from_file, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

        main_path = '/home/projects/11001764/wenjuan/gec_wj/bea19_outputs'
        departure_args_path = ['kakao', 'edin', 'toho']
        departure_args_path = [os.path.join(main_path, departure_args_path[i], 'dev.txt') for i in range(len(departure_args_path))]
        # self.model1 = load_weiqi_single(main_args=args, departure_args_path=departure_args_path)
        self.models = []
        self.src_dict = src_dict
        for i in range(len(departure_args_path)):
            with open(departure_args_path[i], encoding="utf8") as f:
                cands = [line.strip() for line in f]
                self.models.append(cands)

    def get_reward_departure(self, golden_out, departure_out_list, stc_length_out=None,
                             ignore_index=-100):  # have not ignore any tokens
        departure_out_list = departure_out_list.strip().split(' ')
        stc_length = min(len(golden_out), len(departure_out_list))
        stc_dda = sum([0 if golden_out[i] == departure_out_list[i] else 1 for i in range(0, stc_length)])

        reward = stc_dda

        return reward

    def sample_input_to_string(self, sample):
        # batch_size, lenghts_pad = sample['net_input']['src_tokens'].size()
        # lengths = sample['net_input']['src_lengths']
        tokens = [
            [
            self.src_dict.symbols[wd]
            for wd in stc
                if not (wd == self.src_dict.pad_index)
            ]
            for stc in sample
        ]
        return tokens #[' '.join(i) for i in tokens]

    def forward(self, sel, pb, out, mask_id, stc_length_out, sample_input, ignore_index=-100):
        model1_preds = []
        for i in range(len(self.models)):
            model1_preds.append(np.array(self.models[i])[sample_input['id'].cpu().tolist()])
        ####1####
        batch = sel.shape[0]
        out_str = self.sample_input_to_string(out)
        rewards_z1 = []
        for i in range(batch):  # batch
            reward = 0
            for j in range(len(model1_preds)):
                reward = reward + self.get_reward_departure(golden_out=out_str[i], departure_out_list=model1_preds[j][i],
                                                ignore_index=-100)  # we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        # -----------------------------------------------

        rewards = (rewards_z1) * 0.001  # TODO  ppl +
        # rewards = bleus_w * 10  # 8.26

        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        device = pb.device
        for j in range(stc_length_seq):
            # wgt3 = np.asarray([1 if j < min(stc_length_out[i]+1, stc_length_seq) else 0 for i in range(batch)])  # consider in STOP token
            wgt3 = np.asarray([1 if j < stc_length_seq else 0 for i in range(batch)])  # consider in STOP token
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards - self.bl).float().to(device) *  # rewards-self.bl
                    torch.from_numpy(wgt3.astype(float)).float().to(device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)
        self.bl = (self.bl * self.bn + rewards_ave3) / (self.bn + 1)
        self.bn += 1

        loss = ls3

        return loss, np.average(rewards_z1), None  # np.average(ppl) #loss, ls, ls1, bleu, bleu1