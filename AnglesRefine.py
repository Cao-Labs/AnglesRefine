import argparse
import math
import shutil
import numpy as np
import numpy
import linecache
import json
import os
import sympy
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from Model import Transformer
from PDB2Angles import filterPDB, extract_backbone_model
from Angles2PDB import build_PDB_model
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


# stating model ——> structure_a,pre_refined_local_structure,structure_b
def separate(pdbfile, l, seqnum):
    print('\nseparate``````````````````````', file=log)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fixedcoord1 = '%s/%s_fixedcoord1.pdb' % (model_dir, seqnum)
    modifiedcoord = '%s/%s_modifiedcoord.pdb' % (model_dir, seqnum)
    fixedcoord2 = '%s/%s_fixedcoord2.pdb' % (model_dir, seqnum)

    if type(pdbfile) == str:
        f = open(pdbfile, "r")
    else:
        f = open(pdbfile.name, "r")
    lines = f.readlines()

    if type(fixedcoord1) == str:
        fix1 = open(fixedcoord1, "w")
    else:
        fix1 = open(fixedcoord1.name, "w")

    if type(modifiedcoord) == str:
        mod = open(modifiedcoord, "w")
    else:
        mod = open(modifiedcoord.name, "w")

    if type(fixedcoord2) == str:
        fix2 = open(fixedcoord2, "w")
    else:
        fix2 = open(fixedcoord2.name, "w")
    for i in range(0, len(lines)):
        if i >= 0 and i <= int(l[0]) - 2:
            fix1.write(str(lines[i]))
        if i >= int(l[0]) - 1 and i <= int(l[1]) - 1:
            mod.write(str(lines[i]))
        if i >= int(l[1]):
            fix2.write(str(lines[i]))
    fix1.close()
    mod.close()
    fix2.close()
    print('structure_a:%s\npre_refine_local_structure:%s\nstructure_b:%s    SAVED!'%(fixedcoord1, modifiedcoord, fixedcoord2), file=log)
    return fixedcoord1, modifiedcoord, fixedcoord2


def data_convertion(data, maxAngle, minAngle):
    normlist = []
    for val in data:
        val = float(val)
        normVal = (val - minAngle) / (maxAngle - minAngle)
        normlist.append(round(normVal, 3))

    normlist = [i * 1000 for i in normlist]
    return normlist


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    dec_input = dec_input.to(device)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat(
            [dec_input.detach().to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if (next_symbol == 1002):
            terminal = True
    return dec_input


def load_test_source(angle_name, seqnum):
    source_file = '%s/angles_out/%s_mod-%s.json' % (model_dir, seqnum, angle_name)
    with open(source_file, 'r') as sf:
        source_list = json.load(sf)
    sf.close()
    return source_list


def convert_to_Longtensor(data):
    data = torch.LongTensor(data)
    return data


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx]


def pdb2angles(modifiedcoord, angles_out, seqnum):
    print('\npdb2angles`````````````````', file=log)
    pdb_input_path = modifiedcoord
    print('pdb：', pdb_input_path, file=log)
    pdb_input_path_new = modifiedcoord + ".tmp"
    filterPDB(pdb_input_path, pdb_input_path_new)
    # extract all angles and bond lengths
    structure_backbone = extract_backbone_model(pdb_input_path_new, angles_out)
    os.remove(pdb_input_path_new)

    CA_C_N_angle = []
    C_N_CA_angle = []
    peptide_bond = []
    psi_im1 = []
    omega = []
    phi = []
    CA_N_length = []
    CA_C_length = []
    N_CA_C_angle = []
    for line in linecache.getlines(angles_out)[1:]:
        CA_C_N_angle.append(line.split()[1])
        C_N_CA_angle.append(line.split()[2])
        CA_N_length.append(line.split()[3])
        CA_C_length.append(line.split()[4])
        peptide_bond.append(line.split()[5])
        psi_im1.append(line.split()[6])
        omega.append(line.split()[7])
        phi.append(line.split()[8])
        N_CA_C_angle.append(line.split()[11])

    psi_im1 = data_convertion(np.asarray(psi_im1).astype(float), 180.0, -180.0)
    phi = data_convertion(np.asarray(phi).astype(float), 180.0, -180.0)
    omega = data_convertion(np.asarray(omega).astype(float), 180.0, -180.0)
    CA_C_N_angle = data_convertion(np.asarray(CA_C_N_angle).astype(float), 180.0, 0.0)
    C_N_CA_angle = data_convertion(np.asarray(C_N_CA_angle).astype(float), 180.0, 0.0)
    N_CA_C_angle = data_convertion(np.asarray(N_CA_C_angle).astype(float), 180.0, 0.0)

    json_str = json.dumps(CA_C_N_angle)
    with open(
            os.path.join("%s/angles_out/%s_mod-CA_C_N_angle.json" % ( model_dir, seqnum)),
            "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(C_N_CA_angle)
    with open(
            os.path.join("%s/angles_out/%s_mod-C_N_CA_angle.json" % (model_dir, seqnum)),
            "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(N_CA_C_angle)
    with open(
            os.path.join("%s/angles_out/%s_mod-N_CA_C_angle.json" % (model_dir, seqnum)),
            "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(psi_im1)
    with open(os.path.join("%s/angles_out/%s_mod-psi.json" % (model_dir, seqnum)),
              "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(omega)
    with open(os.path.join("%s/angles_out/%s_mod-omega.json" % (model_dir, seqnum)),
              "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(phi)
    with open(os.path.join("%s/angles_out/%s_mod-phi.json" % (model_dir, seqnum)),
              "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(CA_N_length)
    with open(os.path.join("%s/angles_out/%s_mod-CA_N_length.json" % (model_dir, seqnum)),
              "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(CA_C_length)
    with open(os.path.join("%s/angles_out/%s_mod-CA_C_length.json" % (model_dir, seqnum)),
              "w") as fw:
        fw.write(json_str)
        fw.close()
    json_str = json.dumps(peptide_bond)
    with open(os.path.join("%s/angles_out/%s_mod-peptide_bond.json" % (model_dir, seqnum)),"w") as fw:
        fw.write(json_str)
        fw.close()
    print('angles: %s/angles_out/  SAVED!' % model_dir, file=log)


def predict(angle_name, angle_n, seqlen, seqnum):
    # print('predict %s -%s``````````````````````' % (model_name, angle_name), file=log)
    print('predict angles (%s-%s)``````````````````````' % (angle_name, seqlen), file=log)
    ss = 'H'
    # Helix_angle_Transformer_modeldir = "%s/%s_model/%s_%s" % (Helix_angle_Transformer_dir, angle_n, ss, seqlen)
    # path_checkpoint = '%s/ckpt_best.pth' % Helix_angle_Transformer_modeldir

    path_checkpoint = '%s/%s_model/%s_%s.pth' % (Helix_angle_Transformer_dir, angle_n,angle_n,seqlen)

    output_dir = '%s/unmatch_pred/%s/' % (model_dir, angle_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = Transformer().to(device)
    test_source_list = load_test_source(angle_name,seqnum)
    test_enc_inputs = convert_to_Longtensor(test_source_list)
    # print(test_enc_inputs, file=log)
    test_dataset = MyDataSet(test_enc_inputs)
    # test_loader = DataLoader(test_dataset, 1, False)
    test_loader = DataLoader(test_dataset, len(test_enc_inputs), False)
    for step, test_enc_inputs in enumerate(test_loader):
        pass
    # print(' batches: {:5d}  | enci: {} '.format(step, test_enc_inputs), file=log)
    # print('\nBEST EPOCH:    LOSS    ', checkpoint['epoch'], ':    ', checkpoint['loss'], file=log)
    # print('----------PREDICTION-----------', file=log)
    # print('\n-----angle(%s)_ss(%s)_seqlen(%s)-----' % (angle_name, ss, seqlen), file=log)
    model.eval()
    total_acc = 0.
    t_acc = 0.
    # checkpoint = torch.load(path_checkpoint)
    if torch.cuda.is_available() == False:
        checkpoint = torch.load(path_checkpoint,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model'])
    d = {}
    aa = []
    for i, data in enumerate(test_loader):
        a = []
        # print('i:', i, file=log)
        enc_inputs = data
        enc_inputs = enc_inputs.to(device)
        greedy_dec_input = greedy_decoder(model, enc_inputs.view(1, -1), start_symbol=1001)
        predict, _, _, _ = model(enc_inputs.view(1, -1), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        predict = predict[0:len(predict) - 1]
        for j in range(0, len(predict.squeeze().cpu().numpy().tolist())):
            a.append(predict.squeeze().cpu().numpy().tolist()[j])
        print('source:', enc_inputs.squeeze(), '->', 'prediction:', predict.squeeze(), file=log)
        if len(predict.squeeze().cpu().numpy().tolist()) != int(seqlen):
            print(len(predict.squeeze().cpu().numpy().tolist()), file=log)
            print(seqlen, file=log)
            print('WRONG', file=log)
            break
        aa.append(a)
        i += 1
    d[seqlen] = aa

    listdir = os.listdir(output_dir)
    jsonfile = '%s_pred.json' % ss
    if jsonfile in listdir:
        fr = open(os.path.join(output_dir, jsonfile))
        dict_data = json.load(fr)
        fr.close()
        dict_data[seqlen] = aa
        json_str = json.dumps(dict_data)
        with open(os.path.join(output_dir, jsonfile), "w") as fw:
            fw.write(json_str)
            fw.close()
    else:
        json_str = json.dumps(d)
        with open(os.path.join(output_dir, jsonfile), "w") as fw:
            fw.write(json_str)
            fw.close()


def load_residue_sequence(angles_out):
    rseq = []
    for line in linecache.getlines(angles_out):
        rseq.append(line.split()[0])
    return rseq


def load_length_sequence(seqnum):
    with open('%s/angles_out/%s_mod-peptide_bond.json' % (model_dir, seqnum), 'r') as f:
        data = json.load(f)
        peptide_bond_seq = data
    f.close()
    with open('%s/angles_out/%s_mod-CA_N_length.json' % (model_dir, seqnum), 'r') as f:
        data = json.load(f)
        CA_N_length_seq = data
    f.close()
    with open('%s/angles_out/%s_mod-CA_C_length.json' % (model_dir, seqnum), 'r') as f:
        data = json.load(f)
        CA_C_length_seq = data
    f.close()
    return CA_N_length_seq, CA_C_length_seq, peptide_bond_seq


def load_pred(angle_name, seqlen):
    ss = 'H'
    with open('%s/unmatch_pred/%s/%s_pred.json' % (model_dir, angle_name, ss),
              'r') as f:
        data = json.load(f)
        data = data[seqlen][0]
    f.close()
    return data


def data_recovery(data, max, min):
    max = float(max)
    min = float(min)
    b = []
    for i in range(0, len(data)):
        data[i] = (data[i] / 1000) * (max - min) + min
        b.append(data[i])
    return b


def format_json(angles_out, seqlen, seqnum):
    print('\nformat``````````````````````', file=log)
    max1, min1 = 180.0, -180.0
    max3, min3 = 180.0, -180.0
    max2, min2 = 180.0, -180.0
    max4, min4 = 180.0, 0.0
    max5, min5 = 180.0, 0.0
    max6, min6 = 180.0, 0.0

    residue_sequence = load_residue_sequence(angles_out)
    CA_N_length_seq, CA_C_length_seq, peptide_bond_seq = load_length_sequence( seqnum)
    # print(CA_N_length_seq, '\n', CA_C_length_seq, '\n', peptide_bond_seq)
    CA_C_N_angle_pred = load_pred('CA_C_N_angle', str(seqlen))
    C_N_CA_angle_pred = load_pred('C_N_CA_angle', str(seqlen))
    psi_pred = load_pred('psi',str(seqlen))
    omega_pred = load_pred('omega',  str(seqlen))
    phi_pred = load_pred('phi', str(seqlen))
    N_CA_C_angle_pred = load_pred('N_CA_C_angle',  str(seqlen))
    CA_C_N_angle_pred = data_recovery(CA_C_N_angle_pred, max4, min4)
    C_N_CA_angle_pred = data_recovery(C_N_CA_angle_pred, max5, min5)
    psi_pred = data_recovery(psi_pred, max1, min1)
    omega_pred = data_recovery(omega_pred, max3, min3)
    phi_pred = data_recovery(phi_pred, max2, min2)
    N_CA_C_angle_pred = data_recovery(N_CA_C_angle_pred, max6, min6)

    pred_angles_list = []
    output_dir = '%s/final_format_angles/' % model_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+"%s_mod_pred_angles.json" % seqnum,"w") as json_output:
        json_output.write(
            '# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle')
        json_output.write('\n')
        for i in range(0, len(psi_pred)):
            pred_angles_list.append(str(residue_sequence[i + 1]) + ' ')
            pred_angles_list.append(str(CA_C_N_angle_pred[i]) + ' ')
            pred_angles_list.append(str(C_N_CA_angle_pred[i]) + ' ')
            pred_angles_list.append(str(CA_N_length_seq[i]) + ' ')
            pred_angles_list.append(str(CA_C_length_seq[i]) + ' ')
            pred_angles_list.append(str(peptide_bond_seq[i]) + ' ')
            pred_angles_list.append(str(psi_pred[i]) + ' ')
            pred_angles_list.append(str(omega_pred[i]) + ' ')
            pred_angles_list.append(str(phi_pred[i]) + ' ')
            pred_angles_list.append(str(CA_N_length_seq[i]) + ' ')
            pred_angles_list.append(str(CA_C_length_seq[i]) + ' ')
            pred_angles_list.append(str(N_CA_C_angle_pred[i]) + ' ')
            pred_angles_list.append('\n')
        json_output.writelines(pred_angles_list)
    json_output.close()
    print('formatAngles:%s  SAVED!'%(output_dir+'%s_mod_pred_angles.json' % seqnum), file=log)

'''
db(pre_refined_local_structure) ——>
angles ——> source ——> prediction ——> format angles ——>
refined pdbre(fined_local_structure)
'''
def refine(modifiedcoord,ResidueIndex, seqnum):
    # print('refine``````````````````````', file=log)
    output_dir1 = '%s/angles_out/' % model_dir
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    output_dir2 = '%s/unmatch_pred/' % model_dir
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    output_dir3 = '%s/final_format_angles/' % model_dir
    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3)
    output_dir4 = '%s/final_refined_pdb/' % model_dir
    if not os.path.exists(output_dir4):
        os.makedirs(output_dir4)

    angles_out = output_dir1+'%s_mod' %  seqnum

    # pdb2angles
    pdb2angles(modifiedcoord, angles_out, seqnum)

    # predict  :source——>pred.json
    print('\npredicting Helix angles``````````````````````', file=log)
    predict('psi', 'psi', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    predict('phi', 'phi', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    predict('omega', 'omega', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    predict('CA_C_N_angle', 'CCN', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    predict('C_N_CA_angle', 'CNC', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    predict('N_CA_C_angle', 'NCC', ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)
    print('\npredicted Helix Angles: %s/unmatch_pred/ SAVED!' % model_dir, file=log)

    # format json
    format_json(angles_out, ResidueIndex[1] - ResidueIndex[0] + 1, seqnum)

    # angles2PDB
    print('\nangles2PDB``````````````````````', file=log)
    angles_input_path = "%s/final_format_angles/%s_mod_pred_angles.json" % (model_dir, seqnum)
    PDB_out = "%s/%s_mod_pred.pdb" % (model_dir, seqnum)
    print('angles：', angles_input_path, file=log)
    with suppress_stdout_stderr():
        build_PDB_model(angles_input_path, PDB_out)
    print('pdb: %s SAVED!' %PDB_out, file=log)
    refinedcoord = PDB_out

    return refinedcoord


def get_f1_x1(fixedcoord1):
    if type(fixedcoord1) == str:
        fix1 = open(fixedcoord1, "r")
    else:
        fix1 = open(fixedcoord1.name, "r")
    f1_x1 = []
    f1_lines = fix1.readlines()
    if len(f1_lines) != 0:
        indx = len(f1_lines) - 1
        last_res = int(f1_lines[indx].split()[5])
    for i in range(len(f1_lines)):
        if len(f1_lines[i].split()) > 9:
            if int(f1_lines[i].split()[5]) == last_res and f1_lines[i].split()[2] == 'CA':
                last_CA = i
                # print(last_CA)
                f1_1 = float(f1_lines[i][30:38])
                f1_2 = float(f1_lines[i][38:46])
                f1_3 = float(f1_lines[i][46:54])
                f1_x1.append(f1_1)
                f1_x1.append(f1_2)
                f1_x1.append(f1_3)
                break
    return f1_x1


def get_f2_y1(fixedcoord2):
    if type(fixedcoord2) == str:
        fix2 = open(fixedcoord2, "r")
    else:
        fix2 = open(fixedcoord2.name, "r")
    f2_y1 = []
    f2_lines = fix2.readlines()
    # print(fix2)
    # print(f2_lines[0])
    if len(f2_lines) != 0 and len(f2_lines[0].split()) >5:
        first_res = int(f2_lines[0].split()[5])
    for i in range(len(f2_lines)):
        if len(f2_lines[i].split()) > 9:
            if int(f2_lines[i].split()[5]) == first_res and f2_lines[i].split()[2] == 'CA':
                first_CA = i
                f2_1 = float(f2_lines[i][30:38])
                f2_2 = float(f2_lines[i][38:46])
                f2_3 = float(f2_lines[i][46:54])
                f2_y1.append(f2_1)
                f2_y1.append(f2_2)
                f2_y1.append(f2_3)
                break
    return f2_y1


def getkeycoord(orig_modcoord, refinedcoord):
    if type(orig_modcoord) == str:
        orig_mod = open(orig_modcoord, "r")
    else:
        orig_mod = open(orig_modcoord.name, "r")

    mod_lines = orig_mod.readlines()
    if len(mod_lines[len(mod_lines) - 1].split()) != len(mod_lines[0].split()):
        indx = len(mod_lines) - 3
    else:
        indx = len(mod_lines) - 1
    first_res = int(mod_lines[0].split()[5])
    last_res = int(mod_lines[indx].split()[5])
    x1, y1 = [], []
    for i in range(len(mod_lines)):
        if len(mod_lines[i].split()) > 9:
            if int(mod_lines[i].split()[5]) == first_res and mod_lines[i].split()[2] == 'CA':
                first_CA = i
                m1_1 = float(mod_lines[i][30:38])
                m1_2 = float(mod_lines[i][38:46])
                m1_3 = float(mod_lines[i][46:54])
                x1.append(m1_1)
                x1.append(m1_2)
                x1.append(m1_3)

            if int(mod_lines[i].split()[5]) == last_res and mod_lines[i].split()[2] == 'CA':
                last_CA = i
                m2_1 = float(mod_lines[i][30:38])
                m2_2 = float(mod_lines[i][38:46])
                m2_3 = float(mod_lines[i][46:54])
                y1.append(m2_1)
                y1.append(m2_2)
                y1.append(m2_3)

    orig_mod.close()

    if type(refinedcoord) == str:
        mod = open(refinedcoord, "r")
    else:
        mod = open(refinedcoord.name, "r")
    mod_lines = mod.readlines()
    if len(mod_lines[len(mod_lines) - 1].split()) != len(mod_lines[0].split()):
        indx = len(mod_lines) - 3
    else:
        indx = len(mod_lines) - 1
    first_res = int(mod_lines[0].split()[5])
    last_res = int(mod_lines[indx].split()[5])
    medium_res = int((first_res + last_res) / 2)
    x0, y0, m0 = [], [], []
    for i in range(len(mod_lines)):
        if len(mod_lines[i].split()) > 9:
            if int(mod_lines[i].split()[5]) == first_res and mod_lines[i].split()[2] == 'CA':
                first_CA = i
                m1_1 = float(mod_lines[i][30:38])
                m1_2 = float(mod_lines[i][38:46])
                m1_3 = float(mod_lines[i][46:54])
                x0.append(m1_1)
                x0.append(m1_2)
                x0.append(m1_3)
            if int(mod_lines[i].split()[5]) == medium_res and mod_lines[i].split()[2] == 'CA':
                medium_CA = i
                m3_1 = float(mod_lines[i][30:38])
                m3_2 = float(mod_lines[i][38:46])
                m3_3 = float(mod_lines[i][46:54])
                m0.append(m3_1)
                m0.append(m3_2)
                m0.append(m3_3)
            if int(mod_lines[i].split()[5]) == last_res and mod_lines[i].split()[2] == 'CA':
                last_CA = i
                m2_1 = float(mod_lines[i][30:38])
                m2_2 = float(mod_lines[i][38:46])
                m2_3 = float(mod_lines[i][46:54])
                y0.append(m2_1)
                y0.append(m2_2)
                y0.append(m2_3)
    return x0, m0, y0, x1, y1


# translate
def translate(coord1, coord2=None, x0=None, x1=None, trans=None):
    if x0 == None:
        translation = [float(trans[0]), float(trans[1]), float(trans[2])]  # translation=[1,-2,0]
    else:
        translation = [float(x1[0]) - float(x0[0]), float(x1[1]) - float(x0[1]),
                       float(x1[2]) - float(x0[2])]  # translation=[1,-2,0]
    if type(coord1) == str:
        f = open(coord1, "r")
    else:
        f = open(coord1.name, "r")
    lines = f.readlines()
    if coord2 != None:
        if type(coord2) == str:
            nf = open(coord2, "r")
        else:
            nf = open(coord2.name, "r")
        new_lines = nf.readlines()
    if type(coord1) == str:
        fpath, fname = os.path.split(coord1)
    else:
        fpath, fname = os.path.split(coord1.name)

    new_coord = '%s/' % model_dir + fname[:-4] + '_translate.pdb'

    if type(new_coord) == str:
        new_f = open(new_coord, "w")
    else:
        new_f = open(new_coord.name, "w")
    l = []
    count = 0
    for i in range(0, len(lines)):
        if len(lines[i].split()) > 9:
            count += 1
            l.append(str(lines[i].split()[0]))
            l.append(str(lines[i].split()[1]).rjust(7, ' ') + '  ')
            l.append(str(lines[i].split()[2]).ljust(4, ' '))

            l.append(str(lines[i].split()[3]).ljust(3, ' '))
            l.append(str(lines[i].split()[4]).rjust(2, ' '))  # ‘A'
            l.append(str(lines[i].split()[5]).rjust(4, ' '))

            if coord2 != None:

                l.append(str('%.3f' % (float(new_lines[i][30:38]) + translation[0])).rjust(12, ' '))
                l.append(str('%.3f' % (float(new_lines[i][38:46]) + translation[1])).rjust(8, ' '))
                l.append(str('%.3f' % (float(new_lines[i][46:54]) + translation[2])).rjust(8, ' '))
            else:

                l.append(str('%.3f' % (float(lines[i][30:38]) + translation[0])).rjust(12, ' '))
                l.append(str('%.3f' % (float(lines[i][38:46]) + translation[1])).rjust(8, ' '))
                l.append(str('%.3f' % (float(lines[i][46:54]) + translation[2])).rjust(8, ' '))

            l.append(str(lines[i][54:60].strip()).rjust(6, ' '))
            l.append(str(lines[i][60:66].strip()).rjust(6, ' '))
            l.append(str(lines[i][76:78].strip()).rjust(12, ' '))
            l.append('\n')
    new_f.writelines(l)
    while count < len(lines):
        new_f.write(str(lines[count]))
        count = count + 1
    new_f.close()
    coord = new_f
    return coord


def is_parallel(vec1, vec2):

    assert isinstance(vec1, numpy.ndarray), r'input vec1 must be ndarray'
    assert isinstance(vec2, numpy.ndarray), r'input vec2 must be ndarray '
    assert vec1.shape == vec2.shape, r'input shapes must be consistent'

    vec1_normalized = vec1 / numpy.linalg.norm(vec1)
    vec2_normalized = vec2 / numpy.linalg.norm(vec2)

    if 1.0 - abs(numpy.dot(vec1_normalized, vec2_normalized)) < 1e-6:
        return True
    else:
        return False


# rotate
def rotate(x0, y0, y1, coord):
    if type(coord) == str:
        f = open(coord, "r")
    else:
        f = open(coord.name, "r")
    lines = f.readlines()

    if type(coord) == str:
        fpath, fname = os.path.split(coord)
    else:
        fpath, fname = os.path.split(coord.name)

    new_coord = '%s/' % model_dir + fname[:-4] + '_rotate.pdb'

    if type(new_coord) == str:
        new_f = open(new_coord, "w")
    else:
        new_f = open(new_coord.name, "w")
    l = []
    x0 = [float(x0[0]), float(x0[1]), float(x0[2])]
    y0 = [float(y0[0]), float(y0[1]), float(y0[2])]
    y1 = [float(y1[0]), float(y1[1]), float(y1[2])]

    x0 = [round(float(x0[0]), 3), round(float(x0[1]), 3), round(float(x0[2]), 3)]
    y0 = [round(float(y0[0]), 3), round(float(y0[1]), 3), round(float(y0[2]), 3)]
    y1 = [round(float(y1[0]), 3), round(float(y1[1]), 3), round(float(y1[2]), 3)]

    x0y0 = numpy.asarray(y0) - numpy.asarray(x0)
    x0y1 = numpy.asarray(y1) - numpy.asarray(x0)
    x = np.array(x0y0)
    y = np.array(x0y1)

    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)

    angle = np.arccos(cos_angle)
    angle2 = angle * 180 / np.pi

    x, y, z = sympy.symbols("x y z")
    r = math.dist(x0, y0)

    c = (x - x0[0]) * (x - x0[0]) + (y - x0[1]) * (y - x0[1]) + (z - x0[2]) * (z - x0[2]) - r * r

    s = [y1[0] - x0[0], y1[1] - x0[1], y1[2] - x0[2]]

    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
    # print('x0y1:', file=log)
    # print(l1, file=log)
    # print(l2, file=log)
    # print(l3, file=log)

    a = sympy.solve([c, l1, l2, l3], [x, y, z])
    # print(a, file=log)
    if len(a) == 0:

        s = [round(s[0], 3), round(s[1], 3), round(s[2], 3)]
        l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
        l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
        l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

        a = sympy.solve([c, l1, l2, l3], [x, y, z])

        if len(a) == 0:

            s = [round(s[0], 2), round(s[1], 2), round(s[2], 2)]
            l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
            l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
            l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

            a = sympy.solve([c, l1, l2, l3], [x, y, z])

            if len(a) == 0:

                s = [round(s[0], 1), round(s[1], 1), round(s[2], 1)]
                l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

                a = sympy.solve([c, l1, l2, l3], [x, y, z])

                if len(a) == 0:

                    s = [int(s[0]), int(s[1]), int(s[2])]
                    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

                    a = sympy.solve([c, l1, l2, l3], [x, y, z])

                    if len(a) == 0:

                        coord = f
                        return False
    if math.dist(a[0], y1) < math.dist(a[1], y1):
        w = a[0]
    else:
        w = a[1]
    new_y0 = list(w)

    ws = [w[0] - x0[0], w[1] - x0[1], w[2] - x0[2]]
    ws = numpy.array(ws, dtype=numpy.float64)
    s = numpy.array(s, dtype=numpy.float64)
    m0 = [(y0[0] + w[0]) / 2, (y0[1] + w[1]) / 2, (y0[2] + w[2]) / 2]
    n0 = numpy.asarray([float(x0[0] - m0[0]), float(x0[1] - m0[1]), float(x0[2] - m0[2])])

    for line in lines:

        yc = [line[30:38], line[38:46], line[46:54]]
        yc = [float(yc[0]), float(yc[1]), float(yc[2])]

        if yc == x0:
            new_yc = yc
        else:
            if Lx == 0 or Ly == 0:
                new_yc = yc
            else:
                if angle2 == 0:
                    new_yc = yc
                else:
                    if angle2 == 180:
                        mc = m0
                        nc = n0
                        p1 = numpy.asarray([float(yc[0] - mc[0]), float(yc[1] - mc[1]), float(yc[2] - mc[2])])
                    else:
                        cro = numpy.cross(
                            numpy.asarray([float(x0[0] - yc[0]), float(x0[1] - yc[1]), float(x0[2] - yc[2])]), n0)
                        c = np.sqrt(cro.dot(cro))
                        h = c / np.sqrt(n0.dot(n0))
                        l2 = math.dist(x0, yc) * math.dist(x0, yc) - h * h
                        bili = math.sqrt(l2) / math.dist(x0, m0)
                        mc = [bili * (m0[0] - x0[0]) + x0[0], bili * (m0[1] - x0[1]) + x0[1],
                              bili * (m0[2] - x0[2]) + x0[2]]
                        nc = numpy.asarray(
                            [float(x0[0] - mc[0]), float(x0[1] - mc[1]), float(x0[2] - mc[2])])  # 绕nOX0# 旋转轴
                        p1 = numpy.asarray([float(yc[0] - mc[0]), float(yc[1] - mc[1]), float(yc[2] - mc[2])])
                        nc = numpy.array(nc, dtype=numpy.float64)
                        n0 = numpy.array(n0, dtype=numpy.float64)

                    rotationAngle = 180
                    cos = round(math.cos(rotationAngle * math.pi / 180), 3)
                    sin = round(math.sin(rotationAngle * math.pi / 180), 3)
                    rotatedVector = -p1
                    new_yc = rotatedVector + mc
                    x0y0 = numpy.asarray(y0) - numpy.asarray(x0)
                    x0yc = numpy.asarray(new_yc) - numpy.asarray(x0)
                    x = np.array(x0y0, dtype=numpy.float64)
                    y = np.array(x0yc, dtype=numpy.float64)
                    Lx = np.sqrt(x.dot(x))
                    Ly = np.sqrt(y.dot(y))
                    cos_angle = x.dot(y) / (Lx * Ly)
                    angle = np.arccos(cos_angle)
                    angle2 = angle * 180 / np.pi
                    new_yc = numpy.asarray([round(new_yc[0], 3), round(new_yc[1], 3), round(new_yc[2], 3)])
                    w = numpy.asarray([round(w[0], 3), round(w[1], 3), round(w[2], 3)])
                    rotatedVector = numpy.asarray([round(rotatedVector[0], 3), round(rotatedVector[1], 3), round(rotatedVector[2], 3)])
                    if yc == y0:
                        now_y0 = yc
                        now_new_y0 = list(new_yc)
        l.append(str(line.split()[0]))
        l.append(str(line.split()[1]).rjust(7, ' ') + '  ')
        l.append(str(line.split()[2]).ljust(4, ' '))

        l.append(str(line.split()[3]).ljust(3, ' '))
        l.append(str(line.split()[4]).rjust(2, ' '))  # ‘A'
        l.append(str(line.split()[5]).rjust(4, ' '))

        l.append(str('%.3f' % new_yc[0]).rjust(12, ' '))
        l.append(str('%.3f' % new_yc[1]).rjust(8, ' '))
        l.append(str('%.3f' % new_yc[2]).rjust(8, ' '))

        l.append(str(line[54:60].strip()).rjust(6, ' '))
        l.append(str(line[60:66].strip()).rjust(6, ' '))
        l.append(str(line[76:78].strip()).rjust(12, ' '))
        l.append('\n')

    new_f.writelines(l)
    new_f.close()
    coord = new_f
    return coord


def long_rotate(x0, y0, y1, coord):

    if type(coord) == str:
        f = open(coord, "r")
    else:
        f = open(coord.name, "r")
    lines = f.readlines()
    if type(coord) == str:
        fpath, fname = os.path.split(coord)
    else:
        fpath, fname = os.path.split(coord.name)
    new_coord = '%s/' % model_dir + fname[:-4] + '_rotate.pdb'
    if type(new_coord) == str:
        new_f = open(new_coord, "w")
    else:
        new_f = open(new_coord.name, "w")
    l = []
    x0 = [float(x0[0]), float(x0[1]), float(x0[2])]
    y0 = [float(y0[0]), float(y0[1]), float(y0[2])]
    y1 = [float(y1[0]), float(y1[1]), float(y1[2])]
    x0 = [round(float(x0[0]), 3), round(float(x0[1]), 3), round(float(x0[2]), 3)]
    y0 = [round(float(y0[0]), 3), round(float(y0[1]), 3), round(float(y0[2]), 3)]
    y1 = [round(float(y1[0]), 3), round(float(y1[1]), 3), round(float(y1[2]), 3)]

    x0y0 = numpy.asarray(y0) - numpy.asarray(x0)
    x0y1 = numpy.asarray(y1) - numpy.asarray(x0)
    x = np.array(x0y0)
    y = np.array(x0y1)
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 180 / np.pi
    new_y0 = y1
    m0 = [(y0[0] + y1[0]) / 2, (y0[1] + y1[1]) / 2, (y0[2] + y1[2]) / 2]
    n0 = numpy.asarray([float(x0[0] - m0[0]), float(x0[1] - m0[1]), float(x0[2] - m0[2])])
    for line in lines:
        yc = [line[30:38], line[38:46], line[46:54]]
        yc = [float(yc[0]), float(yc[1]), float(yc[2])]
        if yc == x0:
            new_yc = yc
        else:
            if Lx == 0 or Ly == 0:
                new_yc = yc
            else:
                if angle2 == 0:
                    new_yc = yc
                else:
                    if angle2 == 180:
                        mc = m0
                        nc = n0
                        p1 = numpy.asarray([float(yc[0] - mc[0]), float(yc[1] - mc[1]), float(yc[2] - mc[2])])
                    else:
                        cro = numpy.cross(
                            numpy.asarray([float(x0[0] - yc[0]), float(x0[1] - yc[1]), float(x0[2] - yc[2])]), n0)
                        c = np.sqrt(cro.dot(cro))
                        h = c / np.sqrt(n0.dot(n0))
                        l2 = math.dist(x0, yc) * math.dist(x0, yc) - h * h
                        bili = math.sqrt(l2) / math.dist(x0, m0)
                        mc = [bili * (m0[0] - x0[0]) + x0[0], bili * (m0[1] - x0[1]) + x0[1],
                              bili * (m0[2] - x0[2]) + x0[2]]
                        nc = numpy.asarray(
                            [float(x0[0] - mc[0]), float(x0[1] - mc[1]), float(x0[2] - mc[2])])  # 绕nOX0# 旋转轴
                        p1 = numpy.asarray([float(yc[0] - mc[0]), float(yc[1] - mc[1]), float(yc[2] - mc[2])])
                        nc = numpy.array(nc, dtype=numpy.float64)
                        n0 = numpy.array(n0, dtype=numpy.float64)
                    rotationAngle = 180
                    cos = round(math.cos(rotationAngle * math.pi / 180), 3)
                    sin = round(math.sin(rotationAngle * math.pi / 180), 3)
                    rotatedVector = -p1
                    new_yc = rotatedVector + mc
                    x0y0 = numpy.asarray(y0) - numpy.asarray(x0)
                    x0yc = numpy.asarray(new_yc) - numpy.asarray(x0)
                    x = np.array(x0y0, dtype=numpy.float64)
                    y = np.array(x0yc, dtype=numpy.float64)

                    Lx = np.sqrt(x.dot(x))
                    Ly = np.sqrt(y.dot(y))
                    cos_angle = x.dot(y) / (Lx * Ly)
                    angle = np.arccos(cos_angle)
                    angle2 = angle * 180 / np.pi
                    new_yc = numpy.asarray([round(new_yc[0], 3), round(new_yc[1], 3), round(new_yc[2], 3)])
                    rotatedVector = numpy.asarray([round(rotatedVector[0], 3), round(rotatedVector[1], 3), round(rotatedVector[2], 3)])
                    if yc == y0:
                        now_y0 = yc
                        now_new_y0 = list(new_yc)
        l.append(str(line.split()[0]))
        l.append(str(line.split()[1]).rjust(7, ' ') + '  ')
        l.append(str(line.split()[2]).ljust(4, ' '))

        l.append(str(line.split()[3]).ljust(3, ' '))
        l.append(str(line.split()[4]).rjust(2, ' '))  # ‘A'
        l.append(str(line.split()[5]).rjust(4, ' '))

        l.append(str('%.3f' % new_yc[0]).rjust(12, ' '))
        l.append(str('%.3f' % new_yc[1]).rjust(8, ' '))
        l.append(str('%.3f' % new_yc[2]).rjust(8, ' '))

        l.append(str(line[54:60].strip()).rjust(6, ' '))
        l.append(str(line[60:66].strip()).rjust(6, ' '))
        l.append(str(line[76:78].strip()).rjust(12, ' '))
        l.append('\n')

    new_f.writelines(l)
    new_f.close()
    coord = new_f
    return coord


def get_dist(refinedcoord, x0, aim_res_indx, y1):
    if type(refinedcoord) == str:
        mod = open(refinedcoord, "r")
    else:
        mod = open(refinedcoord.name, "r")
    mod_lines = mod.readlines()
    y0 = []
    for i in range(len(mod_lines)):
        if len(mod_lines[i].split()) > 9:
            if int(mod_lines[i].split()[5]) == int(aim_res_indx) and mod_lines[i].split()[2] == 'CA':
                m1_1 = float(mod_lines[i][30:38])
                m1_2 = float(mod_lines[i][38:46])
                m1_3 = float(mod_lines[i][46:54])
                break
    y0.append(m1_1)
    y0.append(m1_2)
    y0.append(m1_3)

    x0 = [float(x0[0]), float(x0[1]), float(x0[2])]
    y0 = [float(y0[0]), float(y0[1]), float(y0[2])]
    y1 = [float(y1[0]), float(y1[1]), float(y1[2])]
    x0 = [round(float(x0[0]), 3), round(float(x0[1]), 3), round(float(x0[2]), 3)]
    y0 = [round(float(y0[0]), 3), round(float(y0[1]), 3), round(float(y0[2]), 3)]
    y1 = [round(float(y1[0]), 3), round(float(y1[1]), 3), round(float(y1[2]), 3)]
    x, y, z = sympy.symbols("x y z")
    r = math.dist(x0, y0)

    c = (x - x0[0]) * (x - x0[0]) + (y - x0[1]) * (y - x0[1]) + (z - x0[2]) * (z - x0[2]) - r * r

    s = [y1[0] - x0[0], y1[1] - x0[1], y1[2] - x0[2]]

    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

    a = sympy.solve([c, l1, l2, l3], [x, y, z])

    if len(a) == 0:

        s = [round(s[0], 3), round(s[1], 3), round(s[2], 3)]
        l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
        l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
        l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]

        a = sympy.solve([c, l1, l2, l3], [x, y, z])
        if len(a) == 0:
            s = [round(s[0], 2), round(s[1], 2), round(s[2], 2)]
            l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
            l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
            l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
            a = sympy.solve([c, l1, l2, l3], [x, y, z])
            if len(a) == 0:
                s = [round(s[0], 1), round(s[1], 1), round(s[2], 1)]
                l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
                a = sympy.solve([c, l1, l2, l3], [x, y, z])
                if len(a) == 0:
                    s = [int(s[0]), int(s[1]), int(s[2])]
                    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
                    a = sympy.solve([c, l1, l2, l3], [x, y, z])
                    if len(a) == 0:
                        return False, False, False
    if math.dist(a[0], y1) < math.dist(a[1], y1):
        w = a[0]
    else:
        w = a[1]

    ws = [w[0] - x0[0], w[1] - x0[1], w[2] - x0[2]]
    ws = numpy.array(ws, dtype=numpy.float64)
    s = numpy.array(s, dtype=numpy.float64)
    # print(is_parallel(ws, s), file=log)
    new_y0 = w
    dist = math.dist(x0, new_y0)
    res_y0 = y0
    aim_res_y0 = new_y0
    return dist, res_y0, aim_res_y0


def get_next_y0(refinedcoord, x0, y1, r, res_indx):
    if type(refinedcoord) == str:
        mod = open(refinedcoord, "r")
    else:
        mod = open(refinedcoord.name, "r")
    mod_lines = mod.readlines()
    y0 = []
    for i in range(len(mod_lines)):
        if len(mod_lines[i].split()) > 9:
            if int(mod_lines[i].split()[5]) == int(res_indx) and mod_lines[i].split()[2] == 'CA':
                m1_1 = float(mod_lines[i][30:38])
                m1_2 = float(mod_lines[i][38:46])
                m1_3 = float(mod_lines[i][46:54])
                break
    y0.append(m1_1)
    y0.append(m1_2)
    y0.append(m1_3)

    x0 = [float(x0[0]), float(x0[1]), float(x0[2])]
    y0 = [float(y0[0]), float(y0[1]), float(y0[2])]
    y1 = [float(y1[0]), float(y1[1]), float(y1[2])]
    x0 = [round(float(x0[0]), 3), round(float(x0[1]), 3), round(float(x0[2]), 3)]
    y0 = [round(float(y0[0]), 3), round(float(y0[1]), 3), round(float(y0[2]), 3)]
    y1 = [round(float(y1[0]), 3), round(float(y1[1]), 3), round(float(y1[2]), 3)]

    x, y, z = sympy.symbols("x y z")
    c = (x - x0[0]) * (x - x0[0]) + (y - x0[1]) * (y - x0[1]) + (z - x0[2]) * (z - x0[2]) - r * r
    s = [y1[0] - x0[0], y1[1] - x0[1], y1[2] - x0[2]]
    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
    a = sympy.solve([c, l1, l2, l3], [x, y, z])
    if len(a) == 0:
        s = [round(s[0], 3), round(s[1], 3), round(s[2], 3)]
        l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
        l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
        l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
        a = sympy.solve([c, l1, l2, l3], [x, y, z])
        if len(a) == 0:
            s = [round(s[0], 2), round(s[1], 2), round(s[2], 2)]
            l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
            l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
            l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
            a = sympy.solve([c, l1, l2, l3], [x, y, z])
            if len(a) == 0:
                s = [round(s[0], 1), round(s[1], 1), round(s[2], 1)]
                l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
                a = sympy.solve([c, l1, l2, l3], [x, y, z])
                if len(a) == 0:
                    s = [int(s[0]), int(s[1]), int(s[2])]
                    l1 = s[1] * x - s[1] * x0[0] - s[0] * y + s[0] * x0[1]
                    l2 = s[2] * y - s[2] * x0[1] - s[1] * z + s[1] * x0[2]
                    l3 = s[2] * x - s[2] * x0[0] - s[0] * z + s[0] * x0[2]
                    a = sympy.solve([c, l1, l2, l3], [x, y, z])
                    if len(a) == 0 or type(a[0]!=float):
                        return False, False
    if math.dist(a[0], y1) < math.dist(a[1], y1):
        w = a[0]
    else:
        w = a[1]
    ws = [w[0] - x0[0], w[1] - x0[1], w[2] - x0[2]]
    ws = numpy.array(ws, dtype=numpy.float64)
    s = numpy.array(s, dtype=numpy.float64)
    # print(is_parallel(ws, s), file=log)
    new_y0 = w
    res_y0 = y0
    aim_res_y0 = new_y0
    return res_y0, aim_res_y0


# stretch
def helix_long_transform(fixedcoord1, fixedcoord2, refinedcoord, x1, y1, x0, y0, orig_modcoord):
    if type(refinedcoord) == str:
        mod = open(refinedcoord, "r")
    else:
        mod = open(refinedcoord.name, "r")
    mod_lines = mod.readlines()
    if len(mod_lines[len(mod_lines) - 1].split()) != len(mod_lines[0].split()):
        indx = len(mod_lines) - 3
    else:
        indx = len(mod_lines) - 1
    first_res = int(mod_lines[0].split()[5])
    last_res = int(mod_lines[indx].split()[5])

    res_num = last_res - first_res + 1
    # print('first_res,last_res,res_num:', first_res, last_res, res_num, file=log)
    if (res_num - 1) * 3.8 < math.dist(x1, y1) - 2 * 3.8:
        return False
    else:
        dist_dict = []
        res_y0_dict = []
        aim_res_y0_dict = []
        aim_res_indx_dict = []
        start_res_indx_dict = []
        for i in range(res_num - 1):
            aim_res_indx = last_res - i - 1
            if aim_res_indx == first_res:
                dist_dict.append((res_num - 1) * 3.8)
                res_y0_dict.append(y0)
                aim_res_y0_dict.append(y0)
                aim_res_indx_dict.append(first_res)
                start_res_indx_dict.append(first_res + 1)
            else:
                dist, res_y0, aim_res_y0 = get_dist(refinedcoord, x0, aim_res_indx, y1)
                if dist == False:
                    return False

                dist_dict.append(dist + 3.8 * (i + 1))
                res_y0_dict.append(res_y0)
                aim_res_y0_dict.append(aim_res_y0)
                aim_res_indx_dict.append(aim_res_indx)
                start_res_indx_dict.append(aim_res_indx + 1)

        for i in range(len(dist_dict)):
            if dist_dict[i] >= math.dist(x1, y1) - 2 * 3.8:
                x = i + 1
                aim_res_indx = aim_res_indx_dict[i]
                start_res_indx = start_res_indx_dict[i]
                res_y0 = res_y0_dict[i]
                aim_res_y0 = aim_res_y0_dict[i]
                break
        refinedcoord = long_rotate(x0, res_y0, aim_res_y0, refinedcoord)
        x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

        start_res_indx = aim_res_indx + 1
        if type(refinedcoord) == str:
            mod = open(refinedcoord, "r")
        else:
            mod = open(refinedcoord.name, "r")

        lines = mod.readlines()

        for i in range(len(mod_lines)):
            if int(lines[i].split()[5]) == int(start_res_indx):
                start_res_line_indx = i
                break
        if type(refinedcoord) == str:
            f = open(refinedcoord, "r")
        else:
            f = open(refinedcoord.name, "r")
        lines = f.readlines()

        if type(refinedcoord) == str:
            fpath, fname = os.path.split(refinedcoord)
        else:
            fpath, fname = os.path.split(refinedcoord.name)
        new_coord = '%s/' % model_dir + fname[:-4] + '_strech.pdb'

        if type(new_coord) == str:
            new_f = open(new_coord, "w")
        else:
            new_f = open(new_coord.name, "w")

        for i in range(start_res_line_indx):
            line = lines[i]
            new_f.writelines(line)
        lines = lines[start_res_line_indx:]

        translation_dict = []
        for i in range(x):
            r = 3.8 * (i + 1)
            next_res_indx = aim_res_indx + i + 1
            next_res_y0, next_aim_res_y0 = get_next_y0(refinedcoord, aim_res_y0, y1, r, next_res_indx)
            if next_res_y0 == False:
                return False
            # t = list(next_aim_res_y0) -list(next_res_y0)
            t = [float(next_aim_res_y0[0]) - float(next_res_y0[0]), float(next_aim_res_y0[1]) - float(next_res_y0[1]),
                 float(next_aim_res_y0[2]) - float(next_res_y0[2])]

            translation_dict.append(t)
        l = []
        count = 0
        for i in range(0, len(lines)):
            # print(lines[i])
            if len(lines[i].split()) > 9:
                # translation=[1,-2,0]
                count += 1
                l.append(str(lines[i].split()[0]))
                l.append(str(lines[i].split()[1]).rjust(7, ' ') + '  ')
                l.append(str(lines[i].split()[2]).ljust(4, ' '))

                l.append(str(lines[i].split()[3]).ljust(3, ' '))
                l.append(str(lines[i].split()[4]).rjust(2, ' '))  # ‘A'
                l.append(str(lines[i].split()[5]).rjust(4, ' '))

                translation = translation_dict[int(lines[i].split()[5]) - int(start_res_indx)]

                l.append(str('%.3f' % (float(lines[i][30:38]) + translation[0])).rjust(12, ' '))
                l.append(str('%.3f' % (float(lines[i][38:46]) + translation[1])).rjust(8, ' '))
                l.append(str('%.3f' % (float(lines[i][46:54]) + translation[2])).rjust(8, ' '))

                l.append(str(lines[i][54:60].strip()).rjust(6, ' '))
                l.append(str(lines[i][60:66].strip()).rjust(6, ' '))
                l.append(str(lines[i][76:78].strip()).rjust(12, ' '))
                l.append('\n')
        new_f.writelines(l)

        while count < len(lines):
            new_f.write(str(lines[count]))
            count = count + 1
        new_f.close()
        refinedcoord = new_f
        x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
    return refinedcoord


def transform(fixedcoord1, modifiedcoord, refinedcoord, fixedcoord2, orig_modcoord,flag):
    print('\ntranslate and rotate```````````````````', file=log)
    d = 3.8
    d_arr = numpy.asarray([math.sqrt(math.pow(d, 2) / 3), math.sqrt(math.pow(d, 2) / 3), math.sqrt(math.pow(d, 2) / 3)])  # [2.19393102 2.19393102 2.19393102]
    x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
    f1_x1 = get_f1_x1(fixedcoord1)
    f2_y1 = get_f2_y1(fixedcoord2)

    if len(f1_x1) == 0:
        refinedcoord = translate(coord1=modifiedcoord, coord2=refinedcoord, x0=y0, x1=y1)
        x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

    elif len(f2_y1) == 0:
        refinedcoord = translate(coord1=modifiedcoord, coord2=refinedcoord, x0=x0, x1=x1)
        x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

    else:
        if flag=='OLD':
            refinedcoord = translate(coord1=modifiedcoord, coord2=refinedcoord, x0=x0, x1=x1)
            x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
            refinedcoord1 = refinedcoord
            refinedcoord = rotate(x0, y0, y1, refinedcoord)
            if refinedcoord == False:
                refinedcoord = refinedcoord1
            x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

        else:
            d_x1y1 = math.dist(x1, y1)
            len_helix = math.dist(x0, y0)
            if len_helix >= d_x1y1:
                refinedcoord = translate(coord1=modifiedcoord, coord2=refinedcoord, x0=x0, x1=x1)
                x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
                refinedcoord1 = refinedcoord
                refinedcoord = rotate(x0, y0, y1, refinedcoord)
                if refinedcoord == False:
                    refinedcoord = refinedcoord1
                x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

            elif len_helix < d_x1y1:
                refinedcoord = translate(coord1=modifiedcoord, coord2=refinedcoord, x0=x0, x1=x1)
                x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
                refinedcoord1 = refinedcoord
                refinedcoord = rotate(x0, y0, y1, refinedcoord)
                if refinedcoord == False:
                    refinedcoord = refinedcoord1
                refinedcoord2 = refinedcoord
                x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
                refinedcoord = helix_long_transform(fixedcoord1, fixedcoord2, refinedcoord, x1, y1, x0, y0,orig_modcoord)
                if refinedcoord == False:
                    refinedcoord = refinedcoord2
                x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

        if y0 != y1:
            fixedcoord2 = translate(coord1=fixedcoord2, x0=f2_y1, x1=y0 + d_arr)
            x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)

        x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
        f1_x1 = get_f1_x1(fixedcoord1)
        f2_y1 = get_f2_y1(fixedcoord2)
        x0y0 = numpy.array([y0[0] - x0[0], y0[1] - x0[1], y0[2] - x0[2]], dtype=numpy.float64)
        x1y1 = numpy.array([y1[0] - x1[0], y1[1] - x1[1], y1[2] - x1[2]], dtype=numpy.float64)
        # print(is_parallel(x0y0, x1y1), file=log)
    if type(fixedcoord1) != str:
       fixedcoord1 = fixedcoord1.name
    if type(refinedcoord) != str:
        refinedcoord = refinedcoord.name
    if type(fixedcoord2) != str:
        fixedcoord2 = fixedcoord2.name
    print('structure_a:%s\nrefined_local_structure:%s\nstructure_b:%s    SAVED!' %(fixedcoord1,refinedcoord ,fixedcoord2),file=log)
    return fixedcoord2, refinedcoord


def combine(orig_modcoord,fixedcoord1, refinedcoord, fixedcoord2,seqnum):
    print('\n', file=log)
    print('combine```````````````````', file=log)
    d = 3.8
    d_arr = numpy.asarray([math.sqrt(math.pow(d, 2) / 3), math.sqrt(math.pow(d, 2) / 3), math.sqrt(math.pow(d, 2) / 3)])  # [2.19393102 2.19393102 2.19393102]
    x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
    f1_x1=get_f1_x1(fixedcoord1)
    f2_y1 = get_f2_y1(fixedcoord2)
    if len(f1_x1)!= 0:
        intervalFront = math.dist(x0, f1_x1)
        intervalFront = float(int(intervalFront*10)/10)
        if float(intervalFront)>3.9 or float(intervalFront)<3.7:
            x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
            fixedcoord1 = translate(coord1=fixedcoord1, x0=f1_x1, x1=x0 + d_arr)
    if len(f2_y1)!= 0:
        intervalBack = math.dist(y0, f2_y1)
        intervalBack = float(int(intervalBack*10)/10)
        if float(intervalBack)>3.9 or float(intervalBack)<3.7:
            x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
            fixedcoord2 = translate(coord1=fixedcoord2, x0=f2_y1, x1=y0 + d_arr)

    x0, m0, y0, x1, y1 = getkeycoord(orig_modcoord, refinedcoord)
    f1_x1 = get_f1_x1(fixedcoord1)
    f2_y1 = get_f2_y1(fixedcoord2)
    if len(f1_x1) != 0:
        intervalFront = math.dist(x0, f1_x1)
    else:
        intervalFront=3.8
    if len(f2_y1) != 0:
        intervalBack = math.dist(y0, f2_y1)
    else:
        intervalBack=3.8
    intervalFront = float(int(intervalFront*10)/10)
    intervalBack = float(int(intervalBack*10)/10)

    if float(intervalBack)<=3.9 and float(intervalBack)>=3.7  and float(intervalFront)<=3.9  and float(intervalBack)>=3.7 :

        pdbfile = '%s/final_refined_pdb/%s_refined.pdb' % (model_dir, seqnum)
        if type(pdbfile) == str:
            f = open(pdbfile, "w")
        else:
            f = open(pdbfile.name, "w")

        if type(fixedcoord1) == str:
            fix1 = open(fixedcoord1, "r")
        else:
            fix1 = open(fixedcoord1.name, "r")
        f1_lines = fix1.readlines()

        if type(refinedcoord) == str:
            mod = open(refinedcoord, "r")
        else:
            mod = open(refinedcoord.name, "r")
        mod_lines = mod.readlines()

        if type(fixedcoord2) == str:
            fix2 = open(fixedcoord2, "r")
        else:
            fix2 = open(fixedcoord2.name, "r")
        f2_lines = fix2.readlines()
        if len(f1_lines) != 0:
            for line in f1_lines:
                f.write(str(line))

        for line in mod_lines:
            f.write(str(line))

        if len(f2_lines) != 0:
            for line in f2_lines:
                f.write(str(line))
        fix1.close()
        mod.close()
        fix2.close()
        f.close()
        refinedpdb = pdbfile
        print('%s    SAVED!'%refinedpdb, file=log)
        return refinedpdb
    else:
        return False


# the translation and rotation strategy
def pdb_reconstruction(pdbfile, LineIndex,  ResidueIndex,flag,i):
    fixedcoord1, modifiedcoord, fixedcoord2 = separate(pdbfile, LineIndex, i)
    orig_modcoord = modifiedcoord
    f1_x1 = get_f1_x1(fixedcoord1)
    f2_y1 = get_f2_y1(fixedcoord2)
    if len(f1_x1) == 0 and len(f2_y1) == 0:
        refinedpdb = refine(pdbfile, ResidueIndex, seqnum=i)
        return refinedpdb

    refinedcoord = refine(modifiedcoord,  ResidueIndex,i)
    if flag=='OLD':
        fixedcoord2, refinedcoord = transform(fixedcoord1, modifiedcoord, refinedcoord, fixedcoord2, orig_modcoord,flag='OLD')
    else:
        fixedcoord2, refinedcoord = transform(fixedcoord1, modifiedcoord, refinedcoord, fixedcoord2, orig_modcoord,flag='NEW')

    if fixedcoord1 == False or fixedcoord1 == False or refinedcoord==False:
        return False
    refinedpdb = combine(orig_modcoord,fixedcoord1, refinedcoord, fixedcoord2, i)
    return refinedpdb


def generate_slices(s):
    slices=[]
    left = 0
    right = left+1
    char_win = list()
    while right < len(s):
        char_win.append(s[right])
        if s[right] == s[left]:
            right += 1
        else:
            slices.append(s[left:right])
            left = right
            right += 1
    slices.append(s[left:right])
    return slices


def MainElem(s):
    count_C,count_E,count_H=0,0,0
    for i in range(0,len(s)):
        if s[i] == 'C':
            count_C += 1
        elif s[i] == 'E':
            count_E += 1
        else:
            count_H += 1
    if(count_C >= count_E):
        main_elem_count = count_C
        main_elem = 'C'
    else:
        main_elem_count = count_E
        main_elem = 'E'
    if (count_H > main_elem_count):
        main_elem_count = count_H
        main_elem = 'H'
    return main_elem


def results(slices):
    dividlist=[[] for x in range(0,len(slices))]
    start=1
    for i in range(0,len(slices)):
        dividlist[i].append(MainElem(slices[i]))
        dividlist[i].append(len(slices[i]))
        dividlist[i].append(start)
        end=start+len(slices[i])-1
        dividlist[i].append(end)
        start=end+1
    return dividlist


def identifyUnmatchHelix(statingSS,targertSS):
    trained_helix_len = [x for x in range(2,38)]
    unmatch = False
    unmatch_seg = []
    for i in range(0, len(statingSS)):
        if statingSS[i] != targertSS[i]:
            if unmatch == False:
                start = i
                end = i
                unmatch = True
            else:
                end = i
        else:
            if unmatch == True:
                unmatch_seg.append([start, end])
                unmatch = False

    # print(targertSS, len(targertSS), file=log)

    # ss_data_str = []
    # ss_data_str.append(''.join(targertSS))
    # print('original ss_str      :', ss_data_str)
    #
    # slices = generate_slices(targertSS)
    # slice_str = []
    # for j in range(len(slices)):
    #     slice_str.append(''.join(slices[j]))
    # print('original ss_partition:', slice_str)
    target_ss_list = results(generate_slices(targertSS))
    # print(target_ss_list, file=log)

    unmatchHexlix = []
    k = 0
    count = target_ss_list[k][1]
    for i in range(0, len(unmatch_seg)):
        for j in range(unmatch_seg[i][0], unmatch_seg[i][1] + 1):
            # print(j)
            if j >= count:
                while j >= count:
                    k += 1
                    count += target_ss_list[k][1]
            unmatchHexlixSeg = [target_ss_list[k][2], target_ss_list[k][3]]
            if unmatchHexlixSeg not in unmatchHexlix and target_ss_list[k][0] == 'H' and int(
                    target_ss_list[k][1]) > 5 and int(target_ss_list[k][1]) in trained_helix_len:
                unmatchHexlix.append(unmatchHexlixSeg)
    return unmatchHexlix


def findLineIndex(startingModel,startResidueIndex,endResidueIndex):
    startflag = False
    with open(startingModel, 'r') as f:
        for lineindex, line in enumerate(f, start=1):
            if len(line.split()) >= 9:
                if startflag == False and int(line.split()[5]) == startResidueIndex:
                    startflag = True
                    startLineIndex = lineindex
                if int(line.split()[5]) == endResidueIndex:
                    endLineIndex = lineindex
                lineindex += 1
    f.close()
    return startLineIndex, endLineIndex


def refineUnmatch(unmactHelixSegs):
    pdbfile=startingModel
    for j in range(len(unmactHelixSegs)):
        unmactHelixSeg=unmactHelixSegs[j]
        print('\n\n----refine idnconsisitent local structure %d' % (unmactHelixSegs.index(unmactHelixSeg) + 1), file=log)
        startResidueIndex, endResidueIndex = unmactHelixSeg[0], unmactHelixSeg[1]
        print('--------------residue index:', [startResidueIndex, endResidueIndex], file=log)
        startLineIndex, endLineIndex = findLineIndex(startingModel,startResidueIndex, endResidueIndex)
        ResidueIndex = [startResidueIndex, endResidueIndex]
        lineIndex=[startLineIndex, endLineIndex]
        if startResidueIndex == 1 and endResidueIndex == lastResidue:
            refinedpdb = refine(pdbfile, ResidueIndex, seqnum=j + 1)
        else:
            if int(startResidueIndex) - int(firstResidue) < max(5, 0.05 * (int(lastResidue) - int(firstResidue) + 1)) or int(lastResidue) - int(endResidueIndex) < max(5, 0.05 * (int(lastResidue) - int(firstResidue) + 1)):
                refinedpdb = pdb_reconstruction(pdbfile, lineIndex,  ResidueIndex,'OLD',j+1)
            else:
                refinedpdb = pdb_reconstruction(pdbfile, lineIndex,  ResidueIndex,'NEW',j+1)
        lastpdbfie=pdbfile
        pdbfile = refinedpdb
        if pdbfile == False:
            pdbfile=lastpdbfie
            break
    return pdbfile


def run_dssp(pdbfile, dsspout):
    # mkdssp -i 1crn.pdb -o 1crn.dssp
    cmd = 'mkdssp -i ' + pdbfile + ' -o ' + dsspout
    print('running dssp to fetch initial secondary structure :',cmd)
    os.system(cmd)


def convert_dssp_to_three_ss(ss8):
    ss3=[]
    for i in range(len(ss8)):
        cha=ss8[i]
        if cha== "G" or cha== "H" or cha== "I": # helix
            cha="H"
        elif cha== "E" or cha== "B": # strand
            cha="E"
        else:
            cha= "C"
        ss3.append(cha)
    return ss3


def read_dssp(dssp_file):
    f=open(dssp_file,'r')
    lines=f.readlines()
    aa=[]
    ss8=[]
    for i in range(len(lines)):

        if lines[i].split()[0]=='#':
            # print(lines[i])
            start_index=i
            dssp_lines=lines[start_index:]
            break

    for i in range(1,len(dssp_lines)):
        line=dssp_lines[i]
        a = line[13:14]
        ss = line[16:17]
        if ss==" ":
            ss="."
        if a!='!':
            aa.append(str(a))
            ss8.append(str(ss))
    ss3 = convert_dssp_to_three_ss(ss8)
    return aa,ss3


def runPSIPRED(running_dir, fasta_path):
        os.chdir(running_dir)
        cmd = '../BLAST+/runpsipredplus ' + fasta_path
        print('running psipred to predict target secondary structure :', cmd)
        os.system(cmd)
        os.chdir(AnglesRefine_dir)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="path of starting model.")
    parser.add_argument("output", type=str, help="path of output folder.")
    parser.add_argument("-select", type=str, default=None, help="autunomous refine : refine the selected local structures to refine. e.g., -select 1,2")
    parser.add_argument("-show", action="store_true", default=False, help="show the information(sequence number,[startResidueIndex, endResidueIndex], length) of inconsistent loacl structures identified to refine.")
    parser.add_argument("-save_log", action="store_true", default=False, help="save log file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    print('device: ',device)

    # AnglesRefine_dir = '/mnt/i/AnglesRefine'
    AnglesRefine_dir = os.path.dirname(os.path.abspath(__file__))+'/'
    print('AnglesRefine_path:',AnglesRefine_dir)
    Helix_angle_Transformer_dir = AnglesRefine_dir + 'model'
    PSIPRED_dir = AnglesRefine_dir + 'tools/psipred'
    startingModel = AnglesRefine_dir + args.input
    model_dir, model_name = os.path.split(startingModel)
    model_dir = AnglesRefine_dir + args.output + model_name[:-4]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log = open(model_dir + '/log.txt', mode='w', encoding='utf-8')
    print('Starting Model: ', startingModel)
    print('device: ', device, file=log)
    print('\nAnglesRefine_path:',AnglesRefine_dir, file=log)
    print('\nINPUT: ',startingModel, file=log)
    print('\nStarting Model: ', model_name, file=log)

    # pdb2fasta, pdb2ss3
    model_name = model_name[:-4]
    dssp = model_dir + '/%s.dssp' % model_name
    ss3 = model_dir + '/%s.startingSS' % model_name
    fasta_path = model_dir + '/%s.fasta' % model_name
    with suppress_stdout_stderr():
        run_dssp(startingModel, dssp)
    aa, ss = read_dssp(dssp)
    if not os.path.exists(fasta_path):
        f = open(fasta_path, 'a')
        for a in aa:
            f.write('%c' % a)
        f.close()
    if not os.path.exists(ss3):
        f = open(ss3, 'a')
        for s in ss:
            f.write('%c' % s)
        f.close()
    f = open(startingModel, "r")
    lines = f.readlines()
    firstResidue=int(lines[0].split()[5])
    print('\nfirst residue index：', firstResidue, file=log)


    # fasta
    f = open(fasta_path, 'r')
    fasta = f.read()
    f.close()
    # startingSS
    f = open(ss3, 'r')
    startingSS = f.read()
    f.close()
    lastResidue=int(len(fasta)+firstResidue-1)
    print('\nlength：', len(fasta), file=log)
    print('\nfasta:', fasta, len(fasta), file=log)
    print('\nstartingSS:', startingSS, len(startingSS), file=log)

    # predict target ss by PSIPRED
    print('\n\nPredicting target ss by PSIPRED------------------ ', file=log)
    running_dir = PSIPRED_dir + '/example'
    with suppress_stdout_stderr():
        runPSIPRED(running_dir, fasta_path)
    psipred_ss2 = running_dir + '/%s.ss2' % model_name
    f = open(psipred_ss2, 'r')
    lines = f.readlines()
    f.close()
    ss2 = model_dir + '/%s.targetSS' % model_name
    f = open(ss2, 'w')
    for i in range(2, len(lines)):
        line = lines[i]
        s = line.split()[2]
        f.write('%c' % s)
    f.close()
    # targetSS
    f = open(ss2, 'r')
    targetSS = f.read()
    f.close()
    print('\ntargetSS:', targetSS, len(targetSS), file=log)
    print('\n%s SAVED! '%ss2, file=log)

    # target ss to unmatch helix segment residue index
    print('\n\nGenerating inconsistent local structures whose target secondary structures are Helix------------------ ', file=log)
    unmactHelixSegs = identifyUnmatchHelix(startingSS, targetSS)

    # Refine
    if len(unmactHelixSegs) == 0:
        print('starting model has no inconsistent local structure whose target secondary structure is Helix!')
        print('\nstarting model has no inconsistent local structure whose target secondary structure is Helix!', file=log)
        # refinedModel = startingModel
        # RefinedModel = AnglesRefine_dir + args.output + 'refined_%s.pdb' % model_name
        # shutil.copy(refinedModel, RefinedModel)
        # print('Refined Model: ', RefinedModel, file=log)
        # print('Refined Model: ', RefinedModel)
        print('Skip refinement, no output refined model!')
        print('\nSkip refinement, no output refined model!', file=log)
    else:
        print('\nstarting model has %s inconsistent local structures whose target secondary structures are Helix ' % len(unmactHelixSegs), file=log)
        print('-----------------------------INFO-----------------------------', file=log)
        print('SequenceNumber---------residue index--------length:', file=log)
        for j in range(len(unmactHelixSegs)):
            unmactHelixSeg = unmactHelixSegs[j]
            startResidueIndex, endResidueIndex = unmactHelixSeg[0], unmactHelixSeg[1]
            print('%s------------------------%s---------------%s' % (
            j + 1, [startResidueIndex, endResidueIndex], endResidueIndex - startResidueIndex + 1), file=log)

        if args.show:
            print('\nstarting model has %s inconsistent local structures whose target secondary structures are Helix' % len(unmactHelixSegs))
            print('-----------------------------INFO-----------------------------')
            print('SequenceNumber---------residue index--------length:')
            for j in range(len(unmactHelixSegs)):
                unmactHelixSeg = unmactHelixSegs[j]
                startResidueIndex, endResidueIndex = unmactHelixSeg[0], unmactHelixSeg[1]
                print('%s------------------------%s---------------%s'%(j+1, [startResidueIndex, endResidueIndex],endResidueIndex-startResidueIndex+1))

        elif args.select != None:
            #  Refine_userAutonomy
            print('refining (UserAutonomy-Mode)------------------ ')
            print('\n\nrefining (UserAutonomy-Mode)------------------ ',file=log)
            pickedunmactHelixSegs = []
            pickedunmactHelixSeqNumList = args.select
            pickedunmactHelixSeqNumList= pickedunmactHelixSeqNumList.split(",")
            for i in range(len(pickedunmactHelixSeqNumList)):
                pickedunmactHelixSegs.append(unmactHelixSegs[int(pickedunmactHelixSeqNumList[i]) - 1])
            unmactHelixSegs = pickedunmactHelixSegs
            print('The following inconsistent local structures are ready to refine to Helix:', file=log)
            print('-----------------------------INFO-----------------------------', file=log)
            print('SequenceNumber---------residue index--------length:',file=log)
            for j in range(len(unmactHelixSegs)):
                unmactHelixSeg = unmactHelixSegs[j]
                startResidueIndex, endResidueIndex = unmactHelixSeg[0], unmactHelixSeg[1]
                print('%s------------------------%s---------------%s' % (
                j + 1, [startResidueIndex, endResidueIndex], endResidueIndex - startResidueIndex + 1),file=log)
            refinedModel = refineUnmatch(unmactHelixSegs)
            shutil.copy(refinedModel, model_dir + '/final_refined_pdb/refined_%s.pdb' % model_name)
            RefinedModel = AnglesRefine_dir + args.output + 'refined_%s.pdb' % model_name
            shutil.copy(refinedModel, RefinedModel)
            print('Done (UserAutonomy-Mode)------------------ ')
            print('Refined Model: ', RefinedModel)
            print('\n\nDone (UserAutonomy-Mode)------------------ ', file=log)
            print('\nOUTPUT: ', RefinedModel, file=log)
            print('\nRefined Model: ', 'refined_%s.pdb' % model_name, file=log)
        else:
            #  Refine_default
            print('refining (Default-Mode)------------------ ')
            print('\n\nrefining (Default-Mode)------------------ ', file=log)
            print('The following inconsistent local structures are ready to refine to Helix:', file=log)
            print('-----------------------------INFO-----------------------------', file=log)
            print('SequenceNumber---------residue index--------length:',file=log)
            for j in range(len(unmactHelixSegs)):
                unmactHelixSeg = unmactHelixSegs[j]
                startResidueIndex, endResidueIndex = unmactHelixSeg[0], unmactHelixSeg[1]
                print('%s------------------------%s---------------%s' % (
                j + 1, [startResidueIndex, endResidueIndex], endResidueIndex - startResidueIndex + 1),file=log)
            refinedModel = refineUnmatch(unmactHelixSegs)
            shutil.copy(refinedModel, model_dir + '/final_refined_pdb/refined_%s.pdb' % model_name)
            RefinedModel=AnglesRefine_dir + args.output+'refined_%s.pdb'%model_name
            shutil.copy(refinedModel, RefinedModel)
            print('Done (Default-Mode)------------------ ')
            print('Refined Model: ', RefinedModel)
            print('\n\nDone (Default-Mode)------------------ ', file=log)
            print('\nOUTPUT: ', RefinedModel, file=log)
            print('\nRefined Model: ', 'refined_%s.pdb'%model_name, file=log)
        if args.save_log==False:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)






