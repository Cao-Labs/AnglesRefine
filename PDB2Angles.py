import sys
import os
from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
# from Bio.PDB.Vector import *
from Bio.PDB.Entity import*
import math
from PeptideBuilder import Geometry
import PeptideBuilder
import numpy
from os import path

resdict = { 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', \
	    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', \
	    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', \
	    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y' }

def extract_backbone_model(pdb_path, angle_out):
    parser=PDBParser()
    structure=parser.get_structure('sample', pdb_path)
    model=structure[0]
    chain=model['A']
    model_structure_geo=[]
    prev="0"
    N_prev="0"
    CA_prev="0"
    CO_prev="0"
    ##O_prev="0"
    prev_res=""
    rad=180.0/math.pi
    fh = open(angle_out, "w")
    ### now first print the headers, please filter them when you really load the file ### 
    headers = "# residue CA_C_N_angle C_N_CA_angle CA_N_length CA_C_length peptide_bond psi_im1 omega phi CA_N_length CA_C_length N_CA_C_angle\n"
    fh.write(headers) 
    for res in chain:
        if(res.get_resname() in resdict.keys()):
            geo=Geometry.geometry(resdict[res.get_resname()])
            if(prev=="0"):
                 N_prev=res['N']
                 CA_prev=res['CA']
                 C_prev=res['C']
                 ##O_prev=res['O']
                 prev="1"
            else:
                 n1=N_prev.get_vector()
                 ca1=CA_prev.get_vector()
                 c1=C_prev.get_vector()
                 ##o1=O_prev.get_vector()
                            
                 ##O_curr=res['O']
                 C_curr=res['C']
                 N_curr=res['N']
                 CA_curr=res['CA']
                                             
                 ##o=O_curr.get_vector()
                 c=C_curr.get_vector()
                 n=N_curr.get_vector()
                 ca=CA_curr.get_vector()

                 geo.CA_C_N_angle=calc_angle(ca1, c1, n)*rad
                 geo.C_N_CA_angle=calc_angle(c1, n, ca)*rad
                 geo.CA_N_length= CA_curr-N_curr
                 geo.CA_C_length= CA_curr-C_curr
                 geo.peptide_bond= N_curr-C_prev

                 psi= calc_dihedral(n1, ca1, c1, n) ##goes to current res
                 omega= calc_dihedral(ca1, c1, n, ca) ##goes to current res
                 phi= calc_dihedral(c1, n, ca, c) ##goes to current res

                 geo.psi_im1=psi*rad
                 geo.omega=omega*rad
                 geo.phi=phi*rad

                 geo.CA_N_length= CA_curr - N_curr
                 geo.CA_C_length= CA_curr - C_curr 
                 ##geo.C_O_length= C_curr - O_curr

                 geo.N_CA_C_angle= calc_angle(n, ca, c)*rad
                 ##geo.CA_C_O_angle= calc_angle(ca, c, o)*rad

                 ##geo.N_CA_C_O= calc_dihedral(n, ca, c, o)*rad

                 N_prev=res['N']
                 CA_prev=res['CA']
                 C_prev=res['C']
                 ##O_prev=res['O']
            # now write the angles to file 
            fh.write(str(resdict[res.get_resname()]) + " ")
            fh.write(str(geo.CA_C_N_angle) + " ")
            fh.write(str(geo.C_N_CA_angle) + " ") 
            fh.write(str(geo.CA_N_length) + " ")
            fh.write(str(geo.CA_C_length) + " ")
            fh.write(str(geo.peptide_bond)+ " ")
            fh.write(str(geo.psi_im1) + " ")
            fh.write(str(geo.omega) + " ")
            fh.write(str(geo.phi) + " ")
            fh.write(str(geo.CA_N_length) + " ")
            fh.write(str(geo.CA_C_length) + " ")
            fh.write(str(geo.N_CA_C_angle) + "\n")
           
                                         
            model_structure_geo.append(geo)
    fh.close()
    return model_structure_geo

def filterPDB(input,output):
    fh = open(input,"r")
    fh_out = open(output,"w")
    for line in fh:
       if len(line) < 65 or line[0:4].upper()!="ATOM":
          # 22.10.21
          # print("filtering "+line)
            pass
       else:
          line = line[0:21] + "A" + line[22:]
          fh_out.write(line) 
    fh.close()
    fh_out.close()

if __name__ == "__main__":
    if(len(sys.argv)<3):
       print("This script need two inputs, the first is the path of PDB file, and the second is the output of angles.\n")
       print("For example:\n")
       print("python "+sys.argv[0]+" ../data/server01_TS1 ../result/angles_server01")
       sys.exit(0)
    pdb_input_path = sys.argv[1]
    angles_out = sys.argv[2]
    # filter the input models, only keep ATOM files and add set chain to A. 
    pdb_input_path_new = sys.argv[1]+".tmp"
    filterPDB(pdb_input_path, pdb_input_path_new)
    # extract all angles and bond lengths
    structure_backbone = extract_backbone_model(pdb_input_path_new, angles_out)
    os.system("rm "+pdb_input_path_new)
    print(structure_backbone)

