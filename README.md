<div align="center">
  
# AnglesRefine: refinement of 3D protein structures using Transformer based on torsion angles


  
</div>

## Description
AnglesRefine, a non-physics-based protein structure refinement using protein secondary structures and torsion angles.

##Requirements
```bash
* python >=3.9
* pytorch >=1.10.0
* numpy
* Sympy
* PeptideBuilder
* biopython
* DSSP
* PSIPRED4.0
```

## Installation
```bash
To install AnglesRefine and it's dependencies following commands can be used in terminal:
1.`git clone https://github.com/Cao-Labs/AnglesRefine.git`
2.`cd AnglesRefine`
To download the model check points  to 'AnglesRefine/model' use the following commands in the terminal:
3. `wget https://cs.plu.edu/~caora/temp/model.zip`
4. `mkdir model && unzip model.zip -d model/`
To install the dependencies and create a conda environment use the following commands
5.`conda create -n AnglesRefine python=3.9`
6.`conda activate AnglesRefine`
## Numpy
7.`conda install numpy`
## Sympy
8.`conda install sympy`
## Pytorch
if GPU computer:
9.`conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch`
for CPU only 
9.`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
## Biopython
10.`conda install biopython`
## PeptideBuilder
11.`pip install PeptideBuilder`
## DSSP
12.`conda install -c salilab dssp`
## PSIPRED
First，download database(swissprot) to 'AnglesRefine/tools/db'
13.`mkdir tools && cd tools`
14.`mkdir db && cd db`
15.`wget http://ftp.ncbi.nih.gov/blast/db/FASTA/swissprot.gz`
16.`gunzip swissprot.gz`
17.`makeblastdb -dbtype prot -in swissprot -out swissprot`
Second，install blast into 'AnglesRefine/tools/blast'
18.`cd ..`
19.`wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz`
20.`tar -zxvf ncbi-blast-2.13.0+-x64-linux.tar.gz && mv ncbi-blast-2.13.0+ blast`
21.`export PATH=$PATH:AnglesRefine_PATH/tools/blast/bin`
22.`export BLASTDB=AnglesRefine_PATH/tools/db`
Third，install PSIPRED4.02 into 'AnglesRefine/tools/psipred'
23.`wget http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/psipred.4.02.tar.gz`
24.`tar -zxvf psipred.4.02.tar.gz`
25.`cd psipred/src`
26.`make`
27.`make install`
28.`cd ../BLAST+ && vim runpsipredplus`
29.Press "i" to enter editing mode and modify lines 14 and 17 to the following content:
    set dbname = AnglesRefine_PATH/tools/db/swissprot
    set ncbidir =AnglesRefine_PATH/tools/blast/bin
   Press ESC and ":wq!" to save changes.
P.S. AnglesRefine_PATH = the absolute path of your AnglesRefine package location (e.g., /home/userA/AnglesRefine)
```

## Run
```bash
conda activate AnglesRefine


$ python AnglesRefine.py -h
usage: AnglesRefine.py [-h] [-select SELECT] [-show] [-save_log] input output

positional arguments:
  input           path of starting model.
  output          path of output folder.

optional arguments:
  -h, --help      show this help message and exit
  -select SELECT  autunomous refine : refine the selected local structures to refine. e.g., -select 1,2 (default:
                  None)
  -show           show the information(sequence number,[startResidueIndex, endResidueIndex], length) of inconsistent
                  loacl structures identified to refine. (default: False)
  -save_log       save log file. (default: False)
  


suppose you have a starting model at `example/CASP11_T0797_4.pdb`, you can run the following command to generate the refined model.


## Default-Mode (Usage 1)

`python AnglesRefine.py example/CASP11_T0797_4.pdb output/`

or

## UserAutonomy-Mode (Usage 2)

## show analysis result of starting model：
`python  AnglesRefine.py example/CASP11_T0797_4.pdb output/ -show` 
## run according to the analysis result INFO:
`python  AnglesRefine.py example/CASP11_T0797_4.pdb output/ -select 1`

The refined model will be saved at `output/refined_CASP11_T0797_4.pdb`. You can use `-save_log` to save the refinement log file.


```


## References
Junyong Zhu, Renzhi Cao. AnglesRefine: refinement of 3D protein structures using Transformer based on torsion angles

# Contact
caora@plu.edu
