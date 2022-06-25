#!/usr/bin/env bash
logdir='./logs'
subject='neus_exp_a_11_2'
ptid='final_00300000.pt'
outname='outmesh.ply'
out=${logdir}'/'${subject}'/meshes/'${outname}
volume_size=2.5
load_pt=${logdir}'/'${subject}'/ckpts/'${ptid}
gpuid='7'
echo current path is $(pwd)
#Note. If want to use nerfpp, just add --isnerfpp
python -m tools.extract_surface --out ${out} --volume_size ${volume_size} --load_pt ${load_pt} --gpuid ${gpuid} #--isnerfpp