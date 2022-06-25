#!/usr/bin/env bash
pt_path="./weights/trained_models/trained_models/unisurf/unisurf_65_new/final_00450000.pt"
out_path="./out/surface/unisurf_65_new.ply"
echo current path is $(pwd)
python -m tools.extract_surface --load_pt ${pt_path} --N 512 --volume_size 2.0 --out ${out_path}