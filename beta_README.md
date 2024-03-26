# Hello beta tests!

Currently the data folder structure isn't solidified since H1 isn't up on DANDI/EvalAI is just coming online. For now, the examples should run assuming the following data structures relative to root of this repo.

`data`
- `h1`
    - `held_in_calib`
    - `held_out_calib`
    - `minival`
    - `eval` (Note this is private data)
- `m1`
    - `sub-MonkeyL-held-in-calib`
    - `sub-MonkeyL-held-out-calib`
    - `minival` (Copy dandiset minival folder into this folder)
    - `eval` (Copy the ground truth held in and held out data into this folder)

H1 should unfold correctly just from unzipping the provided directory. M1 should work by renaming the provided dandiset to `m1` and `minival` folder inside, and then copying the provided eval data into this folder.
Each of the lowest level dirs holds the NWB files.