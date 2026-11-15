# Data sets

This directory contains the data sets of the experiments. The Python package does not include them.
Each data set has its own source and licence. Check the licence before you use a data set for other work.

| Directory | Content | Source | Licence |
|---|---|---|---|
| `cmu_mocap/35_01.npz` | Subject 35, trial 1: a normal walk and the same walk with swapped legs. | CMU Graphics Lab Motion Capture Database, <http://mocap.cs.cmu.edu/> | Free to use. See the database web site. |
| `cmu_mocap/07_*_joint_angles.npy` | Subject 7, walking sequences 1-12, as 62 joint angles. | CMU Graphics Lab Motion Capture Database | Free to use. See the database web site. |
| `cmu_mocap/07_joint_dims.json` | The number of angles of each joint of subject 7, in column order. | Derived from the sequences above. | As above. |
| `frey_faces/frey_rawface.mat` | 1965 grey-scale images of one face, each 28 x 20 pixels. | Brendan Frey, via Sam Roweis's data page. | Public research data set. |
| `colposcopy/*.csv` | Quality Assessment of Digital Colposcopies: 62 attributes and 7 expert assessments for 3 modalities. | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/384> | CC BY 4.0 |
| `skin_cancer_mnist/hmnist_8_8_L.csv` | Skin cancer MNIST: HAM10000 images, scaled to 8 x 8 luminance pixels. | Kaggle "Skin Cancer MNIST: HAM10000", from Tschandl et al. (2018), <https://doi.org/10.7910/DVN/DBW86T> | CC BY-NC 4.0 |
| `skin_cancer_mnist/*_conditions_data_*.npy` | Cached subsets of the file above: one image of each condition in each row. | Derived from the file above with the legacy random stream. | CC BY-NC 4.0 |

## Notes

- The PoseTrack and horse motion capture data of the paper are not available here.
- The three-condition cache files use the legacy prefix `two_conditions_data_`. The readers keep this name.
- To use a different directory, set the `DP_GP_LVM_DATA` environment variable.
