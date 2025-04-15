# ME700_Assignment4Part1

Problem related to **thermal expansion with linear elasticity** was implemented. The most of the code structure was borrowed from the [hyperelasticity example](https://github.com/Lejeune-Lab-Graduate-Course-Materials/fenicsX/blob/main/hyperelasticity_beam.py). The file of interest is named as '*thermal_expansion.py*'

## Instructions for running the script

### Installing FEniCSx on the SCC

```bash
module load miniconda
mamba create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
pip install PyYAML
```

### Running the script on VSCode Server
Launch VSCode Server.\
Open terminal in VSCode.\
Run the following commands one by one in the termainal to clone the repository (after moving to the desired directory):

```bash
git clone https://github.com/rishabh022298/ME700_Assignment4Part1.git
```
Change the folder:
```bash
cd ME700_Assignment4Part1
```
Activate fenicsx-env:
```bash
conda activate fenicsx-env
```
Run the script:
```bash
python thermal_expansion.py
```

Refresh the terminal to check the deformation.gif.
