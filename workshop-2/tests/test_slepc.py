from petsc4py import PETSc
from slepc4py import SLEPc
print("PETSc:", PETSc.Sys.getVersion())
print("SLEPc:", ".".join(map(str, SLEPc.Sys.getVersion())))

