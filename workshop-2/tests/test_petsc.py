from petsc4py import PETSc
x = PETSc.Vec().createMPI(8)
x.set(1.0)
x.assemblyBegin(); x.assemblyEnd()
print("Rank", PETSc.COMM_WORLD.getRank(), "sum =", x.sum())

