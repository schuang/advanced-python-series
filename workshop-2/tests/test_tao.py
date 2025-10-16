from petsc4py import PETSc
tao = PETSc.TAO().create(PETSc.COMM_SELF)
tao.setType("lmvm")
print("TAO:", tao.getType(), "available")
tao.destroy()

