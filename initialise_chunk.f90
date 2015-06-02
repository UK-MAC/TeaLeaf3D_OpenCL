!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Driver for chunk initialisation.
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the user specified chunk initialisation kernel.

SUBROUTINE initialise_chunk(chunk)

  USE tea_module

  IMPLICIT NONE

  INTEGER :: chunk

  REAL(KIND=8) :: xmin,ymin,zmin,dx,dy,dz

  dx=(grid%xmax-grid%xmin)/float(grid%x_cells)
  dy=(grid%ymax-grid%ymin)/float(grid%y_cells)
  dz=(grid%zmax-grid%zmin)/float(grid%z_cells)

  xmin=grid%xmin+dx*float(chunks(chunk)%field%left-1)

  ymin=grid%ymin+dy*float(chunks(chunk)%field%bottom-1)

  zmin=grid%zmin+dz*float(chunks(chunk)%field%back-1)

  IF(use_opencl_kernels)THEN
    CALL initialise_chunk_kernel_ocl(xmin,ymin,zmin,dx,dy,dz)
  ENDIF


END SUBROUTINE initialise_chunk
